from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseDSparkWindow,
    dspark_completed_c4_positions,
    dspark_fixed_buffer_commit_indices,
)
from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend


def test_dspark_spec_reservation_allocates_logical_kv_only(monkeypatch):
    """Fixed C4 scratch, not the optimistic tail, owns physical C4 storage."""
    logical_locs = torch.arange(8, dtype=torch.int64)
    allocator = SimpleNamespace(
        page_size=4,
        alloc_logical_only=MagicMock(return_value=logical_locs),
        alloc_extend=MagicMock(),
    )
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.allocation.evict_from_tree_cache",
        lambda *_args, **_kwargs: None,
    )

    result = alloc_paged_token_slots_extend(
        tree_cache,
        prefix_lens=torch.tensor([0]),
        prefix_lens_cpu=torch.tensor([0]),
        seq_lens=torch.tensor([8]),
        seq_lens_cpu=torch.tensor([8]),
        last_loc=torch.tensor([-1]),
        extend_num_tokens=8,
        logical_only=True,
    )

    torch.testing.assert_close(result, logical_locs)
    allocator.alloc_logical_only.assert_called_once()
    allocator.alloc_extend.assert_not_called()


def test_dspark_page_multiple_commit_stays_token_granular(monkeypatch):
    """Sixteen scattered accepts must not be mistaken for one whole C4 page."""
    count = 16
    transfer = MagicMock()
    stream = SimpleNamespace(synchronize=MagicMock())
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.transfer_cache_dsv4_mla",
        transfer,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.device_module",
        SimpleNamespace(current_stream=lambda: stream),
    )
    host = SimpleNamespace(
        device_ptrs=object(),
        data_ptrs=object(),
        alloc_paged_token_slots=MagicMock(
            side_effect=[torch.tensor([i * 16]) for i in range(count)]
        ),
        backup_from_device_all_layer=MagicMock(),
    )
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.compress_ratio = 4
    coordinator.device = "cpu"
    coordinator.device_buffer_size = 4096
    coordinator.mem_pool_host = host
    coordinator.mem_pool_device = SimpleNamespace(
        full_to_hisparse_device_index_mapping=torch.arange(count, dtype=torch.int64)
    )
    coordinator.req_to_host_pool = object()
    coordinator.req_to_host_pool_allocated_len = object()
    coordinator.req_to_device_buffer = torch.arange(
        count * 4097, dtype=torch.int64
    ).view(count, 4097)
    coordinator.req_device_buffer_tokens = torch.full(
        (1, count, 4097), -1, dtype=torch.int32
    )
    coordinator._dspark_commit_done_event = SimpleNamespace(record=MagicMock())
    coordinator._has_pending_dspark_commit = False
    coordinator._dspark_scratch_compressed_locs = torch.arange(count)
    coordinator._dspark_scratch_valid_mapping = torch.ones(count, dtype=torch.bool)
    window = HiSparseDSparkWindow(
        compressed_locs=torch.arange(count),
        device_locs=torch.arange(100, 100 + count),
        previous_device_mapping=torch.zeros(count, dtype=torch.int64),
        req_offsets=list(range(count)),
        req_pool_indices_cpu=list(range(count)),
        c4_positions=[0] * count,
        prefix_lens_cpu=[0] * count,
    )
    coordinator._active_dspark_window = window

    coordinator.commit_dspark_verify_window(
        window, torch.full((count,), 4, dtype=torch.int64)
    )

    host.backup_from_device_all_layer.assert_not_called()
    first_transfer = transfer.call_args_list[0].kwargs
    torch.testing.assert_close(first_transfer["dst_indices"], torch.arange(count) * 16)
    torch.testing.assert_close(
        first_transfer["src_indices"], torch.arange(100, 100 + count)
    )
    stream.synchronize.assert_not_called()
    coordinator._dspark_commit_done_event.record.assert_called_once_with(stream)


def test_dspark_all_rejected_rollback_records_cross_stream_event(monkeypatch):
    """An all-rejected window still publishes asynchronous mapping rollback."""
    producer_stream = object()
    event = SimpleNamespace(record=MagicMock(), wait=MagicMock())
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.device_module",
        SimpleNamespace(current_stream=lambda: producer_stream),
    )
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.compress_ratio = 4
    coordinator.device = "cpu"
    coordinator.mem_pool_device = SimpleNamespace(
        full_to_hisparse_device_index_mapping=torch.tensor([0, 91, 0])
    )
    coordinator._dspark_scratch_compressed_locs = torch.tensor([1, -1])
    coordinator._dspark_scratch_valid_mapping = torch.tensor([False, True, False])
    coordinator._dspark_commit_done_event = event
    coordinator._has_pending_dspark_commit = False
    window = HiSparseDSparkWindow(
        compressed_locs=torch.tensor([1]),
        device_locs=torch.tensor([91]),
        previous_device_mapping=torch.tensor([7]),
        req_offsets=[0],
        req_pool_indices_cpu=[3],
        c4_positions=[2],
        prefix_lens_cpu=[8],
    )
    coordinator._active_dspark_window = window

    coordinator.commit_dspark_verify_window(window, torch.tensor([0]))

    assert coordinator.mem_pool_device.full_to_hisparse_device_index_mapping[1] == 7
    assert not coordinator._dspark_scratch_valid_mapping[1]
    assert coordinator._active_dspark_window is None
    assert coordinator._has_pending_dspark_commit
    event.record.assert_called_once_with(producer_stream)

    consumer_stream = object()
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.device_module",
        SimpleNamespace(current_stream=lambda: consumer_stream),
    )
    coordinator.is_dsv4_hisparse = True
    coordinator.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((4, 4), dtype=torch.int64)
    )
    coordinator.prepare_dspark_verify_window(
        req_pool_indices=torch.tensor([3]),
        prefix_lens=torch.tensor([0]),
        verify_width=1,
        req_pool_indices_cpu=torch.tensor([3]),
        prefix_lens_cpu=torch.tensor([0]),
    )

    event.wait.assert_called_once_with(consumer_stream)
    assert not coordinator._has_pending_dspark_commit


@pytest.mark.parametrize(
    ("prefix_len", "expected"),
    [(8, [2]), (9, [2]), (10, [2, 3]), (11, [2, 3])],
)
def test_dspark_c4_scratch_crosses_alignment(prefix_len, expected):
    """Every prefix alignment must bind each C4 writer completed by six verify rows."""
    assert dspark_completed_c4_positions(prefix_len, verify_width=6) == expected


def test_dspark_long_commit_has_unique_reserved_destination():
    """Two accepted long C4s from one request must copy only the newest one."""
    assert dspark_fixed_buffer_commit_indices(
        accepted=[0, 1, 2],
        req_offsets=[0, 0, 1],
        req_pool_indices_cpu=[7, 9],
        c4_positions=[4096, 4097, 4098],
        device_buffer_size=4096,
    ) == [1, 2]


def test_dspark_graph_scratch_metadata_updates_in_place():
    """Replay must consume updated metadata without changing captured addresses."""
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator._dspark_scratch_compressed_locs = torch.full(
        (4,), -1, dtype=torch.int64
    )
    coordinator._dspark_scratch_valid_mapping = torch.zeros(16, dtype=torch.bool)
    compressed = torch.tensor([10, 11, 12], dtype=torch.int64)
    swapped = torch.tensor([100, 101, 102], dtype=torch.int32)
    writers = torch.tensor([200, 201, 202], dtype=torch.int32)

    capture_result = coordinator.select_dspark_scratch_locs(
        compressed, swapped, writers
    )
    torch.testing.assert_close(capture_result, swapped)

    metadata_ptr = coordinator._dspark_scratch_compressed_locs.data_ptr()
    coordinator._dspark_scratch_compressed_locs[:2].copy_(compressed[[0, 2]])
    coordinator._dspark_scratch_valid_mapping[compressed[[0, 2]]] = True
    replay_result = coordinator.select_dspark_scratch_locs(compressed, swapped, writers)

    assert coordinator._dspark_scratch_compressed_locs.data_ptr() == metadata_ptr
    torch.testing.assert_close(
        replay_result, torch.tensor([200, 101, 202], dtype=torch.int32)
    )


def test_dspark_prepare_translates_with_coordinator_device_pool():
    """Coordinator stores the DSV4 C4 pool as mem_pool_device."""
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.is_dsv4_hisparse = True
    coordinator.compress_ratio = 4
    coordinator.device = "cpu"
    coordinator._has_pending_dspark_commit = False
    coordinator.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[0, 1, 2, 15]], dtype=torch.int64)
    )
    mapping = torch.zeros(16, dtype=torch.int64)
    translate = MagicMock(return_value=torch.tensor([5], dtype=torch.int64))
    coordinator.mem_pool_device = SimpleNamespace(
        translate_loc_from_full_to_compressed=translate,
        full_to_hisparse_device_index_mapping=mapping,
    )
    coordinator._dspark_scratch_device_locs = torch.tensor([91], dtype=torch.int64)
    coordinator._dspark_scratch_compressed_locs = torch.full(
        (1,), -1, dtype=torch.int64
    )
    coordinator._dspark_scratch_valid_mapping = torch.zeros(16, dtype=torch.bool)
    coordinator._active_dspark_window = None

    window = coordinator.prepare_dspark_verify_window(
        req_pool_indices=torch.tensor([0]),
        prefix_lens=torch.tensor([3]),
        verify_width=1,
        req_pool_indices_cpu=torch.tensor([0]),
        prefix_lens_cpu=torch.tensor([3]),
    )

    translate.assert_called_once()
    torch.testing.assert_close(window.compressed_locs, torch.tensor([5]))
    assert mapping[5] == 91
    assert coordinator._dspark_scratch_valid_mapping[5]


def _coordinator_with_recording_kernel():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.top_k = 2
    coordinator.device = "cpu"
    calls = []

    def run(
        self,
        *,
        req_pool_indices,
        compressed_seq_lens,
        top_k_result,
        layer_id,
        verify_width,
        output_buffer,
    ):
        calls.append(
            (
                req_pool_indices.tolist(),
                compressed_seq_lens.tolist(),
                top_k_result.tolist(),
                layer_id,
                verify_width,
            )
        )
        assert compressed_seq_lens.is_contiguous()
        assert top_k_result.is_contiguous()
        assert output_buffer.is_contiguous()
        output_buffer.copy_(top_k_result.to(torch.int32) + 100)
        return output_buffer

    coordinator._run_swap_in_verify_window = MethodType(run, coordinator)
    return coordinator, calls


@pytest.mark.parametrize(
    ("verify_lens", "seq_lens", "expected_launches"),
    [
        (None, [10, 10, 20, 20], [[7, 9]]),
        ([1, 3], [10, 20], [[7, 9]]),
        (None, [[10, 10], [20, 20]], [[7, 9]]),
    ],
)
def test_dspark_swap_in_is_request_major(verify_lens, seq_lens, expected_launches):
    """One planner owns every ordered step of each request in one launch."""
    coordinator, calls = _coordinator_with_recording_kernel()
    top_k = torch.arange(8, dtype=torch.int64).view(4, 2)
    output = torch.full((4, 2), -1, dtype=torch.int32)

    actual = coordinator.swap_in_selected_pages_spec(
        req_pool_indices=torch.tensor([7, 9]),
        compressed_seq_lens=torch.tensor(seq_lens),
        top_k_result=top_k,
        layer_id=3,
        verify_lens_cpu=verify_lens,
        output_buffer=output,
    )

    assert [call[0] for call in calls] == expected_launches
    expected_width = 3 if verify_lens == [1, 3] else 2
    assert calls[0][4] == expected_width
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, top_k.to(torch.int32) + 100)


def test_dspark_swap_in_ignores_graph_padding_rows():
    """Padding rows must not launch a block or mutate request slot zero."""
    coordinator, calls = _coordinator_with_recording_kernel()
    top_k = torch.arange(12, dtype=torch.int64).view(6, 2)
    output = torch.empty((6, 2), dtype=torch.int32)

    actual = coordinator.swap_in_selected_pages_spec(
        req_pool_indices=torch.tensor([7, 9]),
        compressed_seq_lens=torch.tensor([10, 20]),
        top_k_result=top_k,
        layer_id=3,
        verify_lens_cpu=[1, 3],
        output_buffer=output,
    )

    assert [call[0] for call in calls] == [[7, 9]]
    assert calls[0][4] == 3
    torch.testing.assert_close(actual[:4], top_k[:4].to(torch.int32) + 100)
    torch.testing.assert_close(actual[4:], torch.full((2, 2), -1, dtype=torch.int32))


def test_dspark_verify_window_plans_and_copies_once(monkeypatch):
    """The complete verify window produces one ordered plan and one copy."""
    planner = MagicMock()
    copier = MagicMock()
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.load_cache_to_device_buffer_dsv4_mla",
        planner,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.copy_cache_planned_mla",
        copier,
    )
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.is_dsv4_hisparse = True
    coordinator._speculative_verify_width = 3
    coordinator.top_k = 2
    coordinator.device_buffer_size = 6
    coordinator.swap_in_block_size = 256
    coordinator.item_size_bytes = 16
    coordinator.skip_io = False
    coordinator._init_shared_index_prefetch(
        shared_index_layers=None, layer_num=1, max_num_req_slots=4
    )
    assert not coordinator.enable_prefetch
    coordinator.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    coordinator._spec_miss_src = torch.empty((4, 6), dtype=torch.int64)
    coordinator._spec_miss_dst = torch.empty((4, 6), dtype=torch.int32)
    coordinator._spec_miss_count = torch.zeros(4, dtype=torch.int32)
    coordinator.req_device_buffer_tokens = torch.empty((1, 4, 7), dtype=torch.int32)
    coordinator.req_to_host_pool = torch.empty((4, 16), dtype=torch.int64)
    coordinator.req_device_buffer_token_locs = torch.empty((1, 4, 7), dtype=torch.int32)
    coordinator.lru_slots = torch.empty((1, 4, 6), dtype=torch.int16)
    coordinator.mem_pool_host = SimpleNamespace(kv_buffer=[torch.empty(1)])
    coordinator.mem_pool_device = SimpleNamespace(kv_buffer=[torch.empty(1)])
    top_k = torch.arange(12, dtype=torch.int32).view(6, 2)
    output = torch.full_like(top_k, -1)

    actual = coordinator._run_swap_in_verify_window(
        req_pool_indices=torch.tensor([1, 3]),
        compressed_seq_lens=torch.tensor([10, 10, 10, 20, 20, 20]),
        top_k_result=top_k,
        layer_id=0,
        verify_width=3,
        output_buffer=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    planner.assert_called_once()
    copier.assert_called_once()
    assert planner.call_args.kwargs["verify_width"] == 3
    assert planner.call_args.kwargs["skip_io"]
    assert planner.call_args.kwargs["miss_src"].shape == (2, 6)
    assert copier.call_args.kwargs["num_blocks"] == 4
    assert copier.call_args.kwargs["miss_src"].data_ptr() == (
        planner.call_args.kwargs["miss_src"].data_ptr()
    )


def test_dspark_swap_in_rejects_inconsistent_ragged_geometry():
    """Bad compact geometry must fail instead of associating Top-K with another request."""
    coordinator, _ = _coordinator_with_recording_kernel()
    with pytest.raises(ValueError, match="geometry does not match"):
        coordinator.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([7, 9]),
            compressed_seq_lens=torch.tensor([10, 20]),
            top_k_result=torch.zeros((4, 2), dtype=torch.int64),
            layer_id=3,
            verify_lens_cpu=[2, 3],
        )


def test_dspark_swap_in_rejects_inconsistent_sequence_length_geometry():
    """A short seq-len tensor would make the CUDA kernel read past its allocation."""
    coordinator, _ = _coordinator_with_recording_kernel()
    with pytest.raises(ValueError, match="sequence lengths"):
        coordinator.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([7, 9]),
            compressed_seq_lens=torch.tensor([10, 20, 30]),
            top_k_result=torch.zeros((4, 2), dtype=torch.int64),
            layer_id=3,
        )
