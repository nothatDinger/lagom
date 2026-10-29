from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.dsv4.lagom import resolve_lagom_topk


def make_batch():
    return SimpleNamespace(
        req_pool_indices=torch.tensor([3, 8]),
        req_pool_indices_cpu=torch.tensor([3, 8]),
        lagom_decode_width=3,
        lagom_prefix_lens_cpu=torch.tensor([1024, 2048]),
    )


def test_lagom_csa_queues_each_requests_suffix_as_logical_ids():
    batch = make_batch()
    logical_ids = torch.arange(2 * 5 * 4).reshape(10, 4)
    physical_locs = torch.full((10, 4), 500, dtype=torch.int32)
    coordinator = SimpleNamespace(
        swap_in_lagom_pages=Mock(return_value=physical_locs),
        queue_lagom_prefetch=Mock(),
    )
    result = resolve_lagom_topk(
        coordinator=coordinator,
        forward_batch=batch,
        layer_id=7,
        raw_indices=logical_ids,
        compressed_seq_lens=torch.arange(10),
        output_buffer=physical_locs,
    )
    assert result is physical_locs
    swap = coordinator.swap_in_lagom_pages.call_args.kwargs
    assert swap["decode_width"] == 3  # anchor + W; P never demand-loads
    assert swap["top_k_result"] is logical_ids
    queued = coordinator.queue_lagom_prefetch.call_args.kwargs
    torch.testing.assert_close(
        queued["predicted_c4_entries"], logical_ids.reshape(2, 5, 4)[:, 3:]
    )
    assert queued["source_committed_lens_cpu"] is batch.lagom_prefix_lens_cpu
    assert queued["req_pool_indices"] is batch.req_pool_indices
    assert queued["layer_id"] == 7


@pytest.mark.parametrize(
    "width,rows,prefix", [(5, 10, [1, 2]), (3, 9, [1, 2]), (3, 10, None)]
)
def test_invalid_lagom_geometry_cannot_publish_prefetch(width, rows, prefix):
    batch = make_batch()
    batch.lagom_decode_width = width
    batch.lagom_prefix_lens_cpu = prefix
    coordinator = Mock()
    with pytest.raises(ValueError):
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=batch,
            layer_id=0,
            raw_indices=torch.zeros((rows, 4), dtype=torch.int32),
            compressed_seq_lens=torch.zeros(rows),
            output_buffer=torch.empty((rows, 4), dtype=torch.int32),
        )
    coordinator.swap_in_lagom_pages.assert_not_called()
    coordinator.queue_lagom_prefetch.assert_not_called()


def test_graph_capture_retains_only_tensor_descriptors_until_live_replay():
    batch = make_batch()
    batch.is_lagom_graph_capture = True
    batch.lagom_prefix_lens_cpu = None
    coordinator = Mock()
    # Warmup and capture have different allocations. Only the latest tensor
    # may be published when a real batch replays this graph bucket.
    for fill in (11, 22):
        raw = torch.full((10, 4), fill, dtype=torch.int32)
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=batch,
            layer_id=1,
            raw_indices=raw,
            compressed_seq_lens=torch.zeros(10),
            output_buffer=torch.empty_like(raw),
        )
    coordinator.queue_lagom_prefetch.assert_not_called()
    captured_coordinator, predictions = batch._lagom_pending_prefetch[1]
    assert captured_coordinator is coordinator
    torch.testing.assert_close(predictions, raw.view(2, 5, 4)[:, 3:])
    assert predictions.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
