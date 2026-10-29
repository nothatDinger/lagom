"""CPU checks for the extracted fused swap-in's layout, FFI and serving dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache import hisparse_spec as ops
from sglang.srt.mem_cache.hisparse_spec import (
    HiSparseSpecLayout,
    validate_pr_fused_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def settings():
    return dict(
        cuda=True,
        dsv4=True,
        speculative=True,
        width=6,
        top_k=512,
        hot=3072,
        page_size=64,
        force_h2d=False,
        skip_io=False,
    )


@pytest.mark.parametrize("force_h2d", [False, True])
def test_request_tail_keeps_canonical_and_provisional_slots_fixed(force_h2d):
    assert validate_pr_fused_config(**dict(settings(), force_h2d=force_h2d)) == dict(
        active_slots=3075, scratch_offset=3075, scratch_slots=61
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("cuda", False),
        ("dsv4", False),
        ("speculative", False),
        ("skip_io", True),
        ("width", 0),
        ("hot", 2560),
        ("page_size", 3),
        ("prefetcher", "previous"),
        ("width", 17),
    ],
)
def test_unsupported_configurations_are_rejected(key, value):
    config = settings()
    config[key] = value
    with pytest.raises(ValueError):
        validate_pr_fused_config(**config)


@pytest.mark.parametrize("num_reqs", [4, 137])
def test_state_budget_matches_allocations_and_kv_scratch_aliases_request_tail(num_reqs):
    layout = HiSparseSpecLayout(128, 6, 16)
    locs = torch.arange(2 * num_reqs * 192, dtype=torch.int32).view(2, num_reqs, 192)
    states = ops.allocate_hisparse_spec_states(
        layout, device_buffer_locs=locs, scratch_offset=131
    )
    metadata_bytes = sum(
        tensor.untyped_storage().nbytes()
        for state in states
        for tensor in (state.cache_index, state.cache_policy, state.scratch_state)
    )
    assert (
        layout.fixed_state_bytes(num_layers=2, num_req_slots=num_reqs) == metadata_bytes
    )
    for layer, state in enumerate(states):
        assert state.cache_policy.stride(0) >= num_reqs
        assert state.scratch_state.stride(0) % 2 == 0
        assert state.scratch_locs.shape == (num_reqs, 61)
        assert state.scratch_locs.stride(0) == 192
        assert state.scratch_locs.data_ptr() == locs[layer, :, 131:].data_ptr()


def _coordinator(monkeypatch, *, force_h2d=False):
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    fused, separate, init = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.load_cache_to_device_buffer_spec_mla",
        fused,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.copy_cache_planned_mla", separate
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.initialize_hisparse_spec_state", init
    )
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c._pr_verify_layout = dict(active_slots=131, scratch_offset=131, scratch_slots=61)
    c._speculative_verify_width = 6
    c.force_h2d_each_step = force_h2d
    c.top_k, c.device_buffer_size, c.item_size_bytes, c.skip_io = 16, 128, 584, False
    c.req_device_buffer_tokens = torch.full((1, 4, 192), 17, dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.arange(4 * 192, dtype=torch.int32).view(
        1, 4, 192
    )
    c.lru_slots = torch.zeros(1, 4, 128, dtype=torch.int16)
    c._lru_init = torch.arange(128, dtype=torch.int16)
    c._spec_states = [object()]
    c.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    c.req_to_host_pool = torch.zeros(4, 1024, dtype=torch.int64)
    c.mem_pool_host = SimpleNamespace(kv_buffer=[torch.empty(1)])
    c.mem_pool_device = SimpleNamespace(kv_buffer=[torch.empty(1)])
    return c, fused, separate, init


@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("per_row", [False, True])
@pytest.mark.parametrize("force_h2d", [False, True])
def test_service_dispatch_uses_fused_io_and_preserves_graph_buffers(
    monkeypatch, request_dtype, per_row, force_h2d
):
    c, fused, separate, init = _coordinator(monkeypatch, force_h2d=force_h2d)
    locs_before = c.req_device_buffer_token_locs.clone()
    top_k = torch.arange(192, dtype=torch.int32).view(12, 16)
    result = torch.empty_like(top_k)
    lengths = torch.tensor([1024] * 6 + [2048] * 6 if per_row else [1024, 2048])
    actual = c._run_swap_in_verify_window(
        req_pool_indices=torch.tensor([2, 0], dtype=request_dtype),
        compressed_seq_lens=lengths,
        top_k_result=top_k,
        layer_id=0,
        verify_width=6,
        output_buffer=result,
    )
    fused.assert_called_once()
    separate.assert_not_called()
    if force_h2d:
        init.assert_called_once()
    else:
        init.assert_not_called()
    args = fused.call_args.kwargs
    assert args["req_pool_indices"].dtype == torch.int64
    assert args["device_buffer_locs"].shape == (4, 131)
    assert args["device_buffer_locs"].stride(0) == 192
    assert args["seq_lens"].tolist() == [1024] * 6 + [2048] * 6
    assert args["seq_lens"].dtype == torch.int32
    assert (
        args["top_k_device_locs"].data_ptr() == actual.data_ptr() == result.data_ptr()
    )
    assert args["top_k_tokens"].data_ptr() == top_k.data_ptr()
    assert args["transfer_kind"] == "dsv4_paged"
    assert "miss_src" not in args
    assert (c.req_device_buffer_tokens[0, [2, 0]] == (-1 if force_h2d else 17)).all()
    assert (c.req_device_buffer_tokens[0, [1, 3]] == 17).all()
    torch.testing.assert_close(c.req_device_buffer_token_locs, locs_before)


def test_initializer_normalizes_request_ids_before_ffi(monkeypatch):
    init = MagicMock()
    monkeypatch.setattr(
        ops,
        "_jit_spec_state_module",
        lambda *_: SimpleNamespace(initialize_hisparse_spec_state=init),
    )
    state = ops.allocate_hisparse_spec_states(
        HiSparseSpecLayout(128, 6, 16),
        device_buffer_locs=torch.empty((1, 4, 192), dtype=torch.int32),
        scratch_offset=131,
    )[0]
    ops.initialize_hisparse_spec_state(
        device_buffer_tokens=torch.empty((4, 192), dtype=torch.int32),
        req_pool_indices=torch.tensor([2, 0], dtype=torch.int32),
        state=state,
    )
    assert init.call_args.args[1].dtype == torch.int64
    assert init.call_args.args[1].tolist() == [2, 0]


def test_single_token_fallback_uses_the_same_fused_state():
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator._spec_states = [object()]
    coordinator.top_k = 2
    coordinator.top_k_device_locs_buffer = torch.zeros((2, 2), dtype=torch.int32)
    coordinator._run_swap_in_verify_window = MagicMock()
    real = torch.tensor([1], dtype=torch.int32)
    coordinator._run_swap_in_kernel(
        torch.tensor([3, 1]),
        torch.tensor([10, 20]),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        0,
        num_real_reqs=real,
    )
    call = coordinator._run_swap_in_verify_window.call_args.kwargs
    assert call["verify_width"] == 1
    assert call["num_real_reqs"] is real
    assert (
        call["output_buffer"].data_ptr()
        == coordinator.top_k_device_locs_buffer.data_ptr()
    )


def test_memory_budget_solves_fused_state_and_request_capacity_together():
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

    cfg = DSV4PoolConfigurator.__new__(DSV4PoolConfigurator)
    cfg.hisparse_spec_layout = HiSparseSpecLayout(8, 3, 2)
    cfg.requested_max_running_requests_per_worker = None
    cfg.context_len = 4096
    cfg.disaggregation_mode = "decode"
    cfg.disaggregation_decode_extra_slots = 2
    cfg.num_layers_ca4 = 2
    cfg.num_layers_ca128 = 0
    cfg.bytes_per_full_token = 16
    cfg.swa_ratio = 1
    cfg.swa_page_size = 128
    cfg.c4_shrink_factor = 2
    cfg.c4_ring_size = 1
    available = 500_000
    result = cfg.calculate_pool_sizes(available, 128)
    tokens = result.max_total_num_tokens
    cost = tokens * 16 + cfg._get_spec_swapin_fixed_bytes_for_token_capacity(tokens)
    next_cost = (
        tokens + 128
    ) * 16 + cfg._get_spec_swapin_fixed_bytes_for_token_capacity(tokens + 128)
    assert 0 < cost <= available < next_cost


@pytest.mark.parametrize("config", [None, "{}", '{"force_h2d_each_step":false}'])
def test_hisparse_defaults_preserve_residency_without_lagom(config):
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config

    parsed = parse_hisparse_config(SimpleNamespace(hisparse_config=config))
    assert parsed.force_h2d_each_step is False
    assert parsed.prefetcher is None
    validate_pr_fused_config(**dict(settings(), force_h2d=parsed.force_h2d_each_step))


@pytest.mark.parametrize("widths", [(6, 1, 4, 6), (3, 5, 2, 6)])
def test_preserved_fused_state_keeps_workspace_layout_across_verify_and_fallback(
    monkeypatch, widths
):
    c, fused, _separate, init = _coordinator(monkeypatch)

    def resolve(**kwargs):
        output = kwargs["top_k_device_locs"]
        output.copy_(torch.arange(output.numel(), dtype=torch.int32).view_as(output))

    fused.side_effect = resolve
    for width in widths:
        topk = torch.arange(2 * width * c.top_k, dtype=torch.int32).view(-1, c.top_k)
        output = torch.empty_like(topk)
        c._run_swap_in_verify_window(
            req_pool_indices=torch.tensor([2, 0]),
            compressed_seq_lens=torch.tensor([200, 300]),
            top_k_result=topk,
            layer_id=0,
            verify_width=width,
            output_buffer=output,
        )
        call = fused.call_args.kwargs
        assert call["state"] is c._spec_states[0]
        assert call["top_k_tokens"].shape == (2, 6, c.top_k)
        torch.testing.assert_close(
            call["top_k_tokens"][:, :width].reshape_as(topk), topk
        )
        assert (call["top_k_tokens"][:, width:] == -1).all()
        assert (call["seq_lens"].view(2, 6)[:, width:] == 0).all()
        expected = torch.arange(2 * 6 * c.top_k, dtype=torch.int32).view(2, 6, c.top_k)
        torch.testing.assert_close(output.view(2, width, c.top_k), expected[:, :width])
    init.assert_not_called()
    assert (c.req_device_buffer_tokens == 17).all()


@pytest.mark.parametrize("compressed_len", [3, 128, 129])
@pytest.mark.parametrize("force_h2d", [False, True])
def test_short_prefix_residency_keeps_canonical_slots_for_future_c4_commit(
    monkeypatch, compressed_len, force_h2d
):
    c, _fused, _separate, _init = _coordinator(monkeypatch, force_h2d=force_h2d)
    c.is_dsv4_hisparse = True
    c.padded_buffer_size = 192
    req_slot = 1
    req = SimpleNamespace(
        req_pool_idx=req_slot, extend_range=SimpleNamespace(end=compressed_len * 4)
    )
    c.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(4 * 1024).view(4, 1024)
    )
    c.mem_pool_device.translate_loc_from_full_to_compressed = lambda x: x[3::4] // 4
    buffer = torch.arange(10, 202, dtype=torch.int64)
    c.token_to_kv_pool_allocator = SimpleNamespace(
        alloc_device_buffer=MagicMock(return_value=buffer)
    )
    c.req_to_device_buffer = torch.zeros((4, 192), dtype=torch.int64)
    c.req_device_buffer_size = torch.zeros(4, dtype=torch.int64)
    c._device_buffer_arange_i32 = torch.arange(128, dtype=torch.int32)
    c._initialize_spec_request = MagicMock()
    if force_h2d and compressed_len <= 128:
        with pytest.raises(
            ValueError, match="force_h2d_each_step=true requires long-context"
        ):
            c.alloc_device_buffer(req)
        c.token_to_kv_pool_allocator.alloc_device_buffer.assert_not_called()
        return
    c.alloc_device_buffer(req)
    assert c.req_device_buffer_size[req_slot].item() == 192
    torch.testing.assert_close(
        c.req_device_buffer_tokens[0, req_slot, :128],
        torch.arange(128, dtype=torch.int32),
    )
    torch.testing.assert_close(
        c.req_device_buffer_token_locs[0, req_slot], buffer.to(torch.int32)
    )
    c._initialize_spec_request.assert_called_once_with(req_slot)
