"""CPU checks for hash workspace sizing, FFI geometry, and memory planning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache import hisparse_window_hash as ops
from sglang.srt.mem_cache.hisparse_window_hash import HiSparseWindowHashLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("num_reqs", [2, 11])
def test_state_budget_matches_allocated_storage(num_reqs):
    layout = HiSparseWindowHashLayout(hot_slots=8, verify_width=3, top_k=2)
    states = ops.allocate_window_hash_states(
        layout, num_layers=2, num_req_slots=num_reqs, device="cpu"
    )
    allocated = sum(t.untyped_storage().nbytes() for state in states for t in state)
    assert layout.fixed_state_bytes(num_layers=2, num_req_slots=num_reqs) == allocated
    assert states[0].cache_policy.stride(0) >= num_reqs
    assert states[0].scratch_state.stride(0) % 2 == 0
    assert states[0].scratch_locs.numel() == 0


@pytest.mark.parametrize(
    "hot,width,k", [(1, 1, 1), (8, 0, 2), (8, 3, 3), (32769, 1, 1)]
)
def test_layout_rejects_union_that_cannot_remain_resident(hot, width, k):
    with pytest.raises(ValueError):
        HiSparseWindowHashLayout(hot, width, k)


def _plan_args():
    layout = HiSparseWindowHashLayout(8, 3, 2)
    return {
        "top_k_tokens": torch.zeros((2, 3, 2), dtype=torch.int32),
        "top_k_device_locs": torch.full((2, 3, 2), -1, dtype=torch.int32),
        "device_buffer_tokens": torch.full((4, 9), -1, dtype=torch.int32),
        "device_buffer_locs": torch.arange(36, dtype=torch.int32).view(4, 9),
        "host_cache_locs": torch.arange(128, dtype=torch.int64).view(4, 32),
        "host_cache": torch.empty(0),
        "device_buffer": torch.empty(0),
        "req_pool_indices": torch.tensor([3, 1], dtype=torch.int32),
        "seq_lens": torch.tensor([9, 15], dtype=torch.int64),
        "state": ops.allocate_window_hash_states(
            layout, num_layers=1, num_req_slots=4, device="cpu"
        )[0],
        "num_real_reqs": torch.tensor([2], dtype=torch.int32),
        "miss_src": torch.empty((2, 6), dtype=torch.int64),
        "miss_dst": torch.empty((2, 6), dtype=torch.int32),
        "miss_count": torch.zeros(2, dtype=torch.int32),
    }


@pytest.mark.parametrize("per_row", [False, True])
def test_ffi_preserves_request_major_lengths_and_output_addresses(monkeypatch, per_row):
    args = _plan_args()
    if per_row:
        args["seq_lens"] = torch.tensor([9, 10, 11, 15, 16, 17], dtype=torch.int64)
    planner = MagicMock()
    monkeypatch.setattr(
        ops, "_jit_window_hash_module", lambda *_: SimpleNamespace(plan_window=planner)
    )
    ops.plan_cache_window_hash_mla(**args)
    ffi_args = planner.call_args.args
    assert ffi_args[6].data_ptr() == args["top_k_device_locs"].data_ptr()
    torch.testing.assert_close(ffi_args[7], torch.tensor([3, 1], dtype=torch.int64))
    expected = [9, 10, 11, 15, 16, 17] if per_row else [9, 9, 9, 15, 15, 15]
    torch.testing.assert_close(ffi_args[8], torch.tensor(expected, dtype=torch.int32))


@pytest.mark.parametrize(
    "bad", ["short_plan", "strided_plan", "strided_rows", "bad_lengths", "bad_output"]
)
def test_invalid_geometry_fails_before_jit(monkeypatch, bad):
    args = _plan_args()
    if bad == "short_plan":
        args["miss_src"] = args["miss_src"][:, :5]
    elif bad == "strided_plan":
        args["miss_dst"] = torch.empty((2, 12), dtype=torch.int32)[:, ::2]
    elif bad == "strided_rows":
        args["top_k_tokens"] = torch.zeros((2, 3, 4), dtype=torch.int32)[:, :, ::2]
    elif bad == "bad_lengths":
        args["seq_lens"] = torch.ones(4, dtype=torch.int64)
    else:
        args["top_k_device_locs"] = torch.empty((2, 2, 2), dtype=torch.int32)
    jit = MagicMock()
    monkeypatch.setattr(ops, "_jit_window_hash_module", jit)
    with pytest.raises(ValueError):
        ops.plan_cache_window_hash_mla(**args)
    jit.assert_not_called()


def test_memory_budget_solves_hash_state_and_request_capacity_together():
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

    cfg = DSV4PoolConfigurator.__new__(DSV4PoolConfigurator)
    cfg.hisparse_window_hash_layout = HiSparseWindowHashLayout(8, 3, 2)
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
    cost = tokens * 16 + cfg._get_window_hash_fixed_bytes_for_token_capacity(tokens)
    next_cost = (
        tokens + 128
    ) * 16 + cfg._get_window_hash_fixed_bytes_for_token_capacity(tokens + 128)
    assert 0 < cost <= available < next_cost
