"""Lagom per-layer stream ordering, stable plans, and historical KV ownership."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse_spec import LagomCachePlan
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.hisparse_prefetcher import HiSparsePrefetchStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def coordinator(monkeypatch):
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.prefetcher_name = "lagom"
    c.prefetcher = SimpleNamespace(logical_entries=2, stats=HiSparsePrefetchStats())
    c.top_k, c._lagom_prediction_steps, c.compress_ratio, c.device = 2, 2, 4, "cpu"
    c._speculative_verify_width = 5
    c._pr_verify_layout = {"active_slots": 12}
    c._spec_states = [object(), object()]
    c.req_device_buffer_tokens = torch.full((2, 4, 16), -1, dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.arange(128, dtype=torch.int32).view(2, 4, 16)
    c.req_to_host_pool = torch.zeros((4, 100), dtype=torch.int64)
    c.mem_pool_host = SimpleNamespace(kv_buffer=[object(), object()])
    c.mem_pool_device = SimpleNamespace(kv_buffer=[object(), object()])
    c.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    c._lagom_real_counts = [torch.zeros_like(c.num_real_reqs) for _ in range(2)]
    c._lagom_plans = [
        LagomCachePlan(
            torch.zeros((4, 10), dtype=torch.int64),
            torch.zeros((4, 10), dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros((4, 10), dtype=torch.int32),
        )
        for _ in range(2)
    ]
    c._active_dspark_window = object()  # Prefetch must run during this transaction.
    c.item_size_bytes = 584
    c.order = []

    def plan_ready(layer, batch, transfer, *, capturing):
        c.order.append("ready")
        if not capturing:
            transfer()

    c._lagom_graphs = SimpleNamespace(
        consume_layer=MagicMock(side_effect=lambda _layer: c.order.append("consume")),
        plan_ready=MagicMock(side_effect=plan_ready),
        synchronize=MagicMock(),
        fail=MagicMock(),
    )
    c.plan_kernel = MagicMock(side_effect=lambda **kw: c.order.append("plan"))
    c.copy_kernel = MagicMock(
        side_effect=lambda _plan, **kw: c.order.append(
            "refill" if kw["demand"] else "prefetch"
        )
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.plan_lagom_cache", c.plan_kernel
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.copy_lagom_cache_plan", c.copy_kernel
    )
    return c


def run_layer(c, *, layer=0, capturing=False):
    raw = torch.tensor(
        [
            [[0, 1], [2, 3], [3, 4], [0, 2], [3, 4]],
            [[0, 1], [2, 3], [5, 6], [1, 6], [-1, 7]],
        ],
        dtype=torch.int32,
    )
    original = raw.clone()
    c.swap_in_lagom_pages(
        req_pool_indices=torch.tensor([2, 0]),
        compressed_seq_lens=torch.tensor([4, 4, 4, 5, 5, 7, 7, 7, 8, 8]),
        source_prefix_lens=torch.tensor([13, 24]),
        top_k_result=raw.view(-1, 2),
        layer_id=layer,
        decode_width=3,
        capturing=capturing,
    )
    torch.testing.assert_close(raw, original)


def test_prefetch_is_launched_before_current_refill_while_transaction_active(
    monkeypatch,
):
    c = coordinator(monkeypatch)
    run_layer(c)
    assert c.order == ["consume", "plan", "ready", "prefetch", "refill"]
    assert c._active_dspark_window is not None
    assert [call.kwargs["demand"] for call in c.copy_kernel.call_args_list] == [
        False,
        True,
    ]
    # Both copies consume the same immutable union plan; the kernel's ownership
    # mask assigns shared misses to demand and makes the destinations disjoint.
    assert (
        c.copy_kernel.call_args_list[0].args[0]
        is c.copy_kernel.call_args_list[1].args[0]
    )


def test_only_prediction_rows_are_limited_to_preverify_committed_history(monkeypatch):
    c = coordinator(monkeypatch)
    c.prefetcher.logical_entries = 1
    run_layer(c)
    call = c.plan_kernel.call_args.kwargs
    assert call["top_k_tokens"][:, 3:].tolist() == [
        [[0, 2], [-1, -1]],
        [[1, -1], [-1, -1]],
    ]
    assert call["top_k_tokens"][:, :3].tolist() == [
        [[0, 1], [2, 3], [3, 4]],
        [[0, 1], [2, 3], [5, 6]],
    ]
    assert call["seq_lens"].view(2, 5).tolist() == [[4, 4, 4, 3, 3], [7, 7, 7, 6, 6]]
    assert call["prefetch_top_k"] == 1
    assert call["req_pool_indices"].tolist() == [2, 0]


def test_capture_records_ready_and_keeps_foreground_free_of_prefetch_copy(monkeypatch):
    c = coordinator(monkeypatch)
    run_layer(c, capturing=True)
    assert c.order == ["consume", "plan", "ready", "refill"]
    assert c._lagom_graphs.plan_ready.call_args.kwargs["capturing"]
    c._lagom_graphs.synchronize.assert_not_called()


def test_layer_plans_and_live_counts_survive_shared_metadata_updates(monkeypatch):
    c = coordinator(monkeypatch)
    run_layer(c, layer=0)
    first = c.plan_kernel.call_args.kwargs
    assert first["plan"].miss_src.data_ptr() == c._lagom_plans[0].miss_src.data_ptr()
    c.num_real_reqs.fill_(1)
    run_layer(c, layer=1)
    second = c.plan_kernel.call_args.kwargs
    assert first["num_real_reqs"].item() == 2
    assert second["num_real_reqs"].item() == 1
    assert first["plan"].miss_src.data_ptr() != second["plan"].miss_src.data_ptr()
    assert c._lagom_graphs.consume_layer.call_args_list[0].args == (0,)
    assert c._lagom_graphs.consume_layer.call_args_list[1].args == (1,)


def test_request_release_drains_transfers_before_host_pages_can_be_reused(monkeypatch):
    c = coordinator(monkeypatch)
    run_layer(c)
    c._drain_lagom_before_request_release(2)
    c._lagom_graphs.synchronize.assert_called_once_with()


def test_failed_demand_copy_marks_published_residency_unusable(monkeypatch):
    c = coordinator(monkeypatch)
    failure = RuntimeError("copy launch failed")

    def copy(_plan, **kwargs):
        if kwargs["demand"]:
            raise failure

    c.copy_kernel.side_effect = copy
    with pytest.raises(RuntimeError, match="copy launch failed"):
        run_layer(c)
    c._lagom_graphs.fail.assert_called_once_with(failure)


@pytest.mark.parametrize("decode_width,prediction_steps", [(8, 1), (2, 9)])
def test_alternating_real_probe_and_fallback_windows_keep_workspace_layout(
    monkeypatch, decode_width, prediction_steps
):
    c = coordinator(monkeypatch)
    c._lagom_prediction_steps = prediction_steps
    c._speculative_verify_width = decode_width + prediction_steps
    c._pr_verify_layout = {"active_slots": 40}
    c._spec_states = [object()]
    c.force_h2d_each_step = False
    c.skip_io, c.item_size_bytes = False, 584
    c.req_device_buffer_tokens = torch.full((1, 4, 64), -1, dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.arange(256, dtype=torch.int32).view(1, 4, 64)
    c.req_to_host_pool = torch.zeros((4, 100), dtype=torch.int64)
    c.mem_pool_host = SimpleNamespace(kv_buffer=[object()])
    c.mem_pool_device = SimpleNamespace(kv_buffer=[object()])
    c.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    planning_width = decode_width + prediction_steps
    calls = []

    def fused(**kwargs):
        calls.append(kwargs)
        output = kwargs["top_k_device_locs"]
        output.copy_(torch.arange(output.numel(), dtype=torch.int32).view_as(output))

    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.load_cache_to_device_buffer_spec_mla",
        fused,
    )
    for width in [decode_width, prediction_steps, 1, decode_width, prediction_steps]:
        selected = torch.arange(2 * width * c.top_k, dtype=torch.int32).view(
            -1, c.top_k
        )
        output = torch.empty_like(selected)
        c._run_swap_in_verify_window(
            req_pool_indices=torch.tensor([2, 0]),
            compressed_seq_lens=torch.tensor([60, 70]),
            top_k_result=selected,
            layer_id=0,
            verify_width=width,
            output_buffer=output,
        )
        call = calls[-1]
        assert call["state"] is c._spec_states[0]
        assert call["top_k_tokens"].shape == (2, planning_width, c.top_k)
        torch.testing.assert_close(
            call["top_k_tokens"][:, :width].reshape_as(selected), selected
        )
        assert (call["top_k_tokens"][:, width:] == -1).all()
        assert (call["seq_lens"].view(2, planning_width)[:, width:] == 0).all()
        expected = torch.arange(2 * planning_width * c.top_k, dtype=torch.int32).view(
            2, planning_width, c.top_k
        )
        torch.testing.assert_close(output.view(2, width, c.top_k), expected[:, :width])
