"""Lagom cache ownership, stream edges, and forecast/verify separation."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.hisparse_prefetcher import HiSparsePrefetchStats
from sglang.srt.mem_cache.hisparse_spec import validate_pr_fused_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def coordinator(monkeypatch):
    module = SimpleNamespace(
        Event=lambda **_: SimpleNamespace(
            record=MagicMock(), wait=MagicMock(), synchronize=MagicMock()
        ),
        current_stream=lambda: "target",
        stream=lambda _: nullcontext(),
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.device_module", module
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.prefetcher_name = "lagom"
    c.prefetcher = SimpleNamespace(logical_entries=2, stats=HiSparsePrefetchStats())
    c.top_k, c._lagom_prediction_steps, c.compress_ratio, c.device = 2, 2, 4, "cpu"
    c._prefetch_generation = torch.zeros(4, dtype=torch.int64)
    c._lagom_pending, c._lagom_inflight = {}, {}
    c._lagom_layer_events = [module.Event(external=True)]
    c._active_dspark_window = None
    c._previous_prefetch_stream = SimpleNamespace(wait_stream=MagicMock())
    c._run_swap_in_verify_window = MagicMock()
    return c


def queue(c):
    predictions = torch.tensor([[[0, 2], [3, 4]], [[1, 6], [-1, 7]]], dtype=torch.int32)
    c.queue_lagom_prefetch(
        req_pool_indices=torch.tensor([2, 0]),
        req_pool_indices_cpu=torch.tensor([2, 0]),
        source_committed_lens_cpu=torch.tensor([13, 24]),
        predicted_c4_entries=predictions,
        layer_id=0,
    )
    return predictions


def test_forecasts_own_snapshots_and_filter_uncommitted_c4(monkeypatch):
    c = coordinator(monkeypatch)
    original = queue(c)
    original.fill_(99)  # A later graph replay may overwrite its outputs.
    c._run_swap_in_verify_window.assert_not_called()
    c.submit_lagom_prefetches()
    call = c._run_swap_in_verify_window.call_args.kwargs
    assert call["top_k_result"].tolist() == [[0, 2], [-1, -1], [1, -1], [-1, -1]]
    assert call["compressed_seq_lens"].tolist() == [3, 6]
    assert call["req_pool_indices"].tolist() == [2, 0]
    assert call["num_real_reqs"].tolist() == [2]
    assert call["verify_width"] == 2
    c._previous_prefetch_stream.wait_stream.assert_called_once_with("target")
    c._lagom_layer_events[0].record.assert_called_once_with(c._previous_prefetch_stream)
    assert not c._lagom_pending


def test_prefetch_waits_for_c4_transaction_and_rejects_reused_slots(monkeypatch):
    c = coordinator(monkeypatch)
    queue(c)
    c._active_dspark_window = object()
    with pytest.raises(RuntimeError, match="C4 commit"):
        c.submit_lagom_prefetches()
    c._active_dspark_window = None
    c._prefetch_generation[2] += 1
    c.submit_lagom_prefetches()
    c._run_swap_in_verify_window.assert_not_called()
    assert c.prefetcher.stats.stale_tasks == 1


def test_prefetch_budget_applies_to_each_prediction_row(monkeypatch):
    c = coordinator(monkeypatch)
    c.prefetcher.logical_entries = 1
    queue(c)
    c.submit_lagom_prefetches()
    candidates = c._run_swap_in_verify_window.call_args.kwargs["top_k_result"]
    assert (candidates[:, 1] == -1).all()
    assert c.prefetcher.stats.submitted_entries == 2 * 2 * 1


def test_graph_layer_wait_is_unconditional_and_teardown_is_cpu_complete(monkeypatch):
    c = coordinator(monkeypatch)
    c.consume_lagom_prefetch(0)
    c._lagom_layer_events[0].wait.assert_called_once_with("target")
    queue(c)
    c.submit_lagom_prefetches()
    c.consume_lagom_prefetch(0)
    assert c._lagom_inflight[0].consumed
    # Even consumed tasks retain their event until host allocator reclamation.
    c._drain_lagom_before_request_release(2)
    c._lagom_layer_events[0].synchronize.assert_called_once()
    assert not c._lagom_inflight


def test_abort_discards_forecasts_without_h2d(monkeypatch):
    c = coordinator(monkeypatch)
    queue(c)
    c.discard_lagom_prefetches()
    c.submit_lagom_prefetches()
    c._run_swap_in_verify_window.assert_not_called()


def test_real_rows_and_probe_rows_have_separate_io_ownership(monkeypatch):
    c = coordinator(monkeypatch)
    c._pr_verify_layout = {"active_slots": 5}
    c.req_device_buffer_tokens = torch.zeros((1, 4, 8), dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.zeros_like(c.req_device_buffer_tokens)
    c._spec_states = [object()]
    c.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    c._run_swap_in_verify_window.side_effect = lambda **kw: kw["output_buffer"].fill_(
        11
    )
    lookup = MagicMock(side_effect=lambda **kw: kw["output"].fill_(-1))
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.lookup_hisparse_spec_resident", lookup
    )
    raw = torch.arange(20, dtype=torch.int32).view(10, 2)
    result = c.swap_in_lagom_pages(
        req_pool_indices=torch.tensor([2, 0]),
        compressed_seq_lens=torch.arange(10, 20),
        top_k_result=raw,
        layer_id=0,
        decode_width=3,
    ).view(2, 5, 2)
    assert (result[:, :3] == 11).all()
    assert (result[:, 3:] == -1).all()
    assert (
        c._run_swap_in_verify_window.call_args.kwargs["top_k_result"].tolist()
        == raw.view(2, 5, 2)[:, :3].reshape(-1, 2).tolist()
    )
    assert (
        lookup.call_args.kwargs["top_k_tokens"].tolist()
        == raw.view(2, 5, 2)[:, 3:].tolist()
    )


def test_lagom_preserves_residency_and_separate_planning_capacity():
    config = dict(
        cuda=True,
        dsv4=True,
        speculative=True,
        width=17,
        top_k=512,
        hot=4608,
        page_size=64,
        force_h2d=False,
        skip_io=False,
        prefetcher="lagom",
        prediction_steps=8,
    )
    layout = validate_pr_fused_config(**config)
    assert layout["active_slots"] == 4608 + 1 + 5
    with pytest.raises(ValueError, match="preserved residency"):
        validate_pr_fused_config(**dict(config, force_h2d=True))


@pytest.mark.parametrize("decode_width,prediction_steps", [(8, 1), (2, 9)])
def test_alternating_real_probe_and_fallback_windows_keep_workspace_layout(
    monkeypatch, decode_width, prediction_steps
):
    c = coordinator(monkeypatch)
    del c._run_swap_in_verify_window
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
    planning_width = max(decode_width, prediction_steps)
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
