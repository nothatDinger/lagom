"""Forecast ownership and attention-to-overwrite release dependencies."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse_staging import HiSparseVerificationStaging
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.hisparse_staging import HiSparseStagingLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def coordinator(monkeypatch):
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.prefetcher_name = "lagom"
    c.prefetcher = SimpleNamespace(logical_entries=2)
    c.top_k, c._lagom_prediction_steps, c.compress_ratio, c.device = 2, 2, 4, "cpu"
    c.force_h2d_each_step = False
    c._speculative_verify_width = 5
    c._pr_verify_layout = {"active_slots": 12}
    c._spec_states = [
        SimpleNamespace(cache_index=object(), cache_policy=object(), scratch_state=object())
        for _ in range(2)
    ]
    c.req_device_buffer_tokens = torch.full((2, 4, 16), -1, dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.arange(128, dtype=torch.int32).view(2, 4, 16)
    c.req_to_host_pool = torch.zeros((4, 100), dtype=torch.int64)
    c.mem_pool_host = SimpleNamespace(kv_buffer=[object(), object()])
    c.mem_pool_device = SimpleNamespace(kv_buffer=[object(), object()])
    c.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    c.order = []
    module = SimpleNamespace(**{
        name: MagicMock(side_effect=lambda *args, event=name: c.order.append(event))
        for name in ("plan_verify", "copy_verify_device", "copy_verify_host", "promote", "prefetch")
    })
    monkeypatch.setattr("sglang.kernels.ops.kvcache.hisparse_staging._module", lambda *args: module)
    pool = SimpleNamespace(
        device="cpu", create_buffer=lambda *, num_pages: torch.zeros((num_pages, 37440), dtype=torch.uint8),
    )
    c.verification_staging = HiSparseVerificationStaging(
        layout=HiSparseStagingLayout(4, 5, 2, 2, 2), pool=pool, hot_slots=8,
    )
    def ready(layer, batch, transfer, *, capturing):
        c.order.append("release")
        if not capturing:
            transfer()
    c._lagom_graphs = SimpleNamespace(
        consume_layer=MagicMock(side_effect=lambda layer: c.order.append("consume")),
        plan_ready=MagicMock(side_effect=ready), synchronize=MagicMock(), fail=MagicMock(),
    )
    return c, module


def run_layer(c, layer=0):
    raw = torch.tensor([
        [[0, 1], [2, 3], [3, 4], [0, 2], [3, 4]],
        [[0, 1], [2, 3], [5, 6], [1, 6], [-1, 7]],
    ], dtype=torch.int32)
    original = raw.clone()
    result = c.swap_in_lagom_pages(
        req_pool_indices=torch.tensor([2, 0]),
        compressed_seq_lens=torch.tensor([4, 4, 4, 5, 5, 7, 7, 7, 8, 8]),
        source_prefix_lens=torch.tensor([13, 24]),
        top_k_result=raw.view(-1, 2), layer_id=layer, decode_width=3,
        output_buffer=torch.empty_like(raw.view(-1, 2)),
        window_locs=torch.full_like(raw.view(-1, 2), -1),
    )
    torch.testing.assert_close(raw, original)
    return result


def test_slot_is_released_only_after_attention_and_promotion(monkeypatch):
    c, module = coordinator(monkeypatch)
    run_layer(c)
    assert c.order == ["consume", "plan_verify", "copy_verify_device", "copy_verify_host"]
    module.prefetch.assert_not_called()
    c.order.append("attention")
    c.finish_staged_attention(0, capturing=False)
    assert c.order[-4:] == ["attention", "promote", "release", "prefetch"]


def test_capture_only_records_release_without_foreground_prefetch(monkeypatch):
    c, module = coordinator(monkeypatch)
    run_layer(c)
    c.finish_staged_attention(0, capturing=True)
    assert c.order[-2:] == ["promote", "release"]
    module.prefetch.assert_not_called()
    assert c._lagom_graphs.plan_ready.call_args.kwargs["capturing"]


def test_forecasts_only_include_committed_history_and_rank_budget(monkeypatch):
    c, module = coordinator(monkeypatch)
    c.prefetcher.logical_entries = 1
    run_layer(c)
    staging = c.verification_staging
    assert staging.forecast_tokens[0][:2].tolist() == [[0, -1, -1, -1], [1, -1, -1, -1]]
    assert staging.forecast_reqs[0][:2].tolist() == [2, 0]
    assert staging.history[:2].tolist() == [3, 6]
    assert module.plan_verify.call_args.args[0][:, :3].tolist() == [
        [[0, 1], [2, 3], [3, 4]], [[0, 1], [2, 3], [5, 6]],
    ]


def test_per_layer_forecast_descriptors_survive_shared_plan_updates(monkeypatch):
    c, module = coordinator(monkeypatch)
    run_layer(c, 0)
    c.finish_staged_attention(0, capturing=True)
    staging = c.verification_staging
    first = staging.forecast_tokens[0].clone()
    c.num_real_reqs.fill_(1)
    run_layer(c, 1)
    assert staging.forecast_live[0].item() == 2
    assert staging.forecast_live[1].item() == 1
    torch.testing.assert_close(staging.forecast_tokens[0], first)
    assert len({staging.arena.data_ptr(), *(x.data_ptr() for x in staging.forecasts)}) == 3


def test_wrong_consumer_cannot_release_an_arena(monkeypatch):
    c, module = coordinator(monkeypatch)
    run_layer(c)
    with pytest.raises(RuntimeError, match="wrong attention layer"):
        c.finish_staged_attention(1, capturing=False)
    module.prefetch.assert_not_called()


def test_request_reuse_drains_writers_then_clears_forecast_indices(monkeypatch):
    c, module = coordinator(monkeypatch)
    run_layer(c)
    c._drain_lagom_before_request_release(2)
    c._lagom_graphs.synchronize.assert_called_once_with()
    for index in c.verification_staging.forecast_indices:
        index[2].fill_(123)
    c.verification_staging.reset_request(2)
    assert all((index[2] == -1).all() for index in c.verification_staging.forecast_indices)


def test_failed_refill_does_not_overwrite_forecast_slot(monkeypatch):
    c, module = coordinator(monkeypatch)
    module.copy_verify_host.side_effect = RuntimeError("copy launch failed")
    with pytest.raises(RuntimeError, match="copy launch failed"):
        run_layer(c)
    c._lagom_graphs.plan_ready.assert_not_called()
    module.prefetch.assert_not_called()
