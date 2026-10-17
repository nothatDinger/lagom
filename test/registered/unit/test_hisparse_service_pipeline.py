from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.benchmark.hisparse_service_pipeline import (
    COMMIT,
    COPY,
    GATHER,
    MARKER,
    analyze_service,
)
from sglang.srt.utils.hisparse_verify_backend import validate_verify_backend
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
        force_h2d=True,
        skip_io=False,
    )


def test_backend_default_and_request_tail_scratch_are_explicit():
    assert validate_verify_backend("split", **settings()) is None
    assert validate_verify_backend("pr_fused", **settings()) == dict(
        active_slots=3075, scratch_offset=3075, scratch_slots=61
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("cuda", False),
        ("dsv4", False),
        ("speculative", False),
        ("force_h2d", False),
        ("skip_io", True),
        ("width", 0),
        ("hot", 2560),
        ("page_size", 3),
        ("prefetcher", "previous"),
    ],
)
def test_unsupported_backend_configurations_fail_before_serving(key, value):
    config = settings()
    config[key] = value
    with pytest.raises(ValueError):
        validate_verify_backend("pr_fused", **config)


@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
def test_pr_dispatch_has_no_independent_copy_and_profile_counts_follow_request_order(
    monkeypatch,
    request_dtype,
):
    from sglang.srt.utils.hisparse_verify_backend import (
        initialize_verify_hash_state,
        run_pr_fused_verify,
    )

    fused, separate, init = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(
        "sglang.benchmark.hisparse_pr40670_retry_candidate.load_cache_to_device_buffer_spec_mla",
        fused,
    )
    monkeypatch.setattr(
        "sglang.benchmark.hisparse_pr40670_reference.initialize_hisparse_spec_state",
        init,
    )
    monkeypatch.setattr(
        "sglang.kernels.ops.kvcache.hisparse.copy_cache_planned_mla", separate
    )
    c = SimpleNamespace()
    c._verify_impl = "pr_fused"
    c._pr_verify_layout = dict(active_slots=131, scratch_offset=131, scratch_slots=61)
    c._speculative_verify_width = 6
    c.force_h2d_each_step = True
    c.is_dsv4_hisparse = True
    c.top_k, c.device_buffer_size, c.item_size_bytes, c.skip_io = 16, 128, 584, False
    c.req_device_buffer_tokens = torch.full((1, 4, 192), -1, dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.arange(4 * 192, dtype=torch.int32).view(
        1, 4, 192
    )
    c.lru_slots = torch.zeros(1, 4, 128, dtype=torch.int16)
    c._lru_init = torch.arange(128, dtype=torch.int16)
    metadata = torch.zeros(5, 480, dtype=torch.int32)
    metadata[0, :4] = torch.tensor([3, 8, 13, 21])
    c._window_hash_states = [SimpleNamespace(scratch_state=metadata)]
    c._spec_miss_src = torch.zeros(4, 96, dtype=torch.int64)
    c._spec_miss_dst = torch.zeros(4, 96, dtype=torch.int32)
    c._spec_miss_count = torch.zeros(4, dtype=torch.int32)
    c.num_real_reqs = torch.tensor([2], dtype=torch.int32)
    c.req_to_host_pool = torch.zeros(4, 1024, dtype=torch.int64)
    c.mem_pool_host = SimpleNamespace(kv_buffer=[torch.empty(1)])
    c.mem_pool_device = SimpleNamespace(kv_buffer=[torch.empty(1)])
    c._h2d_probe = SimpleNamespace(record=MagicMock())
    locs_before = c.req_device_buffer_token_locs.clone()
    result = torch.empty(2, 6, 16, dtype=torch.int32)
    initialize_verify_hash_state(
        "pr_fused",
        device_buffer_tokens=c.req_device_buffer_tokens[0],
        req_pool_indices=torch.tensor([2, 0], dtype=request_dtype),
        state=c._window_hash_states[0],
    )
    run_pr_fused_verify(
        c,
        common=dict(
            host_cache_locs=c.req_to_host_pool,
            host_cache=c.mem_pool_host.kv_buffer[0],
            device_buffer=c.mem_pool_device.kv_buffer[0],
            num_real_reqs=c.num_real_reqs,
        ),
        req_pool_indices=torch.tensor([2, 0], dtype=request_dtype),
        compressed_seq_lens=torch.tensor([1024, 1024]),
        top_k_result=result,
        layer_id=0,
        verify_width=6,
        output_buffer=result,
        skip_io=False,
        miss_count=c._spec_miss_count[:2],
    )
    fused.assert_called_once()
    separate.assert_not_called()
    init.assert_called_once()
    assert init.call_args.kwargs["req_pool_indices"].dtype == torch.int64
    assert fused.call_args.kwargs["req_pool_indices"].dtype == torch.int64
    assert fused.call_args.kwargs["device_buffer_locs"].shape == (4, 131)
    assert fused.call_args.kwargs["seq_lens"].shape == (12,)
    assert "miss_src" not in fused.call_args.kwargs  # PR RecordMissPlan stays false.
    assert c._spec_miss_count[:2].tolist() == [13, 3]
    assert c._h2d_probe.record.call_args.kwargs["copy_blocks"] == 0
    torch.testing.assert_close(c.req_device_buffer_token_locs, locs_before)


def data(backend):
    rows, events = [], []
    policy = (
        "Dsv4PagedKVTransferPolicy"
        if backend == "pr_fused"
        else "MetadataOnlyKVTransferPolicy"
    )
    for i in range(4):
        rows.append(
            dict(
                index=i,
                real_batch=2,
                captured_batch=4,
                verify_width=6,
                layer=i % 2,
                misses_per_request=[30, 40],
                unique_misses=70,
                item_bytes=584,
                h2d_bytes=70 * 584,
                skip_io=False,
            )
        )

        def event(name, ts, grid, block):
            return dict(
                ph="X",
                cat="kernel",
                name=name,
                ts=i * 100 + ts,
                dur=5,
                args=dict(device=0, stream=7, grid=grid, block=block),
            )

        events.extend(
            [
                event(GATHER + policy, 0, [4, 48, 1], [64, 1, 1]),
                event(COMMIT + policy, 10, [4, 1, 1], [512, 1, 1]),
            ]
        )
        if backend == "split":
            events.append(event(COPY, 20, [4, 1, 1], [1024, 1, 1]))
        events.append(event(MARKER, 30, [1, 1, 1], [128, 1, 1]))
    return dict(
        schema_version=1,
        verify_backend=backend,
        verify_config=dict(top_k=512, force_h2d_each_step=True),
        samples=rows,
        total_records=4,
        overflow=False,
        pid=123,
        device=0,
    ), events


@pytest.mark.parametrize("backend", ["split", "pr_fused"])
def test_real_payload_pairs_with_whole_pipeline_and_component_times(backend):
    probe, events = data(backend)
    result = analyze_service(probe, events, backend)
    group = result["groups"][0]
    assert group["unique_misses_per_request_mean"] == 35
    assert group["gather_us"]["mean"] == 5
    assert (
        group["copy_us"] is None
        if backend == "pr_fused"
        else group["copy_us"]["mean"] == 5
    )
    assert group["verify_cadence"]["mean_period_ms"] == 0.2


@pytest.mark.parametrize(
    "fault",
    [
        "overflow",
        "wrong_capture_count",
        "wrong_policy",
        "extra_copy",
        "shape",
        "bytes",
        "missing_commit",
    ],
)
def test_untrustworthy_profile_is_rejected(fault):
    probe, events = data("pr_fused")
    if fault == "overflow":
        probe["overflow"] = True
    elif fault == "wrong_capture_count":
        probe["total_records"] = 3
    elif fault == "wrong_policy":
        events[0]["name"] = GATHER + "MetadataOnlyKVTransferPolicy"
    elif fault == "extra_copy":
        e = deepcopy(events[0])
        e.update(name=COPY, ts=20)
        events.append(e)
    elif fault == "shape":
        events[0]["args"]["grid"] = [32, 48, 1]
    elif fault == "bytes":
        probe["samples"][0]["h2d_bytes"] = 1
    else:
        events.pop(1)
    with pytest.raises(ValueError):
        analyze_service(probe, events, "pr_fused")
