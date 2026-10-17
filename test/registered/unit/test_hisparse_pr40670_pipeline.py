from copy import deepcopy

import pytest

from sglang.benchmark.hisparse_pr40670_pipeline import (
    METHODS,
    ORDERS,
    summarize,
    validate_pipeline_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def trace(local=False):
    events = []
    for i, (name, grid, block) in enumerate(
        [
            ("load_cache_to_device_buffer_spec_gather_kernel", [32, 2, 1], [64, 1, 1]),
            ("load_cache_to_device_buffer_spec_commit_kernel", [32, 1, 1], [512, 1, 1]),
            *(
                [("copy_cache_planned_kernel", [4, 1, 1], [1024, 1, 1])]
                if local
                else []
            ),
        ]
    ):
        events.append(
            dict(
                cat="kernel",
                name=name,
                ts=i * 10,
                dur=5.0,
                args=dict(grid=grid, block=block, device=0),
            )
        )
    return dict(traceEvents=events)


@pytest.mark.parametrize("local", [False, True])
def test_full_pipeline_requires_every_kernel(local):
    data = trace(local)
    assert validate_pipeline_profile(data, METHODS[int(local)], 32)["valid"]
    data["traceEvents"].pop()
    with pytest.raises(ValueError):
        validate_pipeline_profile(data, METHODS[int(local)], 32)


@pytest.mark.parametrize("fault", ["grid", "device", "order", "duration", "copy_count"])
def test_profile_rejects_wrong_launches(fault):
    data = trace(True)
    event = data["traceEvents"][0]
    if fault == "grid":
        event["args"]["grid"] = [4, 2, 1]
    elif fault == "device":
        event["args"]["device"] = 1
    elif fault == "order":
        event["ts"] = 100
    elif fault == "duration":
        event["dur"] = float("nan")
    else:
        data["traceEvents"].append(deepcopy(data["traceEvents"][-1]))
    with pytest.raises(ValueError):
        validate_pipeline_profile(data, METHODS[1], 32)


def waves():
    return [
        dict(trial=t, method=m, us=[10.0 if m == METHODS[0] else 20.0] * 32)
        for t, order in enumerate(ORDERS)
        for m in order
    ]


def test_comparison_direction_and_balanced_order():
    assert sum(o[0] == METHODS[0] for o in ORDERS) == 3
    summary = summarize(waves(), 32)
    assert summary["local_over_pr"] == 2.0
    assert summary["local_over_pr_wave_ratios"] == [2.0] * 6
    assert summary["samples_per_method"] == 192


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "zero", "short"])
def test_invalid_waves_do_not_produce_ratio(fault):
    data = waves()
    if fault == "missing":
        data.pop()
    elif fault == "duplicate":
        data[-1] = data[0]
    elif fault == "nan":
        data[0]["us"][0] = float("nan")
    elif fault == "zero":
        data[0]["us"][0] = 0
    else:
        data[0]["us"].pop()
    with pytest.raises(ValueError):
        summarize(data, 32)


@pytest.mark.parametrize("fault", [None, "scale_byte", "guard", "unresolved"])
def test_content_oracle_checks_payload_scale_and_guards(monkeypatch, fault):
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "scripts/pd_test")
    )
    from bench_hisparse_pr40670_pipeline import PipelineKVError, oracle
    from check_hisparse_pr40670 import c4_indices
    from sglang.benchmark.hisparse_pr40670_w6 import input_tokens

    initial = torch.arange(144, dtype=torch.int32).view(1, -1)
    before = torch.full((3, 37440), 165, dtype=torch.uint8)
    host = (
        torch.arange(9 * 37440, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(9, -1)
    )
    output = before.clone()
    top = torch.tensor(input_tokens(8), dtype=torch.int32).view(1, 6, 16)
    loc = (top - 128).clone()
    si = c4_indices(torch, top.reshape(-1).long())
    di = c4_indices(torch, loc.reshape(-1).long())
    output.view(-1)[di] = host.view(-1)[si]
    if fault == "scale_byte":
        output.view(-1)[int(di[0, -1])] ^= 1
    elif fault == "guard":
        output[-1, -1] ^= 1
    elif fault == "unresolved":
        loc[0, 0, 0] = -1
    pr = SimpleNamespace(host_stride=576, host=host, initial_locs=initial)
    reader = SimpleNamespace(
        args=dict(
            top_k_device_locs=loc,
            device_buffer=output,
            req_pool_indices=torch.tensor([0]),
            top_k_tokens=top,
            device_buffer_locs=initial[:, :129],
            miss_count=torch.tensor([8]),
        )
    )
    if fault is None:
        assert oracle(torch, pr, before, reader, 8, METHODS[1])["valid"]
    elif fault == "unresolved":
        with pytest.raises(PipelineKVError) as caught:
            oracle(torch, pr, before, reader, 8, METHODS[1])
        assert caught.value.diagnostic["unresolved_count"] == 1
        assert caught.value.diagnostic["unresolved"] == [
            dict(b=0, rid=0, w=0, j=0, token=128, dst=-1)
        ]
        assert caught.value.diagnostic["miss_count"] == [8]
    else:
        with pytest.raises(ValueError):
            oracle(torch, pr, before, reader, 8, METHODS[1])


def test_missing_union_token_diagnostic_uses_epoch_and_local_miss_count(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "scripts/pd_test")
    )
    from bench_hisparse_pr40670_pipeline import unresolved_diagnostic

    top = torch.arange(128, 224, dtype=torch.int32).view(1, 6, 16)
    loc = torch.arange(96, dtype=torch.int32).view(1, 6, 16)
    loc[top == 196] = -1
    rows = torch.zeros((2, 480), dtype=torch.int32)
    rows[0, 3] = 1
    for slot, token in enumerate(t for t in range(128, 224) if t != 196):
        rows[1, 2 * slot] = token
        rows[1, 2 * slot + 1] = 1
    state = SimpleNamespace(scratch_state=rows, cache_index=torch.zeros(1, 2, 256))
    args = dict(
        top_k_tokens=top,
        top_k_device_locs=loc,
        req_pool_indices=torch.tensor([0]),
        device_buffer=torch.zeros(3, 37440),
        miss_count=torch.tensor([95]),
        state=state,
    )
    result = unresolved_diagnostic(torch, args)
    assert result["unresolved_count"] == 1
    assert result["unresolved"][0]["token"] == 196
    assert result["miss_count"] == [95]
    assert result["final_tables"][0]["table_unique"] == 95
    assert result["final_tables"][0]["requested_missing_from_table"] == [196]


@pytest.mark.parametrize(
    "milliseconds", [0.0125, 0.0, -1.0, float("nan"), float("inf")]
)
def test_graph_timing_synchronizes_and_rejects_invalid_intervals(milliseconds):
    from types import SimpleNamespace

    from sglang.benchmark.hisparse_pr40670_pipeline import graph_sample_us

    calls = []
    end = SimpleNamespace(synchronize=lambda: calls.append("sync"))

    def elapsed(event):
        assert event is end and calls == ["sync"]
        return milliseconds

    start = SimpleNamespace(elapsed_time=elapsed)
    if milliseconds == 0.0125:
        assert graph_sample_us(start, end) == 12.5
    else:
        with pytest.raises(ValueError, match="Invalid captured event"):
            graph_sample_us(start, end)
