import json
from collections import Counter

import pytest

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_windows_perf import (
    ORDERS,
    REVIEWED,
    perturbation,
    summarize_invocation,
    verify_codegen,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def waves():
    return [
        dict(trial=t, variant=v, blocks=b, copy_us=[60.0] * 64)
        for t in range(8)
        for v in REVIEWED
        for b in (4, 8)
    ]


def test_balanced_positions_and_grid_order():
    for v in REVIEWED:
        assert Counter(order.index(v) for order in ORDERS) == dict.fromkeys(range(4), 2)
        assert sum((t % 4 + t // 4) % 2 == 0 for t in range(8)) == 4
        for t in range(4):
            assert ORDERS[t] == ORDERS[t + 4]
            assert (t % 4 + t // 4) % 2 != ((t + 4) % 4 + (t + 4) // 4) % 2


def test_probe_gates_are_independent():
    data = waves()
    assert perturbation(data)["valid"]
    for w in data:
        if w["variant"] == "tail" and w["blocks"] == 8:
            w["copy_us"] = [57.6] * 64
    r = perturbation(data)
    assert r["by_probe"] == dict(issue=True, tail=False)
    assert r["gates"]["tail/control_CTA8"]["valid"]
    assert not r["gates"]["relative_grid_ratio_tail"]["valid"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "short"])
def test_bad_coverage_rejected(change):
    data = waves()
    if change == "missing":
        data.pop()
    elif change == "duplicate":
        data[-1] = data[0]
    elif change == "nan":
        data[0]["copy_us"][0] = float("nan")
    else:
        data[0]["copy_us"].pop()
    with pytest.raises(ValueError):
        perturbation(data)


def test_weighted_window_is_not_copy_partition():
    rows = [
        dict(kv=k, sm=i, begin_ns=0, end_ns=6000, interval_ns=5000)
        for i, k in enumerate((12, 13))
    ]
    result = summarize_invocation(rows)
    assert result["means"]["interval_per_kv_us"] == 10 / 25
    assert result["means"]["interval_per_warp_us"] == 5
    assert result["by_kv"]["12"]["means"]["interval_per_kv_us"] == 5 / 12
    assert "copy_per_kv_us" not in result["means"]


def test_evidence_binding(tmp_path):
    src = tmp_path / "source"
    src.write_text("reviewed")
    artifacts = [
        dict(variant=v, sha256=h, controls_complete=True, timer_presence_valid=True)
        for v, h in REVIEWED.items()
    ]
    check = tmp_path / "check.json"
    check.write_text(
        json.dumps(
            dict(
                success=True,
                correctness=[{}] * 16,
                jit_artifacts=artifacts,
                code=dict(sha256={"source": sha256(src)}),
            )
        )
    )
    doc = dict(
        success=True,
        no_spill=True,
        artifacts=artifacts,
        check_report=str(check),
        input_sha256=sha256(check),
    )
    codegen = tmp_path / "codegen.json"
    codegen.write_text(json.dumps(doc))
    assert verify_codegen(codegen, tmp_path)[1]["timing_semantics_reviewed"]
    doc["artifacts"][2]["sha256"] = "old-probe"
    codegen.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_codegen(codegen, tmp_path)
    doc["artifacts"][2]["sha256"] = REVIEWED["issue"]
    codegen.write_text(json.dumps(doc))
    src.write_text("changed")
    with pytest.raises(ValueError):
        verify_codegen(codegen, tmp_path)
