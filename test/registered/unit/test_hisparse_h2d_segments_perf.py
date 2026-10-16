import json

import pytest

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_latency import perturbation
from sglang.benchmark.hisparse_h2d_segments_perf import (
    REVIEWED,
    summarize_invocation,
    verify_codegen,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_weighted_kv_and_partition_accounting():
    rows = []
    for i, kv in enumerate((12, 13)):
        rows.append(
            dict(
                kv=kv,
                sm=i,
                begin_ns=100,
                end_ns=6100,
                prep_ns=1000,
                issue_ns=1000,
                residual_ns=2000,
                store_ns=1000,
                copy_ns=5000,
            )
        )
    result = summarize_invocation(rows)
    assert result["total_kv"] == 25
    assert result["means"]["copy_per_kv_us"] == 10 / 25
    assert result["means"]["copy_per_warp_us"] == 5
    assert result["means"]["other_per_warp_us"] == 1
    assert result["by_kv"]["12"]["means"]["copy_per_kv_us"] == 5 / 12
    assert result["by_kv"]["13"]["means"]["copy_per_kv_us"] == 5 / 13
    assert sum(
        result["means"][s + "_per_kv_us"]
        for s in ("prep", "issue", "residual", "store")
    ) == pytest.approx(0.4)


def waves():
    return [
        dict(trial=t, variant=v, blocks=b, copy_us=[60.0] * 64)
        for t in range(6)
        for v in REVIEWED
        for b in (4, 8)
    ]


def test_perturbation_rejects_cta_specific_instrumentation():
    data = waves()
    assert perturbation(data)["valid"]
    for w in data:
        if w["variant"] == "trace" and w["blocks"] == 8:
            w["copy_us"] = [72.0] * 64
    result = perturbation(data)
    assert not result["valid"]
    assert not result["gates"]["relative_grid_ratio"]["valid"]


def test_incomplete_timing_not_accepted():
    with pytest.raises(ValueError):
        perturbation(waves()[:-1])


@pytest.fixture
def evidence(tmp_path):
    source = tmp_path / "kernel.cuh"
    source.write_text("unchanged source")
    artifacts = [
        dict(variant=v, sha256=h, controls_complete=True, timer_presence_valid=True)
        for v, h in REVIEWED.items()
    ]
    check = tmp_path / "check.json"
    check.write_text(
        json.dumps(
            dict(
                success=True,
                correctness=[{}] * 12,
                jit_artifacts=artifacts,
                code=dict(sha256={"kernel.cuh": sha256(source)}),
            )
        )
    )
    codegen = tmp_path / "codegen.json"
    codegen.write_text(
        json.dumps(
            dict(
                success=True,
                no_spill=True,
                artifacts=artifacts,
                check_report=str(check),
                input_sha256=sha256(check),
            )
        )
    )
    return tmp_path, source, check, codegen


def test_reviewed_artifacts_accepted_without_editing_old_report(evidence):
    root, _, _, codegen = evidence
    _, review = verify_codegen(codegen, root)
    assert review["timing_semantics_reviewed"]


@pytest.mark.parametrize("change", ["source", "check", "artifact", "controls"])
def test_stale_or_unreviewed_evidence_rejected(evidence, change):
    root, source, check, codegen = evidence
    if change == "source":
        source.write_text("modified kernel")
    elif change == "check":
        check.write_text(check.read_text() + " ")
    else:
        doc = json.loads(codegen.read_text())
        if change == "artifact":
            doc["artifacts"][2]["sha256"] = "unreviewed"
        else:
            doc["artifacts"][2]["controls_complete"] = False
        codegen.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_codegen(codegen, root)
