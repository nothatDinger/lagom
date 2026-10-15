import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.benchmark.hisparse_h2d_grid_perf import LEGACY_LIBRARY
from sglang.benchmark.hisparse_h2d_linear import (
    BINDING_KEYS,
    EDGE_BYTES,
    LINEAR_LIBRARY,
    METHODS,
    PAYLOAD,
    compact_report,
    method_order,
    report_status,
    validate_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def report_fixture(mode="main"):
    report = dict(
        success=True,
        config=dict(mode=mode, order=[0, 1, 1, 0] if mode == "main" else []),
        windows=[],
        edge_checks=[
            dict(
                device=d,
                blocks=b,
                size=n,
                source=s,
                eager=True,
                graph=True,
                guards=True,
            )
            for d in (0, 1)
            for b in (4, 8)
            for n in EDGE_BYTES
            for s in ("host", "hbm")
        ],
        chunk_graphs=[
            dict(
                device=d,
                method=m,
                capture_device=d,
                correctness=True,
                per_copy_timing_events=0,
                launches_per_chunk=1,
                nbytes=64 * 2**20 if m == "ce_bulk" else PAYLOAD,
                banks=7 if m == "ce_bulk" else 64,
                waves=2,
                copy_nodes=14 if m == "ce_bulk" else 128,
                num_blocks={"gather4": 4, "linear4": 4, "linear8": 8}.get(m),
                block_size=1024 if m in ("gather4", "linear4", "linear8") else None,
            )
            for d in (0, 1)
            for m in METHODS
        ],
        bindings=[
            dict(
                device=d,
                valid=True,
                checks=dict.fromkeys(BINDING_KEYS, True),
            )
            for d in (0, 1)
        ],
        jit_artifacts=[
            dict(variant=v, path="/cache/" + name, sha256="a" * 64)
            for v, name in (("legacy", LEGACY_LIBRARY), ("linear", LINEAR_LIBRARY))
        ],
    )
    local = {0: 0, 1: 0}
    for trial, d in enumerate(report["config"]["order"]):
        for position, m in enumerate(method_order(local[d])):
            rates = {
                "gather4": 30 if d == 0 else 23,
                "linear4": 50,
                "linear8": 54,
                "ce_same": 52,
                "ce_bulk": 55,
            }
            report["windows"].append(
                dict(
                    device=d,
                    trial=trial,
                    method=m,
                    position=position,
                    sampler_enabled=bool(local[d]),
                    metrics=dict(
                        valid=True,
                        logical_wall_GBps=rates[m],
                        max_enqueue_fraction=0.001,
                        chunk_gpu_cv=0.001,
                    ),
                    calibration=dict(valid=True),
                )
            )
        local[d] += 1
    return report


@pytest.mark.parametrize("mode", ["check", "main"])
def test_complete_report_separates_correctness_from_performance(mode):
    report = report_fixture(mode)
    assert report_status(report)["ready"]
    text = compact_report(report)
    assert "results_valid=True" in text
    assert ("RATIOS" in text) == (mode == "main")
    if mode == "main":
        assert "CTA8/4_us=[0.9259, 0.9259]" in text


@pytest.mark.parametrize(
    "fault",
    [
        "edge",
        "capture",
        "payload",
        "blocks",
        "chunk_work",
        "artifact",
        "binding",
        "missing",
        "duplicate",
        "order",
        "sampling",
        "quality",
        "on_off",
    ],
)
def test_incomplete_or_confounded_data_never_produces_performance_conclusions(fault):
    r = report_fixture()
    if fault == "edge":
        r["edge_checks"][0]["guards"] = False
    elif fault == "capture":
        r["chunk_graphs"][5]["capture_device"] = 0
    elif fault == "payload":
        r["chunk_graphs"][0]["nbytes"] += 8
    elif fault == "blocks":
        r["chunk_graphs"][2]["num_blocks"] = 4
    elif fault == "chunk_work":
        r["chunk_graphs"][2].update(waves=3, copy_nodes=192)
    elif fault == "artifact":
        r["jit_artifacts"][1]["path"] = "/cache/another.so"
    elif fault == "binding":
        r["bindings"][0]["checks"]["same_target"] = False
    elif fault == "missing":
        r["windows"].pop()
    elif fault == "duplicate":
        r["windows"].append(copy.deepcopy(r["windows"][0]))
    elif fault == "order":
        r["windows"][0]["position"] = 4
    elif fault == "sampling":
        r["windows"][0]["sampler_enabled"] = True
    elif fault == "quality":
        r["windows"][0]["metrics"]["valid"] = False
    else:
        r["windows"][-2]["metrics"]["logical_wall_GBps"] *= 0.7
    assert not report_status(r)["ready"]
    assert "RATIOS" not in compact_report(r)


@pytest.mark.parametrize("fault", ["quality", "on_off", "zero", "nan"])
def test_invalid_same_size_ce_never_blocks_independent_sm_comparisons(fault):
    r = report_fixture()
    for w in r["windows"]:
        if w["method"] != "ce_same":
            continue
        if fault == "quality":
            # The observed round-17 failure: all CE-same enqueue ratios exceed
            # the unchanged 0.10 threshold, with every SM/bulk window valid.
            w["metrics"].update(valid=False, max_enqueue_fraction=0.4)
        elif fault == "on_off" and w["sampler_enabled"]:
            w["metrics"]["logical_wall_GBps"] *= 0.7
        elif fault in ("zero", "nan"):
            w["metrics"]["logical_wall_GBps"] = 0 if fault == "zero" else float("nan")
    before = copy.deepcopy(r)
    status = report_status(r)
    assert not status["ready"] and not status["timing"]
    assert status["comparisons"] == dict(sm=True, ce_same=False, bulk_reference=True)
    text = compact_report(r)
    assert "results_valid=False" in text
    assert "RATIOS_SM d0" in text and "RATIOS_SM d1" in text
    assert "CROSS_SM" in text and "BULK_REFERENCE" in text
    assert "RATIOS_CEsame" not in text
    # Analysis must not turn failed acquisition-window flags into passes.
    assert r == before


@pytest.mark.parametrize("fault", ["quality", "calibration"])
def test_invalid_bulk_reference_does_not_invalidate_logical_sm_timing(fault):
    r = report_fixture()
    row = next(
        w for w in r["windows"] if w["method"] == "ce_bulk" and w["sampler_enabled"]
    )
    if fault == "quality":
        row["metrics"]["valid"] = False
    else:
        row["calibration"]["valid"] = False
    status = report_status(r)
    assert not status["ready"]
    assert status["comparisons"] == dict(sm=True, ce_same=True, bulk_reference=False)
    text = compact_report(r)
    assert "RATIOS_SM" in text and "RATIOS_CEsame" in text
    assert "BULK_REFERENCE" not in text


@pytest.mark.parametrize("sm_valid", [True, False])
def test_offline_reanalysis_preserves_input_and_reports_scoped_exit_code(
    tmp_path, sm_valid
):
    r = report_fixture()
    for w in r["windows"]:
        if w["method"] == "ce_same" or (not sm_valid and w["method"] == "linear8"):
            w["metrics"]["valid"] = False
    source = json.dumps(r).encode()
    report = tmp_path / "linear.json"
    report.write_bytes(source)
    output = tmp_path / "reanalysis"
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        str(root / "scripts/pd_test/reanalyze_h2d_linear.py"),
        "--report",
        str(report),
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == (0 if sm_valid else 1), result.stderr
    assert report.read_bytes() == source
    meta = json.loads((output / "reanalysis.json").read_text())
    assert meta["comparisons"]["sm"] is sm_valid
    assert meta["full_results_valid"] is False
    assert len(meta["input_report_sha256"]) == 64
    assert "results_valid=False" in (output / "linear-compact.txt").read_text()
    # Reject an existing destination instead of overwriting acquisition/results.
    refused = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert refused.returncode != 0 and "not replaced" in refused.stderr


def test_early_failure_still_has_a_readable_compact_report():
    r = dict(
        success=False, config=dict(mode="check", order=[]), error="CUDA unavailable"
    )
    text = compact_report(r)
    assert "results_valid=False" in text and "CUDA unavailable" in text


@pytest.mark.parametrize("prefix", [128, 136])
def test_byte_oracle_checks_payload_and_both_guard_regions(prefix):
    import torch

    source = torch.arange(264, dtype=torch.int32).to(torch.uint8)
    guard = torch.full((264 + 2 * prefix,), 165, dtype=torch.uint8)
    guard[prefix : prefix + 264] = source
    validate_bytes(torch, guard, source, 264, prefix=prefix)
    for pos in (0, prefix + 33, len(guard) - 1):
        original = guard[pos].item()
        guard[pos] ^= 1
        with pytest.raises(ValueError, match="payload or guard"):
            validate_bytes(torch, guard, source, 264, prefix=prefix)
        guard[pos] = original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
