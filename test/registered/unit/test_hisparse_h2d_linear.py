import copy

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
        "calibration",
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
    elif fault == "calibration":
        r["windows"][-5]["calibration"]["valid"] = False
    elif fault == "quality":
        r["windows"][0]["metrics"]["valid"] = False
    else:
        r["windows"][-2]["metrics"]["logical_wall_GBps"] *= 0.7
    assert not report_status(r)["ready"]
    assert "RATIOS" not in compact_report(r)


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
