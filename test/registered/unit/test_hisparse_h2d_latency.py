import copy
import json
from pathlib import Path

import pytest
from sglang.benchmark.hisparse_h2d_latency import (
    VARIANTS,
    perturbation,
    summarize_invocation,
    validate_records,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def records(blocks, counts, real):
    warps = blocks * 32
    return (
        [[-1] * 9]
        + [
            [
                w // 32,
                w % 32,
                w,
                w // 32,
                1000 + w,
                9000 + w,
                2000 if w < sum(counts[:real]) else 0,
                len(range(w, sum(counts[:real]), warps)),
                real,
            ]
            for w in range(warps)
        ]
        + [[-1] * 9]
    )


@pytest.mark.parametrize("blocks", [4, 8])
@pytest.mark.parametrize(
    "counts,real",
    [
        ([100] * 32, 32),
        ([0] * 65, 65),
        ([0, 1, 65, 100, 127, 128, 129] * 10, 33),
        ([1272] * 65, 0),
    ],
)
def test_schedule_and_guard_validation(blocks, counts, real):
    rows = validate_records(records(blocks, counts, real), blocks, counts, real)
    assert sum(r["kv"] for r in rows) == sum(counts[:real])
    assert all(r["visits"] == real for r in rows)
    if counts == [100] * 32:
        assert [sum(r["kv"] for r in rows if r["cta"] == c) for c in range(blocks)] == (
            [800] * 4 if blocks == 4 else [416] * 4 + [384] * 4
        )


@pytest.mark.parametrize(
    "field,value", [(0, 999), (3, -1), (4, -1), (5, 999), (6, 9000), (7, 999), (8, 999)]
)
def test_invalid_records_fail(field, value):
    rows = records(4, [100] * 32, 32)
    rows[1][field] = value
    with pytest.raises(ValueError):
        validate_records(rows, 4, [100] * 32, 32)


def test_guards_and_uninstrumented_records():
    rows = [[-1] * 9 for _ in range(130)]
    assert validate_records(rows, 4, [100] * 32, 32, False) == []
    rows[0][0] = 0
    with pytest.raises(ValueError):
        validate_records(rows, 4, [100] * 32, 32, False)


def test_region_subtraction_before_quantiles_and_no_warp_sum():
    rows = validate_records(records(4, [100] * 32, 32), 4, [100] * 32, 32)
    rows[0]["copy_ns"] = 7000
    result = summarize_invocation(rows)
    assert result["metrics"]["other_us"]["p50"] == 6
    assert result["metrics"]["loop_us"]["max"] == 8
    assert result["envelope_us"] == 8.127
    assert result["sm_count"] == 4


def waves():
    return [
        dict(trial=t, variant=v, blocks=b, copy_us=[60.0] * 64)
        for t in range(6)
        for v in VARIANTS
        for b in (4, 8)
    ]


def test_perturbation_requires_per_wave_agreement_not_just_mean():
    data = waves()
    assert perturbation(data)["valid"]
    for r in data:
        if r["variant"] == "trace" and r["trial"] in (0, 1):
            r["copy_us"] = [70.0] * 64
    assert not perturbation(data)["valid"]


def test_grid_ratio_guard_and_missing_wave():
    data = waves()
    for r in data:
        if r["variant"] == "trace":
            r["copy_us"] = [61.5 if r["blocks"] == 8 else 58.5] * 64
    assert not perturbation(data)["gates"]["relative_grid_ratio"]["valid"]
    with pytest.raises(ValueError):
        perturbation(data[:-1])
    data[-1] = copy.deepcopy(data[0])
    with pytest.raises(ValueError):
        perturbation(data)


def test_jit_exports_concrete_template_specializations(monkeypatch):
    import sglang.kernels.jit.utils as jit
    from sglang.benchmark.hisparse_h2d_latency import module

    calls = []
    monkeypatch.setattr(jit, "load_jit", lambda *a, **kw: calls.append((a, kw)))
    module.cache_clear()
    try:
        module(False)
        module(True)
        assert calls[0][1]["cuda_wrappers"] == [
            ("copy_cache_latency", "copy_cache_latency<1024, false>")
        ]
        assert calls[1][1]["cuda_wrappers"] == [
            ("copy_cache_latency", "copy_cache_latency<1024, true>")
        ]
        assert calls[0][0] != calls[1][0]
    finally:
        module.cache_clear()


def test_review_binds_check_source_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "scripts/pd_test")
    )
    from bench_h2d_latency import verify_review
    from sglang.benchmark.hisparse_h2d_codegen import sha256

    code = {"sha256": {"kernel": "abc"}}
    artifacts = [{"variant": v, "sha256": v} for v in VARIANTS]
    check = tmp_path / "check.json"
    check.write_text(json.dumps(dict(success=True, code=code, jit_artifacts=artifacts)))
    extraction = tmp_path / "codegen.json"
    extraction.write_text(
        json.dumps(
            dict(
                success=True,
                no_spill=True,
                check_report=str(check),
                input_sha256=sha256(check),
                artifacts=artifacts,
            )
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            dict(
                timing_semantics_reviewed=True,
                notes="Reviewed timer/consumer order",
                codegen_report=str(extraction),
                codegen_sha256=sha256(extraction),
            )
        )
    )
    verify_review(review, code)
    with pytest.raises(ValueError, match="code changed"):
        verify_review(review, {"sha256": {"kernel": "changed"}})
    check.write_text(check.read_text() + "\n")
    with pytest.raises(ValueError, match="report changed"):
        verify_review(review, code)


def test_unreviewed_template_does_not_enable_main(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "scripts/pd_test")
    )
    from bench_h2d_latency import verify_review

    p = tmp_path / "review.json"
    p.write_text(json.dumps(dict(timing_semantics_reviewed=False, notes="")))
    with pytest.raises(ValueError, match="Require completed"):
        verify_review(p, {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
