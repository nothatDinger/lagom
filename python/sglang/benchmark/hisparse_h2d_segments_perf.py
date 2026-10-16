"""CPU-only review binding and statistics for the segmented timing experiment."""

import json
import statistics
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_latency import stats

# Manually reviewed v2 artifacts, including full SASS control words.
REVIEWED = {
    "legacy": "1bb816c524eb665041b323c5826dcc849ea15495d3a45aa207c9cf40154ffa6c",
    "control": "db12f493a0e1566ea09211fd6879fc8e38ea593a98c06034df16b413db9ef00d",
    "trace": "d7d1ed9c2252bd3774c1895c983ba5ccb317ddc92dcd423821960eed7e3d5162",
}
SEGMENTS = ("prep", "issue", "residual", "store")


def verify_codegen(path, root=None):
    """Accept only the already reviewed binaries; never auto-approve a new build."""
    root = root or Path(__file__).resolve().parents[3]
    extraction = json.loads(path.read_text())
    if not extraction.get("success") or not extraction.get("no_spill"):
        raise ValueError("Require successful no-spill extraction")
    artifacts = extraction.get("artifacts", [])
    if (
        len(artifacts) != 3
        or {a["variant"]: a["sha256"] for a in artifacts} != REVIEWED
    ):
        raise ValueError("Artifacts differ from manually reviewed v2")
    if not all(
        a.get("controls_complete") and a.get("timer_presence_valid") for a in artifacts
    ):
        raise ValueError("Incomplete control-word/timer extraction")
    check_path = Path(extraction["check_report"])
    if sha256(check_path) != extraction["input_sha256"]:
        raise ValueError("Correctness report changed")
    check = json.loads(check_path.read_text())
    if not check.get("success") or len(check.get("correctness", [])) != 12:
        raise ValueError("Require completed 12-case check")
    if (
        len(check.get("jit_artifacts", [])) != 3
        or {a["variant"]: a["sha256"] for a in check["jit_artifacts"]} != REVIEWED
    ):
        raise ValueError("Correctness artifact mismatch")
    hashes = check.get("code", {}).get("sha256", {})
    if not hashes:
        raise ValueError("Missing correctness source provenance")
    for name, digest in hashes.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Correctness source changed: {name}")
    return extraction, dict(
        timing_semantics_reviewed=True,
        codegen_sha256=sha256(path),
        artifacts=REVIEWED,
        notes="v2: T2 before W5 wait; 0a90 M0x20 consumes data before T3; "
        "T3/T4 M0x02 address reuse; trace changes scheduling, perturbation required.",
    )


def summarize_invocation(rows):
    """One launch: distributions are per-warp; averages per KV are KV-weighted."""
    if not rows or any(r["kv"] <= 0 for r in rows):
        raise ValueError("This B32/miss100 summary requires active warps")

    def means(group):
        total = sum(r["kv"] for r in group)
        result = dict(kv_per_warp=statistics.fmean(r["kv"] for r in group))
        for name in (*SEGMENTS, "copy"):
            value = sum(r[name + "_ns"] for r in group) / 1000
            result[name + "_per_warp_us"] = value / len(group)
            result[name + "_per_kv_us"] = value / total
        result["loop_per_warp_us"] = statistics.fmean(
            (r["end_ns"] - r["begin_ns"]) / 1000 for r in group
        )
        result["other_per_warp_us"] = (
            result["loop_per_warp_us"] - result["copy_per_warp_us"]
        )
        return result

    return dict(
        total_kv=sum(r["kv"] for r in rows),
        sm_count=len({r["sm"] for r in rows}),
        means=means(rows),
        distributions={
            name: stats([r[name + "_ns"] / 1000 for r in rows])
            for name in (*SEGMENTS, "copy")
        },
        by_kv={
            str(k): dict(
                warps=sum(r["kv"] == k for r in rows),
                means=means([r for r in rows if r["kv"] == k]),
            )
            for k in sorted({r["kv"] for r in rows})
        },
        last_loop_warp=max(rows, key=lambda r: r["end_ns"]),
        envelope_us=(max(r["end_ns"] for r in rows) - min(r["begin_ns"] for r in rows))
        / 1000,
    )
