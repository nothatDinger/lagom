"""CPU-only review binding and statistics for the independent-window timing experiment."""

import json
import statistics
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_latency import stats

# Manually reviewed windows-v1 artifacts, including full SASS control words.
REVIEWED = {
    "legacy": "1bb816c524eb665041b323c5826dcc849ea15495d3a45aa207c9cf40154ffa6c",
    "control": "4ac41b70a9f904b81207cd2fe1137a2498a119365e35bcef7fe81178cfcb9eba",
    "issue": "3c90ff898c0fe1ed5bb6856278429c793fabbd4207d0bc7084db5b6e26af16dd",
    "tail": "2436d368af016201e0fe82871e807058e5e5d406c534d1c4d7d658bb0ce70466",
}
# Four-position Williams design; repeat with opposite within-variant grid order.
_BASE = (
    ("legacy", "control", "tail", "issue"),
    ("control", "issue", "legacy", "tail"),
    ("issue", "tail", "control", "legacy"),
    ("tail", "legacy", "issue", "control"),
)
ORDERS = _BASE + _BASE
PROBES = ("issue", "tail")


def verify_codegen(path, root=None):
    """Accept only the already reviewed binaries; never auto-approve a new build."""
    root = root or Path(__file__).resolve().parents[3]
    extraction = json.loads(path.read_text())
    if not extraction.get("success") or not extraction.get("no_spill"):
        raise ValueError("Require successful no-spill extraction")
    artifacts = extraction.get("artifacts", [])
    if (
        len(artifacts) != 4
        or {a["variant"]: a["sha256"] for a in artifacts} != REVIEWED
    ):
        raise ValueError("Artifacts differ from manually reviewed windows-v1")
    if not all(
        a.get("controls_complete") and a.get("timer_presence_valid") for a in artifacts
    ):
        raise ValueError("Incomplete control-word/timer extraction")
    check_path = Path(extraction["check_report"])
    if sha256(check_path) != extraction["input_sha256"]:
        raise ValueError("Correctness report changed")
    check = json.loads(check_path.read_text())
    if not check.get("success") or len(check.get("correctness", [])) != 16:
        raise ValueError("Require completed 16-case check")
    if (
        len(check.get("jit_artifacts", [])) != 4
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
        notes="issue 0770..0860 has no W5 data wait; tail 0860..08c0 waits W5 at "
        "0890. Both group three loads on W5 unlike control; interval placement "
        "is understood, production attribution requires perturbation gates.",
    )


def summarize_invocation(rows):
    if not rows or any(r["kv"] <= 0 for r in rows):
        raise ValueError("Require active B32/miss100 warps")

    def means(group):
        total = sum(r["kv"] for r in group)
        interval = sum(r["interval_ns"] for r in group) / 1000
        return dict(
            kv_per_warp=total / len(group),
            interval_per_warp_us=interval / len(group),
            interval_per_kv_us=interval / total,
            loop_per_warp_us=statistics.fmean(
                (r["end_ns"] - r["begin_ns"]) / 1000 for r in group
            ),
        )

    return dict(
        total_kv=sum(r["kv"] for r in rows),
        sm_count=len({r["sm"] for r in rows}),
        means=means(rows),
        distributions=stats([r["interval_ns"] / 1000 for r in rows]),
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


def perturbation(waves):
    import math

    expected = {(t, v, b) for t in range(8) for v in REVIEWED for b in (4, 8)}
    keys = [(w["trial"], w["variant"], w["blocks"]) for w in waves]
    if len(keys) != 64 or set(keys) != expected:
        raise ValueError("Require 64 unique balanced waves")
    if any(
        len(w["copy_us"]) != 64
        or any(not math.isfinite(x) or x <= 0 for x in w["copy_us"])
        for w in waves
    ):
        raise ValueError("Require 64 finite positive events per wave")
    data = {k: statistics.fmean(w["copy_us"]) for k, w in zip(keys, waves)}
    gates = {}

    def gate(name, values, tolerance):
        hits = sum(1 - tolerance <= x <= 1 + tolerance for x in values)
        median = statistics.median(values)
        gates[name] = dict(
            ratios=values,
            median=median,
            within=hits,
            required=7,
            tolerance=tolerance,
            valid=1 - tolerance <= median <= 1 + tolerance and hits >= 7,
        )

    for b in (4, 8):
        gate(
            f"control/legacy_CTA{b}",
            [data[t, "control", b] / data[t, "legacy", b] for t in range(8)],
            0.03,
        )
        for probe in PROBES:
            gate(
                f"{probe}/control_CTA{b}",
                [data[t, probe, b] / data[t, "control", b] for t in range(8)],
                0.05,
            )
    for probe in PROBES:
        gate(
            f"relative_grid_ratio_{probe}",
            [
                (data[t, probe, 8] / data[t, probe, 4])
                / (data[t, "legacy", 8] / data[t, "legacy", 4])
                for t in range(8)
            ],
            0.03,
        )
    by_probe = {
        p: all(
            gates[k]["valid"]
            for k in [
                "control/legacy_CTA4",
                "control/legacy_CTA8",
                f"{p}/control_CTA4",
                f"{p}/control_CTA8",
                f"relative_grid_ratio_{p}",
            ]
        )
        for p in PROBES
    }
    return dict(valid=all(by_probe.values()), by_probe=by_probe, gates=gates)
