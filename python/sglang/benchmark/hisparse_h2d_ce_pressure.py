"""Paced CE interference. Pulsed traffic is not a hardware rate limiter."""

import json
import math
import statistics
import time

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_sm_ce import mixed_overlap

RATES = (5, 15, 30)  # Decimal GB/s requested; actual rate is measured.


def core_metrics(sm, ce, sm_bytes, ce_bytes):
    """Exact SM blocks, bounded CE bytes across the SAME central GPU interval."""
    if len(sm) < 8 or any(b <= a for a, b in sm + ce):
        raise ValueError("Insufficient/nonpositive GPU intervals")
    lo0, hi0 = sm[0][0], sm[-1][1]
    lo, hi = lo0 + (hi0 - lo0) * 0.2, lo0 + (hi0 - lo0) * 0.8
    kept = [(a, b) for a, b in sm if a >= lo and b <= hi]
    if len(kept) < 4:
        raise ValueError("Insufficient central SM blocks")
    lo, hi = kept[0][0], kept[-1][1]
    complete = [(a, b) for a, b in ce if a >= lo and b <= hi]
    intersecting = [(a, b) for a, b in ce if a < hi and b > lo]
    width = hi - lo
    return dict(
        core_us=width,
        core_start_us=lo,
        core_end_us=hi,
        sm_blocks=len(kept),
        sm_GBps=len(kept) * sm_bytes / width / 1000,
        sm_event_coverage=sum(b - a for a, b in kept) / width,
        ce_complete=len(complete),
        ce_intersecting=len(intersecting),
        ce_GBps_lower=len(complete) * ce_bytes / width / 1000,
        ce_GBps_upper=len(intersecting) * ce_bytes / width / 1000,
        ce_event_duty=sum(min(b, hi) - max(a, lo) for a, b in intersecting) / width,
    )


class SMWindow:
    """One submission keeps SM work queued while the CPU paces CE."""

    def __init__(self, torch, sm, repeats):
        self.origin = torch.cuda.Event(enable_timing=True, external=True)
        self.pairs = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(repeats)
        ]
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=sm.stream):
            self.origin.record()
            for a, b in self.pairs:
                a.record()
                sm.graph.replay()
                b.record()
        with torch.cuda.stream(sm.stream):
            self.graph.replay()
        torch.cuda.synchronize()
        sm.check(sm.banks - 1)


def window(torch, readers, rate, timeline):
    sm, ce = readers["A"], readers["B"]
    origin, pairs = timeline.origin, timeline.pairs
    pulses = []
    enqueue_started = time.monotonic()
    done = torch.cuda.Event()
    with torch.cuda.stream(sm.stream):
        timeline.graph.replay()
        done.record()
    enqueue_s = time.monotonic() - enqueue_started
    period = ce.source.numel() / (rate * 1e9) if rate else 0.001
    next_due = time.monotonic()
    skipped = 0
    while not done.query():
        now = time.monotonic()
        if now < next_due:
            time.sleep(min(next_due - now, 0.001))
            continue
        if rate:
            if pulses and not pulses[-1][1].query():
                skipped += 1  # Never queue a backlog or catch-up burst.
            else:
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                with torch.cuda.stream(ce.stream):
                    ce.stream.wait_event(origin)
                    a.record()
                    ce.graph.replay()
                    b.record()
                pulses.append((a, b))
        next_due += period
        if next_due < time.monotonic():
            next_due = time.monotonic() + period
            skipped += 1
    torch.cuda.synchronize()

    def intervals(pairs):
        return [
            (origin.elapsed_time(a) * 1000, origin.elapsed_time(b) * 1000)
            for a, b in pairs
        ]

    sm_spans, ce_spans = intervals(pairs), intervals(pulses)
    metrics = core_metrics(sm_spans, ce_spans, sm.payload_bytes, ce.source.numel())
    metrics.update(
        target_GBps=rate,
        ce_pulses=len(pulses),
        skipped=skipped,
        enqueue_fraction=enqueue_s * 1e6 / sm_spans[-1][1],
    )
    # Rate check is conservative: both CE-byte bounds must be near target.
    metrics["rate_valid"] = rate == 0 or (
        metrics["ce_complete"] >= 4
        and metrics["ce_GBps_lower"] >= rate * 0.8
        and metrics["ce_GBps_upper"] <= rate * 1.2
    )
    metrics["valid"] = (
        metrics["rate_valid"]
        and metrics["enqueue_fraction"] <= 0.1
        and metrics["sm_event_coverage"] >= 0.98
        and all(math.isfinite(v) for v in metrics.values() if isinstance(v, float))
    )
    return dict(metrics=metrics, sm_intervals_us=sm_spans, ce_intervals_us=ce_spans)


def execute_pressure(torch, args, report, readers, warmer, observer):
    repeats = 32 if args.mode == "smoke" else 128
    trials = 1 if args.mode == "smoke" else 4
    report["pressure_config"] = dict(
        rates=RATES,
        repeats=repeats,
        trials=trials,
        trim_fraction=0.2,
        ce_copy_bytes=readers["B"].source.numel(),
        ce_copies_per_pulse=readers["B"].banks,
        pacing="CPU deadlines, at most one CE pulse in flight; no catch-up",
    )
    report["pressure_waves"] = []
    report["pressure_profiles"] = []
    timeline = SMWindow(torch, readers["A"], repeats)
    profile_timeline = timeline if repeats == 32 else SMWindow(torch, readers["A"], 32)
    report["pressure_config"]["single_submission"] = True
    # Each pressure window has its own before/after zero-CE baseline.
    for trial in range(trials):
        rates = RATES if trial % 2 == 0 else tuple(reversed(RATES))
        for rate in rates:
            group = []
            for phase, requested in (("off_before", 0), ("on", rate), ("off_after", 0)):
                warmer.run(250)
                before = observer.sample(0)
                row = window(torch, readers, requested, timeline)
                row.update(
                    trial=trial,
                    rate=rate,
                    phase=phase,
                    state_before=before,
                    state_after=observer.sample(0),
                )
                readers["A"].check(readers["A"].banks - 1)
                readers["B"].check(0)
                row["correctness"] = True
                report["pressure_waves"].append(row)
                group.append(row)
            baseline = statistics.mean(
                w["metrics"]["sm_GBps"] for w in (group[0], group[2])
            )
            group[1]["retention"] = group[1]["metrics"]["sm_GBps"] / baseline
            group[1]["baseline_after_before"] = (
                group[2]["metrics"]["sm_GBps"] / group[0]["metrics"]["sm_GBps"]
            )
    # Separate structural traces: profiling may perturb CPU pacing. Do NOT use
    # these rates as the unprofiled rates or require target-rate matching here.
    for rate in RATES:
        warmer.run(250)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            row = window(torch, readers, rate, profile_timeline)
        path = args.output_dir / f"pressure-{rate}.trace.json"
        prof.export_chrome_trace(str(path))
        activity = mixed_overlap(
            json.loads(path.read_text()),
            32 * readers["A"].banks,
            row["metrics"]["ce_pulses"],
            readers["B"].source.numel(),
            True,
        )
        report["pressure_profiles"].append(
            dict(
                rate=rate,
                activity=activity,
                trace=path.name,
                sha256=sha256(path),
                profiled_metrics=row["metrics"],
            )
        )
    waves = report["pressure_waves"]
    report["success"] = True
    report["results_valid"] = (
        len(waves) == trials * len(RATES) * 3
        and len(report["pressure_profiles"]) == 3
        and all(w["metrics"]["valid"] for w in waves)
        and all(
            0.9 <= w["baseline_after_before"] <= 1.1
            for w in waves
            if w["phase"] == "on"
        )
        and all(
            not w[k]["errors"]
            and w[k].get("pstate") == 0
            and w[k].get("pcie_gen") == 5
            and w[k].get("pcie_width") == 16
            for w in waves
            for k in ("state_before", "state_after")
        )
    )


def compact_pressure(r):
    lines = [
        f"PRESSURE mode={r['mode']} success={r['success']} results_valid={r['results_valid']} checks={len(r['checks'])}/8 profiles={len(r.get('pressure_profiles', []))}/3",
        f"IDENTITY {r.get('identity')} HOST node1={r.get('host_node1')}",
        f"BINDINGS {r.get('bindings')}",
        f"JIT {r.get('jit')}",
        f"CONFIG {r.get('config')} PRESSURE_CONFIG {r.get('pressure_config')} CE_calibration={r.get('ce_calibration')}",
    ]
    for w in r.get("pressure_waves", []):
        lines.append(
            f"WINDOW trial={w['trial']} rate={w['rate']} phase={w['phase']} metrics={w['metrics']} retention={w.get('retention')} drift={w.get('baseline_after_before')} correct={w['correctness']}"
        )
        lines.append(
            "STATE "
            + str(
                {
                    k: [w[s].get(k) for s in ("state_before", "state_after")]
                    for k in ("sm_MHz", "power_W", "temperature_C", "throttle_bits")
                }
            )
        )
    for p in r.get("pressure_profiles", []):
        lines.append(
            f"ACTIVITY target={p['rate']} {p['activity']} profiled_metrics={p['profiled_metrics']}"
        )
    if "error" in r:
        lines.append("ERROR " + r["error"])
    lines.append(
        "CE is pulsed, not continuously rate limited. Low average traffic does not exclude instantaneous link contention."
    )
    return "\n".join(lines) + "\n"
