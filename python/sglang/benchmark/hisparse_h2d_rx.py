"""NVML sampled-rate analysis, not a PCIe transaction or TLB counter."""

import math
import statistics
import threading
import time

from sglang.benchmark.hisparse_h2d_pair import clock_text


class RxSampler:
    """Read NVML in a host thread; never issue CUDA work from this thread."""

    def __init__(self, reader, period_s, state_reader=None):
        self.reader, self.period_s, self.state_reader = reader, period_s, state_reader
        self.rows, self.states = [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        state_deadline = 0
        while not self.stop_event.is_set():
            before = time.monotonic_ns()
            try:
                rates = self.reader()
                after = time.monotonic_ns()
                self.rows.append(
                    dict(t_ns=(before + after) // 2, call_ns=after - before, **rates)
                )
                if self.state_reader is not None and after >= state_deadline:
                    self.states.append(self.state_reader())
                    state_deadline = after + 200_000_000
            except Exception as exc:
                self.rows.append(
                    dict(t_ns=time.monotonic_ns(), error=f"{type(exc).__name__}: {exc}")
                )
                return
            elapsed = (time.monotonic_ns() - before) / 1e9
            self.stop_event.wait(max(0, self.period_s - elapsed))

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("NVML sampling thread did not stop")


def sampled_rates(rows, lo, hi, *, min_samples, max_gap_ns):
    """Integrate rates on the exact wall interval using linear interpolation.

    Do not extrapolate missing edges or bridge long sampling gaps. NVML rates
    themselves are averaged by the driver; this does not reconstruct TLPs.
    """
    failed = [r["error"] for r in rows if r.get("error")]
    points = [r for r in rows if not r.get("error")]
    count = sum(lo <= r["t_ns"] <= hi for r in points)
    if hi <= lo or failed or len(points) < 2:
        return dict(
            valid=False,
            reason="Missing/failed counter samples",
            errors=failed,
            samples=count,
        )
    if any(b["t_ns"] <= a["t_ns"] for a, b in zip(points, points[1:])):
        return dict(
            valid=False, reason="Counter timestamps are not increasing", samples=count
        )
    rate_keys = ["rx_kbps"]
    if any(r.get("tx_kbps") is not None for r in points):
        rate_keys.append("tx_kbps")
    if any(
        not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) or r[k] < 0
        for r in points
        for k in rate_keys
    ):
        return dict(valid=False, reason="Invalid counter rate", samples=count)
    integrals, rx_square, covered, gaps = dict.fromkeys(rate_keys, 0.0), 0.0, 0, []
    for a, b in zip(points, points[1:]):
        left, right = max(lo, a["t_ns"]), min(hi, b["t_ns"])
        if right <= left:
            continue
        gap = b["t_ns"] - a["t_ns"]
        gaps.append(gap)
        if gap > max_gap_ns:
            continue
        dt = right - left
        covered += dt
        for key in integrals:
            slope = (b[key] - a[key]) / gap
            x, y = a[key] + slope * (left - a["t_ns"]), a[key] + slope * (
                right - a["t_ns"]
            )
            integrals[key] += (x + y) * dt / 2
            if key == "rx_kbps":
                rx_square += (x * x + x * y + y * y) * dt / 3
    coverage = covered / (hi - lo)
    if not covered:
        return dict(
            valid=False, reason="No covered interval", samples=count, coverage=coverage
        )
    means = {k: v / covered for k, v in integrals.items()}
    means.setdefault("tx_kbps", None)
    rx = means["rx_kbps"]
    cv = math.sqrt(max(0, rx_square / covered - rx * rx)) / rx if rx > 0 else None
    valid = count >= min_samples and coverage >= 0.95
    return dict(
        valid=valid,
        samples=count,
        coverage=coverage,
        max_gap_ms=max(gaps) / 1e6,
        rx_cv=cv,
        **means,
        reason=None if valid else "Insufficient samples or interval coverage",
    )


def chunk_metrics(window, trim_s, *, require_steady=False):
    """Timing-only quality check, also used with all NVML polling disabled."""
    chunks = window["chunks"]
    if len(chunks) < 3:
        return dict(valid=False, reason="Too few transfer chunks")
    lo, hi = chunks[0]["begin_ns"] + trim_s * 1e9, chunks[-1]["end_ns"] - trim_s * 1e9
    core = [c for c in chunks if c["begin_ns"] >= lo and c["end_ns"] <= hi]
    if len(core) < 3:
        return dict(valid=False, reason="Too little steady work after trimming")
    lo, hi = core[0]["begin_ns"], core[-1]["end_ns"]
    wall_s, gpu_s = (hi - lo) / 1e9, sum(c["gpu_us"] for c in core) / 1e6
    replays = sum(c["replays"] for c in core)
    payload = window["bytes_per_graph"] * replays
    if (
        any(
            c["replays"] <= 0
            or c["end_ns"] <= c["begin_ns"]
            or c["gpu_us"] <= 0
            or not math.isfinite(c["gpu_us"])
            for c in core
        )
        or gpu_s <= 0
        or wall_s <= 0
        or payload <= 0
    ):
        return dict(valid=False, reason="Invalid GPU/chunk timing")
    if any(b["begin_ns"] < a["end_ns"] for a, b in zip(chunks, chunks[1:])):
        return dict(valid=False, reason="Transfer chunks overlap")
    gpu_wall_ratio = gpu_s / wall_s
    # Device event intervals can include host starvation. Also bound the
    # submission interval; event/wall agreement alone is not a busy metric.
    enqueue_fraction = max(c["enqueue_ns"] / (c["gpu_us"] * 1000) for c in core)
    normalized = [c["gpu_us"] / c["replays"] for c in core]
    timing_cv = statistics.pstdev(normalized) / statistics.fmean(normalized)
    third = max(1, len(core) // 3)
    late_early = statistics.fmean(normalized[-third:]) / statistics.fmean(
        normalized[:third]
    )
    steady = timing_cv <= 0.10 and 0.90 <= late_early <= 1.10
    valid = (
        0.90 <= gpu_wall_ratio <= 1.02
        and 0 <= enqueue_fraction <= 0.10
        and (steady or not require_steady)
    )
    return dict(
        valid=valid,
        reason=(
            None
            if valid
            else "Submission, GPU/wall timing, or steady-state check failed"
        ),
        core_begin_ns=lo,
        core_end_ns=hi,
        core_seconds=wall_s,
        core_replays=replays,
        core_chunks=len(core),
        logical_bytes=payload,
        logical_wall_GBps=payload / wall_s / 1e9,
        logical_gpu_GBps=payload / gpu_s / 1e9,
        gpu_wall_ratio=gpu_wall_ratio,
        max_enqueue_fraction=enqueue_fraction,
        chunk_gpu_cv=timing_cv,
        late_over_early_gpu=late_early,
    )


def window_metrics(window, period_s, trim_s, *, require_steady=False):
    timing = chunk_metrics(window, trim_s, require_steady=require_steady)
    if "core_begin_ns" not in timing:
        return timing
    rates = sampled_rates(
        window["samples"],
        timing["core_begin_ns"],
        timing["core_end_ns"],
        min_samples=20,
        max_gap_ns=max(0.08, period_s * 4) * 1e9,
    )
    if not rates["valid"]:
        return dict(timing, valid=False, reason=rates["reason"], rates=rates)
    stable = rates["rx_cv"] is not None and rates["rx_cv"] <= 0.15
    idle_rows = window["idle_samples"]
    idle = (
        sampled_rates(
            idle_rows,
            idle_rows[0]["t_ns"] + 150e6,
            idle_rows[-1]["t_ns"] - 50e6,
            min_samples=5,
            max_gap_ns=max(0.08, period_s * 4) * 1e9,
        )
        if idle_rows
        else dict(valid=False)
    )
    idle_ratio = (
        idle.get("rx_kbps", math.inf) / rates["rx_kbps"]
        if rates["rx_kbps"] > 0
        else None
    )
    background_ok = idle_ratio is not None and idle_ratio <= 0.05
    return dict(
        timing,
        valid=timing["valid"] and stable and idle["valid"] and background_ok,
        rates=rates,
        idle=idle,
        rx_GBps_1024=rates["rx_kbps"] * 1024 / 1e9,
        tx_GBps_1024=(
            rates["tx_kbps"] * 1024 / 1e9 if rates["tx_kbps"] is not None else None
        ),
        idle_over_active=idle_ratio if idle["valid"] else None,
        reason=(
            timing["reason"]
            if not timing["valid"]
            else (
                None
                if stable and idle["valid"] and background_ok
                else "Unstable Rx, invalid idle samples, or high background"
            )
        ),
    )


def ce_calibration(metrics):
    if not metrics.get("valid"):
        return dict(valid=False, reason="Invalid measurement window")
    ratio = metrics["rx_GBps_1024"] / metrics["logical_wall_GBps"]
    if metrics["rates"]["rx_kbps"] <= 0:
        return dict(valid=False, reason="CE Rx is zero")
    idle_ratio = metrics["idle"]["rx_kbps"] / metrics["rates"]["rx_kbps"]
    valid = (
        0.85 <= ratio <= 1.15
        and idle_ratio <= 0.05
        and metrics["logical_wall_GBps"] >= 45
    )
    return dict(
        valid=valid,
        nominal_rx_over_logical=ratio,
        idle_over_active=idle_ratio,
        reason=(
            None if valid else "CE Rx mismatch, high idle traffic, or slow CE reference"
        ),
    )


def sampler_controls(report):
    """Require both sampled runs and both unpolled brackets, never a lucky run."""
    controls = {}
    device = report["config"].get("check_device", 0)
    for method in ("ce_bulk", "sm"):
        rows = [w for w in report.get("windows", []) if w["method"] == method]
        if (
            len(rows) != 4
            or [w.get("sampler_enabled") for w in rows] != [False, True, True, False]
            or [w.get("trial") for w in rows] != list(range(4))
            or any(
                w.get("device") != device
                or not w.get("metrics", {}).get("core_replays")
                for w in rows
            )
        ):
            controls[method] = dict(
                valid=False,
                reason="Missing, invalid or reordered OFF/ON/ON/OFF windows",
            )
            continue
        us = [
            w["metrics"]["core_seconds"] * 1e6 / w["metrics"]["core_replays"]
            for w in rows
        ]
        baseline = statistics.fmean([us[0], us[3]])
        ratios = [us[1] / baseline, us[2] / baseline]
        drift = us[3] / us[0]
        quality = all(w["metrics"]["valid"] for w in rows)
        controls[method] = dict(
            valid=quality
            and 0.90 <= drift <= 1.10
            and all(0.90 <= r <= 1.10 for r in ratios),
            windows_quality=quality,
            bank_sequence_wall_us=us,
            on_over_off=ratios,
            off_late_over_early=drift,
        )
    return controls


def sampler_check_report(report):
    windows, checked = report.get("windows", []), report.get("correctness", [])
    device = report["config"].get("check_device", 0)
    controls = sampler_controls(report)
    correct = (
        len(checked) == 2
        and {(r["device"], r["method"]) for r in checked}
        == {(device, "sm"), (device, "ce_bulk")}
        and all(r["payload"] is True and r["graph"] is True for r in checked)
        and all(r["untouched"] is True for r in checked if r["method"] == "sm")
    )
    graphs = report.get("chunk_graphs", [])
    graph_ok = (
        len(graphs) == 2
        and {g.get("method") for g in graphs} == {"sm", "ce_bulk"}
        and all(
            g.get("correctness") is True
            and g.get("device") == device
            and g.get("capture_device") == device
            and g.get("per_copy_timing_events") == 0
            and g.get("launches_per_chunk") == 1
            for g in graphs
        )
    )
    ce = [w for w in windows if w["method"] == "ce_bulk" and w.get("sampler_enabled")]
    cal = sum(w.get("calibration", {}).get("valid") is True for w in ce)
    ready = (
        report.get("success") is True
        and report.get("counters_available") is True
        and correct
        and graph_ok
        and cal == 2
        and len(ce) == 2
        and len(windows) == 8
        and all(c["valid"] for c in controls.values())
    )
    lines = [
        f"NVML sampler self-check: CUDA d{device} only; OFF/ON/ON/OFF; identical event-free chunk graph within each method.",
        f'success={report["success"]} counters_available={report.get("counters_available")} correctness={len(checked)}/2 '
        f"correctness_valid={correct} chunk_graphs_valid={graph_ok} windows={len(windows)}/8 CE_calibration={cal}/2 results_valid={ready}",
        f'CONFIG B32/miss100 SM=legacy_CTA4/1024 banks=64 CE=64MiB/7banks register_flags={report.get("register_flags")} '
        f'duration_s={report["config"]["duration_s"]} trim_s={report["config"]["trim_s"]} sample_ms={report["config"]["sample_ms"]} RX_only=True',
        f'CODE git={report.get("code",{}).get("git")} JIT={report.get("jit_artifacts")}',
    ]
    if report.get("error"):
        lines.append(f'ERROR active={report.get("active_case")} {report["error"]}')
    for d, meta in sorted(report.get("devices", {}).items()):
        lines.append(
            f'IDENTITY d{d} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
        )
    for buf in report.get("host_snapshot", {}).get("buffers", []):
        lines.append(
            f'HOST bytes={buf["nbytes"]} NUMA={[r.get("numa") for r in buf["regions"]]}'
        )
    for graph in graphs:
        lines.append(f"GRAPH {graph}")
    for w in windows:
        m = w["metrics"]
        text = f'WINDOW {w["method"]} trial={w["trial"]} sampler={"ON" if w["sampler_enabled"] else "OFF"} valid={m["valid"]}'
        for key in (
            "logical_wall_GBps",
            "max_enqueue_fraction",
            "chunk_gpu_cv",
            "late_over_early_gpu",
        ):
            if key in m:
                text += f" {key}={m[key]:.4f}"
        rates = m.get("rates", {})
        if rates.get("rx_kbps") is not None:
            text += f' raw_rx_KBps={rates["rx_kbps"]:.1f} n={rates["samples"]} coverage={rates["coverage"]:.3f} CV={rates["rx_cv"]}'
        calls = [r["call_ns"] / 1e6 for r in w.get("samples", []) if "call_ns" in r]
        if calls:
            text += f" query_ms_min/med/max={[round(v,3) for v in (min(calls),statistics.median(calls),max(calls))]}"
        if not m["valid"]:
            text += f' reason={m.get("reason")} '
        lines.append(text)
    for method, control in controls.items():
        lines.append(f"CONTROL {method} {control}")
    states = [s for w in windows for s in w.get("states", [])]
    lines.append(
        f"STATE d{device} "
        + (clock_text([dict(before=s) for s in states]) if states else "not_collected")
    )
    lines.append(
        "This checks the measurement process. Do not infer production H2D root cause or run the old main from this result."
    )
    return "\n".join(lines) + "\n"


def rx_compact_report(report):
    if report["config"].get("sampler_check"):
        return sampler_check_report(report)
    config, windows = report["config"], report.get("windows", [])
    expected_keys = {
        (d, t, m) for t, d in enumerate(config["order"]) for m in ("sm", "ce_bulk")
    }
    keys = [(w["device"], w["trial"], w["method"]) for w in windows]
    complete = len(keys) == len(expected_keys) and set(keys) == expected_keys
    checked = report.get("correctness", [])
    correct = (
        len(checked) == 4
        and {(r["device"], r["method"]) for r in checked}
        == {(d, m) for d in config["devices"] for m in ("sm", "ce_bulk")}
        and all(r["payload"] is True and r["graph"] is True for r in checked)
        and all(r["untouched"] is True for r in checked if r["method"] == "sm")
    )
    ce = [w for w in windows if w["method"] == "ce_bulk"]
    calibrated = len(ce) == len(config["order"]) and all(
        w.get("calibration", {}).get("valid") is True for w in ce
    )
    quality = complete and all(
        w.get("metrics", {}).get("valid") is True for w in windows
    )
    ready = (
        report.get("success") is True
        and report.get("counters_available") is True
        and correct
        and quality
        and calibrated
    )
    lines = [
        "Sustained legacy host gather / bulk CE with NVML Rx; sampled link rates, not transaction counts.",
        f'success={report["success"]} counters_available={report.get("counters_available")} correctness={len(checked)}/4 '
        f"correctness_valid={correct} windows={len(windows)}/{len(expected_keys)} quality={quality} "
        f'CE_calibration={sum(w.get("calibration",{}).get("valid") is True for w in ce)}/{len(config["order"])} results_valid={ready}',
        f"CONFIG B32/miss100 source=512MiB SM=CTA4/1024 banks=64 bulk=64MiB "
        f'duration_s={config["duration_s"]} trim_s={config["trim_s"]} sample_ms={config["sample_ms"]} order={config["order"]}',
        "NVML raw units are reported KB/s. GBps_1024 uses nominal 1024 B/KB; CE checks that conversion, never rescales counters to force agreement.",
    ]
    if report.get("error"):
        lines.append(f'ERROR active={report.get("active_case")} ' + report["error"])
    for d, meta in sorted(report.get("devices", {}).items()):
        lines.append(
            f'IDENTITY d{d} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
        )
    lines.append(
        f'CODE git={report.get("code",{}).get("git")} JIT={report.get("jit_artifacts")}'
    )
    for buf in report.get("host_snapshot", {}).get("buffers", []):
        lines.append(
            f'HOST bytes={buf["nbytes"]} mapped={buf["mapped_bytes"]} NUMA={[r.get("numa") for r in buf["regions"]]}'
        )
    for w in windows:
        m = w.get("metrics", {})
        if not m.get("valid"):
            lines.append(
                f'WINDOW d{w["device"]} trial={w["trial"]} {w["method"]} valid=False metrics={m}'
            )
            continue
        lines.append(
            f'WINDOW d{w["device"]} trial={w["trial"]} {w["method"]} '
            f'logical_wall/gpu_GBps={m["logical_wall_GBps"]:.3f}/{m["logical_gpu_GBps"]:.3f} '
            f'rx/tx_GBps_1024={m["rx_GBps_1024"]:.3f}/{m["tx_GBps_1024"]:.3f} '
            f'raw_rx_KBps={m["rates"]["rx_kbps"]:.1f} idle_rx_KBps={m["idle"]["rx_kbps"]:.1f} '
            f'rx/logical={m["rx_GBps_1024"]/m["logical_wall_GBps"]:.3f} '
            f'core_s={m["core_seconds"]:.3f} n={m["rates"]["samples"]} coverage={m["rates"]["coverage"]:.3f} '
            f'CV={m["rates"]["rx_cv"]:.3f} gpu/wall={m["gpu_wall_ratio"]:.3f} '
            f'enqueue_max={m["max_enqueue_fraction"]:.3f} '
            f'CE_check={w.get("calibration", {}).get("valid", "not_applicable")}'
        )
    indexed = {(w["device"], w["trial"], w["method"]): w for w in windows}
    if ready:
        for d in config["devices"]:
            pairs = [
                (
                    indexed[(d, t, "sm")]["metrics"],
                    indexed[(d, t, "ce_bulk")]["metrics"],
                )
                for t, current in enumerate(config["order"])
                if current == d
            ]
            rx_ratios = [
                a["rates"]["rx_kbps"] / b["rates"]["rx_kbps"] for a, b in pairs
            ]
            logical_ratios = [
                a["logical_wall_GBps"] / b["logical_wall_GBps"] for a, b in pairs
            ]
            lines.append(
                f"PAIRS d{d} SM/CE_rx={[round(r,4) for r in rx_ratios]} "
                f"SM/CE_logical={[round(r,4) for r in logical_ratios]} "
                f"Rx_mean_ratio={statistics.fmean(rx_ratios):.4f}"
            )
    for d in config["devices"]:
        states = [s for w in windows if w["device"] == d for s in w.get("states", [])]
        lines.append(f"STATE d{d} " + clock_text([dict(before=s) for s in states]))
    lines.append(
        "Logical rates include graph events and host gaps on the SAME steady wall interval as Rx. Rates are approximate aggregates; CE does not prove SM-counter coverage or TLP shapes. No conclusion from failed calibration or unstable sampling."
    )
    return "\n".join(lines) + "\n"
