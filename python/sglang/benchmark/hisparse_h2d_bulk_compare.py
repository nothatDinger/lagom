"""Validation and reporting for same-buffer 64MiB SM/CE copies on GPU4."""

import math
import statistics
from pathlib import Path

from sglang.benchmark.hisparse_h2d_device_order import HOST_BYTES, devices, local_host
from sglang.benchmark.hisparse_h2d_linear import LINEAR_LIBRARY
from sglang.benchmark.hisparse_h2d_pair import clock_text
from sglang.benchmark.hisparse_h2d_rx import ce_calibration

COPY_BYTES = 64 * 2**20
METHODS = ("sm64", "ce64")
ORDER = ("sm64", "ce64", "ce64", "sm64")
# Actual loaded linear artifact from the passed round-20 GPU4 run.
REFERENCE_SHA256 = "bb5529b24000709ef1ad0f0cf89b7c1138e9def24088037f3e271cf69e9f6042"
CONFIG = dict(
    device=0,
    physical_device=4,
    copy_bytes=COPY_BYTES,
    host_bytes=HOST_BYTES,
    banks=7,
    waves=12,
    num_blocks=4,
    block_size=1024,
    register_flags=1,
    order=list(ORDER),
    warm_s=2.0,
    duration_s=2.5,
    trim_s=0.3,
    rx_sampler=False,
)
# Balanced method order, with OFF/ON/ON/OFF brackets for each method.
RX_ORDER = ("sm64", "ce64", "ce64", "sm64", "sm64", "ce64", "ce64", "sm64")
RX_ENABLED = (False, False, True, True, True, True, False, False)
RX_CONFIG = dict(
    CONFIG,
    order=list(RX_ORDER),
    rx_sampler=True,
    sampler_order=list(RX_ENABLED),
    sample_ms=50.0,
    sampler_tolerance=0.03,
)
BINDINGS = {
    "same_sources",
    "same_target",
    "same_guard",
    "same_bytes_banks",
    "same_chunk_work",
    "aligned_128",
    "host_pinned",
}


def rates(report, method, key="logical_wall_GBps"):
    return [w["metrics"][key] for w in report["windows"] if w["method"] == method]


def rx_controls(report):
    controls = {}
    for method in METHODS:
        rows = [w for w in report.get("windows", []) if w["method"] == method]
        values = [w["metrics"].get("logical_wall_GBps") for w in rows]
        if (
            len(rows) != 4
            or [w.get("sampler_enabled") for w in rows] != [False, True, True, False]
            or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for v in values
            )
        ):
            controls[method] = dict(valid=False, reason="Incomplete sampler brackets")
            continue
        # Same bytes in every window: inverse rate is proportional to time.
        times = [1 / v for v in values]
        baseline = statistics.fmean([times[0], times[3]])
        ratios = [times[1] / baseline, times[2] / baseline]
        drift = times[3] / times[0]
        quality = all(w["metrics"].get("valid") is True for w in rows)
        controls[method] = dict(
            valid=quality and all(0.97 <= r <= 1.03 for r in [drift, *ratios]),
            windows_quality=quality,
            on_over_off=ratios,
            off_late_over_early=drift,
        )
    return controls


def rx_window_valid(window):
    m = window.get("metrics", {})
    r, idle = m.get("rates", {}), m.get("idle", {})
    raw, nominal = r.get("rx_kbps"), m.get("rx_GBps_1024")
    return (
        m.get("valid") is True
        and r.get("valid") is True
        and idle.get("valid") is True
        and all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            for v in (raw, nominal, m.get("logical_wall_GBps"))
        )
        and math.isclose(nominal, raw * 1024 / 1e9, rel_tol=1e-9)
        and r.get("samples", 0) >= 20
        and r.get("coverage", 0) >= 0.95
        and isinstance(r.get("rx_cv"), (int, float))
        and 0 <= r["rx_cv"] <= 0.15
        and 0 <= idle.get("rx_kbps", -1) / raw <= 0.05
    )


def report_status(report):
    rx = report.get("config", {}).get("rx_sampler") is True
    expected_config = RX_CONFIG if rx else CONFIG
    order = RX_ORDER if rx else ORDER
    enabled = RX_ENABLED if rx else (False,) * len(ORDER)
    meta = devices(report)
    registration = report.get("registration", {})
    identity = (
        report.get("config") == expected_config
        and report.get("visible_devices") == "4"
        and set(meta) == {0}
        and meta[0]["physical_index"] == 4
        and meta[0]["pci_bus"].lower().endswith("98:00.0")
        and registration == dict(device=0, physical=4, flags=1, rc=0)
    )
    checked = report.get("correctness", [])
    correct = (
        len(checked) == 14
        and {(r["method"], r["bank"]) for r in checked}
        == {(m, b) for m in METHODS for b in range(7)}
        and all(r.get("all_bytes") is True and r.get("guards") is True for r in checked)
    )
    graphs = report.get("chunk_graphs", [])
    graph_ok = (
        len(graphs) == 2
        and {g["method"] for g in graphs} == set(METHODS)
        and all(
            g.get("correctness") is True
            and g["device"] == g["capture_device"] == 0
            and g["nbytes"] == COPY_BYTES
            and g["banks"] == 7
            and g["waves"] == 12
            and g["copy_nodes"] == 84
            and g["per_copy_timing_events"] == 0
            and g["launches_per_chunk"] == 1
            and g["num_blocks"] == (4 if g["method"] == "sm64" else None)
            and g["block_size"] == (1024 if g["method"] == "sm64" else None)
            for g in graphs
        )
    )
    binding = report.get("binding", {})
    bound = set(binding) == BINDINGS and all(v is True for v in binding.values())
    artifact = report.get("jit_artifact", {})
    jit = (
        artifact.get("variant") == "linear"
        and Path(artifact.get("path", "")).name == LINEAR_LIBRARY
        and artifact.get("sha256") == REFERENCE_SHA256
    )
    windows = report.get("windows", [])
    timing = [(w["trial"], w["method"]) for w in windows] == list(
        enumerate(order)
    ) and all(
        w.get("device") == 0
        and w.get("sampler_enabled") is sampled
        and w.get("bytes_per_graph") == COPY_BYTES * 7
        and w["metrics"].get("valid") is True
        and isinstance(w["metrics"].get("logical_wall_GBps"), (int, float))
        and math.isfinite(w["metrics"]["logical_wall_GBps"])
        and w["metrics"]["logical_wall_GBps"] > 0
        for w, sampled in zip(windows, enabled)
    )
    drift = {}
    if timing:
        for m in METHODS:
            pair = rates(report, m)
            drift[m] = pair[0] / pair[-1]
        timing = all(0.9 <= r <= 1.1 for r in drift.values())
    ce = [w for w in windows if w["method"] == "ce64"]
    reference = len(ce) == (4 if rx else 2) and all(
        w["metrics"].get("logical_wall_GBps", 0) >= 45 for w in ce
    )
    status = dict(
        identity=identity,
        correctness=correct,
        graphs=graph_ok,
        host=local_host(report.get("host_snapshot", {})),
        bindings=bound,
        jit=jit,
        timing=timing,
        bulk_reference=reference,
    )
    controls, calibrations = {}, []
    if rx:
        controls = rx_controls(report)
        calibrations = [
            (
                ce_calibration(w["metrics"])
                if rx_window_valid(w)
                else dict(valid=False, reason="Invalid Rx measurement window")
            )
            for w in ce
            if w.get("sampler_enabled") is True
        ]
        status["sampler_controls"] = all(c["valid"] for c in controls.values())
        sampled = [w for w in windows if w.get("sampler_enabled") is True]
        status["rx_samples"] = len(sampled) == 4 and all(
            rx_window_valid(w) for w in sampled
        )
        status["ce_calibration"] = len(calibrations) == 2 and all(
            c["valid"] for c in calibrations
        )
    return dict(
        **status,
        ready=report.get("success") is True and all(status.values()),
        drift=drift,
        **(dict(controls=controls, calibrations=calibrations) if rx else {}),
    )


def compact_report(report):
    s = report_status(report)
    rx = report["config"].get("rx_sampler") is True
    lines = [
        "Same 64MiB source/target SM vs CE; physical GPU4; unchanged linear CUDA kernel.",
        f'success={report["success"]} results_valid={s["ready"]} correctness={len(report.get("correctness", []))}/14 graphs={len(report.get("chunk_graphs", []))}/2 gates={s}',
        f'CONFIG {report["config"]}',
    ]
    if report.get("error") or report.get("cleanup_error"):
        lines.append(
            f'ERROR active={report.get("active_case")} error={report.get("error")} cleanup={report.get("cleanup_error")}'
        )
    for d, m in devices(report).items():
        lines.append(
            f'IDENTITY d{d} physical={m["physical_index"]} BDF={m["pci_bus"]} UUID={m["uuid"]}'
        )
    lines.append(
        f'REGISTRATION {report.get("registration")} BINDING {report.get("binding")}'
    )
    lines.append(
        f'HOST valid={s["host"]} regions={[r for b in report.get("host_snapshot", {}).get("buffers", []) for r in b["regions"]]}'
    )
    lines.append(f'JIT {report.get("jit_artifact")}')
    lines.append(
        f'TIMING valid={s["timing"]} windows={len(report.get("windows", []))}/{8 if rx else 4} drift={s["drift"]}'
    )
    for method in METHODS:
        rows = [w for w in report.get("windows", []) if w["method"] == method]
        if rows:
            quality = [
                {
                    k: w["metrics"].get(k)
                    for k in (
                        "valid",
                        "max_enqueue_fraction",
                        "chunk_gpu_cv",
                        "gpu_wall_ratio",
                        "reason",
                    )
                }
                for w in rows
            ]
            lines.append(
                f'RESULT {method} wall_GBps={[w["metrics"].get("logical_wall_GBps") for w in rows]} gpu_GBps={[w["metrics"].get("logical_gpu_GBps") for w in rows]} quality={quality}'
            )
    if rx:
        lines.append(f'SAMPLER_CONTROLS {s["controls"]}')
        lines.append(f'CE_CALIBRATION {s["calibrations"]}')
        for w in report.get("windows", []):
            if not w.get("sampler_enabled"):
                continue
            m = w["metrics"]
            samples = [r for r in w.get("samples", []) if not r.get("error")]
            call_ms = [r["call_ns"] / 1e6 for r in samples]
            period_ms = [
                (b["t_ns"] - a["t_ns"]) / 1e6 for a, b in zip(samples, samples[1:])
            ]
            logical, nominal = m.get("logical_wall_GBps"), m.get("rx_GBps_1024")
            per_payload = nominal / logical if nominal is not None and logical else None
            lines.append(
                f'RX {w["method"]} trial={w["trial"]} valid={m.get("valid")} '
                f'logical_GBps={logical} raw_rx_KBps={m.get("rates", {}).get("rx_kbps")} '
                f"nominal_rx_GBps={nominal} rx/logical={per_payload} "
                f'rates={m.get("rates")} idle/active={m.get("idle_over_active")} '
                f"query_ms_min_med_max={([min(call_ms), statistics.median(call_ms), max(call_ms)] if call_ms else [])} "
                f"period_ms_median={statistics.median(period_ms) if period_ms else None}"
            )
    states = [v for w in report.get("windows", []) for v in w.get("states", [])]
    lines.append(
        "STATE "
        + (clock_text([dict(before=v) for v in states]) if states else "not_collected")
    )
    if s["ready"]:
        a, b = rates(report, "sm64"), rates(report, "ce64")
        lines.append(
            f"RATIOS SM/CE_time={[round(y/x, 6) for x, y in zip(a,b)]} SM/CE_GBps={[round(x/y, 6) for x, y in zip(a,b)]}"
        )
        if rx:
            on = {
                method: [
                    w["metrics"]
                    for w in report["windows"]
                    if w["method"] == method and w["sampler_enabled"]
                ]
                for method in METHODS
            }
            logical = [
                a["logical_wall_GBps"] / b["logical_wall_GBps"]
                for a, b in zip(on["sm64"], on["ce64"])
            ]
            nominal = [
                a["rx_GBps_1024"] / b["rx_GBps_1024"]
                for a, b in zip(on["sm64"], on["ce64"])
            ]
            lines.append(
                f"RATIOS_RX SM/CE_logical={[round(v, 6) for v in logical]} "
                f"SM/CE_nominal_Rx={[round(v, 6) for v in nominal]} "
                f"relative_Rx_per_logical={[round(rx_rate/logical_rate, 6) for rx_rate, logical_rate in zip(nominal, logical)]}"
            )
    if rx:
        lines.append(
            "NVML raw KB/s uses nominal 1024 B/KB; no counter rescaling. Rx and logical rates use the same ON steady wall interval. Sampled aggregates do not establish TLP sizes/counts, pure payload traffic, or a hardware ceiling."
        )
    lines.append(
        "This tests the current continuous SM implementation at 64MiB; it does not establish a hardware ceiling or isolate the cause of payload-size effects."
    )
    return "\n".join(lines) + "\n"
