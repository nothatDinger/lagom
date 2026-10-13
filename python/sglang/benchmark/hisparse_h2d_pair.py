"""Standard-library checks and short reports for the shared-host GPU control."""

import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def pci_path(bus, sys_root="/sys/bus/pci/devices"):
    """Read endpoint/ancestor topology and link attributes; never read/write config."""
    try:
        domain, suffix = bus.strip().split(":", 1)
        normalized = f"{int(domain, 16):04x}:{suffix.lower()}"
        if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized):
            raise ValueError("Invalid PCI BDF")
        endpoint = (Path(sys_root) / normalized).resolve(strict=True)
    except (OSError, ValueError) as exc:
        return dict(error=str(exc), chain=[])
    chain = []
    for path in [endpoint, *endpoint.parents]:
        if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", path.name):
            continue
        row = dict(bdf=path.name, errors={})
        for key in (
            "current_link_speed",
            "current_link_width",
            "max_link_speed",
            "max_link_width",
            "numa_node",
        ):
            try:
                row[key] = (path / key).read_text().strip()
            except OSError as exc:
                row["errors"][key] = str(exc)
        chain.append(row)
    return dict(chain=chain)


def abba_order(devices, epochs, reverse=False):
    if len(devices) != 2 or len(set(devices)) != 2 or epochs <= 0:
        raise ValueError("Require two distinct devices and positive epochs")
    a, b = reversed(devices) if reverse else devices
    return [a, b, b, a] * epochs


def activity_check(events, wave):
    """Validate all copy nodes in one synchronized CPU annotation range.

    No per-node event matching is guessed: every node in a wave has the same
    payload, device and operation. A missing node invalidates the activity rate.
    """
    named = [e for e in events if e.get("ph") == "X" and e.get("name") == wave["scope"]]
    # Kineto may also project record_function onto a GPU stream under the
    # same name, with cat=gpu_user_annotation. Only the CPU range brackets
    # the entire synchronized measurement. Never select a range by duration
    # or arbitrarily discard a second CPU annotation.
    scopes = [e for e in named if e.get("cat") == "user_annotation"]
    evidence = dict(
        named_events=len(named),
        cpu_scopes=len(scopes),
        categories=dict(Counter(e.get("cat", "<missing>") for e in named)),
        examples=[
            {
                **{k: e[k] for k in ("cat", "pid", "tid", "ts", "dur") if k in e},
                **{
                    k: e["args"][k]
                    for k in ("device", "stream", "External id", "correlation")
                    if k in e.get("args", {})
                },
            }
            for e in named[:4]
        ],
    )

    def invalid(reason, **details):
        return dict(valid=False, reason=reason, scope_evidence=evidence, **details)

    if len(scopes) != 1:
        return invalid(f"Expected one CPU user_annotation scope, found {len(scopes)}")
    scope = scopes[0]
    if (
        any(
            not isinstance(scope.get(key), (int, float))
            or not math.isfinite(scope[key])
            for key in ("ts", "dur")
        )
        or scope["dur"] <= 0
    ):
        return invalid("Invalid CPU annotation timestamp or duration")
    lo, hi = scope["ts"] - 1, scope["ts"] + scope["dur"] + 1
    candidates = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("args", {}).get("device") == wave["device"]
        and e["ts"] >= lo
        and e["ts"] + e.get("dur", 0) <= hi
    ]
    if wave["method"] == "sm":
        copies = [
            e
            for e in candidates
            if "kernel" in e.get("cat", "").lower()
            and "copy_cache_planned_kernel" in e["name"]
        ]
    else:
        copies = [
            e
            for e in candidates
            if "gpu_memcpy" in e.get("cat", "").lower() and "htod" in e["name"].lower()
        ]
    if len(copies) != wave["banks"]:
        return invalid(
            f'Expected {wave["banks"]} copy nodes, found {len(copies)}',
            activity_names=sorted(
                {
                    e.get("name", "")
                    for e in candidates
                    if "gpu" in e.get("cat", "") or "kernel" in e.get("cat", "")
                }
            )[:8],
            gpu_devices_seen=sorted(
                {
                    e["args"]["device"]
                    for e in events
                    if e.get("ph") == "X"
                    and "device" in e.get("args", {})
                    and e["ts"] >= lo
                    and e["ts"] + e.get("dur", 0) <= hi
                },
                key=str,
            ),
        )
    for copy in copies:
        if not math.isfinite(copy["dur"]) or copy["dur"] <= 0:
            return invalid("Invalid GPU activity duration")
        recorded_bytes = copy.get("args", {}).get("bytes")
        if (
            wave["method"] == "ce"
            and recorded_bytes is not None
            and int(recorded_bytes) != wave["payload_bytes"]
        ):
            return invalid("CE activity byte count disagrees with payload")
    active = sum(e["dur"] for e in copies)
    interval = sum(wave["copy_us"])
    if interval <= 0 or not math.isfinite(interval):
        return invalid("Invalid graph event interval")
    if active > interval + max(1, 0.5 * len(copies)):
        return invalid(
            "Activity time exceeds bracketing event time beyond timer resolution",
            activity_gpu_us=active,
            event_gpu_us=interval,
        )
    return dict(
        valid=True,
        scope_evidence=evidence,
        activities=len(copies),
        activity_gpu_us=active,
        activity_GBps=wave["payload_bytes"] * len(copies) / active / 1000,
        event_over_activity=interval / active,
        event_gpu_us=interval,
        event_GBps=wave["payload_bytes"] * len(copies) / interval / 1000,
    )


def expected_case_keys(report):
    """The acquisition captures each device/case/method exactly once."""
    config = report["config"]
    cases = {
        (f'B{config["batch"]}-miss{misses}', method, device)
        for misses in config["misses"]
        for method in ("sm", "ce")
        for device in config["devices"]
    }
    cases.update(
        (f'bulk-ce-{config["bulk_ce_mib"]}MiB', "ce", device)
        for device in config["devices"]
    )
    return cases


def correctness_summary(report):
    """Read graph/copy assertions independently of profiler parsing status."""
    rows = report.get("correctness", [])
    keys = [(r["case"], r["method"], r["device"]) for r in rows]
    expected = expected_case_keys(report)
    payload = sum(r.get("all_payload_bytes_checked") is True for r in rows)
    replay = sum(r.get("graph_replay_checked") is True for r in rows)
    coverage = len(keys) == len(expected) and set(keys) == expected
    return dict(
        expected=len(expected),
        records=len(rows),
        payload_checked=payload,
        graph_replay_checked=replay,
        complete=bool(expected) and coverage and payload == replay == len(expected),
    )


def validate_profile(report, events):
    """Refresh only analysis fields; retain acquisition status and raw timings."""
    waves = report.get("profile_waves", [])
    for wave in waves:
        wave["activity_check"] = activity_check(events, wave)
    expected = expected_case_keys(report)
    keys = [(w["case"], w["method"], w["device"]) for w in waves]
    errors = []
    if not expected or len(keys) != len(expected) or set(keys) != expected:
        errors.append(
            "Profile waves do not cover each configured device/case/method exactly once"
        )
    if any(
        w["scope"] != f'h2d_pair:{w["case"]}:{w["method"]}:device{w["device"]}'
        for w in waves
    ):
        errors.append("Scope name disagrees with wave identity")
    validated = sum(w["activity_check"]["valid"] for w in waves)
    report["profile_success"] = not errors and validated == len(expected)
    report["profile_validation"] = dict(
        expected=len(expected), recorded=len(waves), validated=validated, errors=errors
    )
    return report["profile_success"]


def clock_text(waves):
    samples = [s for w in waves for s in (w["before"], w["after"])]

    def values(key):
        found = sorted({s[key] for s in samples if key in s})
        if len(found) > 3 and all(isinstance(v, (int, float)) for v in found):
            return f"{found[0]}..{found[-1]}"
        return "/".join(str(v) for v in found) if found else "unknown"

    return (
        f'SM_MHz={values("sm_MHz")} MEM_MHz={values("mem_MHz")} Pstate={values("pstate")} '
        f'PCIe_gen={values("pcie_gen")} width={values("pcie_width")} throttle={values("throttle_bits")} '
        f'power_W={values("power_W")} limit_W={values("power_limit_W")} temp_C={values("temperature_C")} '
        f'errors={sum(len(s.get("errors", {})) for s in samples)}'
    )


def compact_report(report):
    lines = [
        "Shared-host H2D control: GPU event intervals and GPU activities are different metrics.",
        f'success={report["success"]} profile_success={report.get("profile_success", "not_requested")} '
        f'host_register_flags={report["host_register_flags"]} (1=Portable; previous solo used 0)',
        f'config={report["config"]}',
    ]
    if report.get("error"):
        lines.append("ERROR " + report["error"])
    if "correctness" in report:
        check = correctness_summary(report)
        lines.append(
            f'CORRECTNESS records={check["records"]}/{check["expected"]} '
            f'payload_checked={check["payload_checked"]}/{check["expected"]} '
            f'graph_replay_checked={check["graph_replay_checked"]}/{check["expected"]} '
            f'complete={check["complete"]} (independent of profile_success)'
        )
    if "profile_validation" in report:
        lines.append(f'PROFILE_COVERAGE {report["profile_validation"]}')
    if "reanalysis" in report:
        lines.append(f'REANALYSIS {report["reanalysis"]}')
    for device, meta in sorted(report.get("devices", {}).items()):
        lines.append(
            f'logical={device} identity={ {k:v for k,v in meta.items() if k != "pci_path"} }'
        )
        topology = meta.get("pci_path")
        if topology:
            lines.append(
                "  PCI_endpoint_to_root="
                + " -> ".join(
                    f'{p["bdf"]} {p.get("current_link_speed", "?")} x{p.get("current_link_width", "?")} node={p.get("numa_node", "?")}'
                    for p in topology["chain"]
                )
                + f' errors={sum(len(p["errors"]) for p in topology["chain"])} {topology.get("error", "")}'
            )
    snapshot = report.get("host_snapshot", {})
    for buf in snapshot.get("buffers", []):
        placement = [
            (r.get("numa"), r["buffer_covers_whole_vma"]) for r in buf["regions"]
        ]
        lines.append(
            f'host={buf["name"]} address={buf["address"]} nbytes={buf["nbytes"]} VMA={placement}'
        )
    lines.append(
        "Rates use total payload / total time, decimal GB/s. SM uses CTA4/1024 threads."
    )
    groups = defaultdict(lambda: defaultdict(list))
    for wave in report.get("waves", []):
        groups[(wave["phase"], wave["case"], wave["method"])][wave["device"]].append(
            wave
        )
    for (phase, case, method), devices in sorted(groups.items()):
        lines.append(f"{phase} {case} {method}")
        rates = []
        for device, waves in sorted(devices.items()):
            times = [t for w in waves for t in w["copy_us"]]
            if not times or any(t <= 0 or not math.isfinite(t) for t in times):
                raise ValueError("Invalid paired event timing")
            byte_count = sum(w["payload_bytes"] * len(w["copy_us"]) for w in waves)
            rate = byte_count / sum(times) / 1000
            rates.append(rate)
            ordered = sorted(times)
            lines.append(
                f"  d{device} GB/s={rate:.2f} mean/median/p95_us={statistics.fmean(times):.2f}/"
                f"{statistics.median(times):.2f}/{ordered[math.ceil(.95*len(ordered))-1]:.2f} "
                f'wave_means_us={[round(statistics.fmean(w["copy_us"]), 2) for w in waves]} '
                + clock_text(waves)
            )
        if len(rates) == 2:
            lines.append(f"  first/second rate={rates[0]/rates[1]:.3f}")
    for wave in report.get("profile_waves", []):
        check = wave.get(
            "activity_check",
            dict(valid=False, reason="Profiling stopped before activity validation"),
        )
        evidence = check.get("scope_evidence")
        if evidence:
            lines.append(
                f'SCOPE {wave["scope"]} named={evidence["named_events"]} '
                f'cpu={evidence["cpu_scopes"]} categories={evidence["categories"]} '
                f'examples={evidence["examples"]}'
            )
        if check["valid"]:
            lines.append(
                f'PROFILE {wave["case"]} {wave["method"]} d{wave["device"]} n={check["activities"]} '
                f'event/activity_GBps={check["event_GBps"]:.2f}/{check["activity_GBps"]:.2f} '
                f'event_time/activity_time={check["event_over_activity"]:.3f}'
            )
        else:
            lines.append(f'PROFILE {wave["scope"]} UNASSIGNED {check}')
    lines.append(
        "NVML snapshots are before/after waves; they do not prove constant clocks throughout."
    )
    lines.append(
        "Compute warming also changes caches. Profiled results are validation, not the unprofiled baseline."
    )
    return "\n".join(lines) + "\n"
