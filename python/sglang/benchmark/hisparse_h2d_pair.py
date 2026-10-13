"""Standard-library checks and short reports for the shared-host GPU control."""

import math
import re
import statistics
from collections import defaultdict
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
    scopes = [
        e for e in events if e.get("ph") == "X" and e.get("name") == wave["scope"]
    ]
    if len(scopes) != 1:
        return dict(valid=False, reason=f"Expected one scope, found {len(scopes)}")
    scope = scopes[0]
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
        return dict(
            valid=False,
            reason=f'Expected {wave["banks"]} copy nodes, found {len(copies)}',
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
            return dict(valid=False, reason="Invalid GPU activity duration")
        recorded_bytes = copy.get("args", {}).get("bytes")
        if (
            wave["method"] == "ce"
            and recorded_bytes is not None
            and int(recorded_bytes) != wave["payload_bytes"]
        ):
            return dict(
                valid=False, reason="CE activity byte count disagrees with payload"
            )
    active = sum(e["dur"] for e in copies)
    interval = sum(wave["copy_us"])
    if interval <= 0 or not math.isfinite(interval):
        return dict(valid=False, reason="Invalid graph event interval")
    if active > interval + max(1, 0.5 * len(copies)):
        return dict(
            valid=False,
            reason="Activity time exceeds bracketing event time beyond timer resolution",
            activity_gpu_us=active,
            event_gpu_us=interval,
        )
    return dict(
        valid=True,
        activities=len(copies),
        activity_gpu_us=active,
        activity_GBps=wave["payload_bytes"] * len(copies) / active / 1000,
        event_over_activity=interval / active,
        event_gpu_us=interval,
        event_GBps=wave["payload_bytes"] * len(copies) / interval / 1000,
    )


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
