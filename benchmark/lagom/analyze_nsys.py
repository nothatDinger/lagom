#!/usr/bin/env python3
"""Conservative GPU-idle accounting inside explicit CPU dependency ranges.

Uses Nsight's correlated clock; the result is NOT all CPU-caused GPU waiting.
Taking busy intervals across all devices provides a lower bound in TP runs.
"""

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path


def union(intervals):
    output = []
    for start, end in sorted(intervals):
        if output and start <= output[-1][1]:
            output[-1][1] = max(end, output[-1][1])
        else:
            output.append([start, end])
    return output


def analyze(database):
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "NVTX_EVENTS" not in tables:
            raise ValueError("Nsight export has no NVTX_EVENTS; capture cuda,nvtx")
        names = (
            dict(db.execute("SELECT id,value FROM StringIds"))
            if "StringIds" in tables
            else {}
        )
        busy = []
        activity_tables = tables & {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        }
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in activity_tables:
            raise ValueError("Nsight export has no CUDA kernel activities")
        for table in sorted(activity_tables):
            busy.extend(
                (r[0], r[1])
                for r in db.execute(
                    f'SELECT start,end FROM "{table}" WHERE end >= start'
                )
            )
        busy = union(busy)
        rows = []
        for entry in db.execute("SELECT * FROM NVTX_EVENTS WHERE end >= start"):
            entry = dict(entry)
            text = entry.get("text") or names.get(entry.get("textId"), "")
            match = re.search(r"lagom/cpu_dependency_commit/step=(\d+)", text)
            if not match:
                continue
            start, end = entry["start"], entry["end"]
            occupied = sum(
                max(0, min(end, b) - max(start, a))
                for a, b in busy
                if a < end and b > start
            )
            rows.append(
                {
                    "step": int(match[1]),
                    "global_tid": entry.get("globalTid"),
                    "dependency_wall_ms": (end - start) / 1e6,
                    "all_gpu_idle_in_dependency_ms": max(0, end - start - occupied)
                    / 1e6,
                }
            )
        return {
            "rows": rows,
            "definition": "All captured GPUs idle during an annotated CPU data-dependency range. Lower bound, not total CPU-caused GPU wait; do not add across TP ranks.",
        }


def step_breakdown(database):
    """Correlate the two step stamps with CUPTI and partition each rank's GPU timeline."""
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        names = (
            dict(db.execute("SELECT id,value FROM StringIds"))
            if "StringIds" in tables
            else {}
        )
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
        api_tables = tables & {
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_DRIVER",
        }
        if not {kernel_table, "NVTX_EVENTS"} <= tables or not api_tables:
            raise ValueError(
                "Step breakdown requires CUDA kernel/API activities and NVTX ranges"
            )
        ranges = {}
        for r in db.execute("SELECT * FROM NVTX_EVENTS WHERE end > start"):
            r = dict(r)
            label = r.get("text") or names.get(r.get("textId"), "")
            match = re.fullmatch(
                r"lagom/(step|step_begin|step_end)/step=(\d+)(?:/layer=None)?", label
            )
            if match:
                entry = ranges.setdefault((r["globalTid"], int(match[2])), {})
                entry[match[1]] = (r["start"], r["end"])
        if not ranges:
            raise ValueError("No marked DSPARK steps in Nsight export")
        results = []
        for (tid, step), markers in sorted(ranges.items()):
            if not {"step", "step_begin", "step_end"} <= markers.keys():
                raise ValueError(
                    f"Incomplete NVTX markers for step {step} on thread {tid}"
                )
            pid = (
                tid & ~0xFFFFFF
            )  # Nsight globalTid packs the thread ID in the low 24 bits.
            host_start, host_end = markers["step"]
            apis = [
                r
                for table in api_tables
                for r in db.execute(
                    f"SELECT start,end,correlationId FROM {table} WHERE globalTid=? AND start>=? AND end<=?",
                    (tid, host_start, host_end),
                )
            ]

            def gpu_marker(name):
                start, end = markers[name]
                ids = {
                    r["correlationId"]
                    for r in apis
                    if start <= r["start"] and r["end"] <= end
                }
                matches = [
                    r
                    for cid in ids
                    for r in db.execute(
                        f"SELECT start,end,deviceId FROM {kernel_table} WHERE globalPid=? AND correlationId=?",
                        (pid, cid),
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one CUPTI stamp for {name}, step {step}, got {len(matches)}"
                    )
                return matches[0]

            first, last = gpu_marker("step_begin"), gpu_marker("step_end")
            start, end, device = first["end"], last["start"], first["deviceId"]
            if device != last["deviceId"] or start >= end:
                raise ValueError("Invalid GPU stamp order or device")
            edges = [(start, "idle", 0), (end, "idle", 0)]
            memcpy_bytes, memcpy_count, boundary_copies = 0, 0, 0
            for table, default_kind in (
                (kernel_table, "kernel"),
                ("CUPTI_ACTIVITY_KIND_MEMCPY", "other"),
                ("CUPTI_ACTIVITY_KIND_MEMSET", "other"),
            ):
                if table not in tables:
                    continue
                memcpy = table == "CUPTI_ACTIVITY_KIND_MEMCPY"
                columns = "start,end,copyKind,bytes" if memcpy else "start,end"
                for r in db.execute(
                    f"SELECT {columns} FROM {table} WHERE globalPid=? AND deviceId=? AND start<? AND end>?",
                    (pid, device, end, start),
                ):
                    kind = default_kind
                    if memcpy and r["copyKind"] == 1:  # CUPTI_ACTIVITY_MEMCPY_KIND_HTOD
                        kind = "h2d"
                        if start <= r["start"] and r["end"] <= end:
                            memcpy_count += 1
                            memcpy_bytes += r["bytes"]
                        else:
                            boundary_copies += 1
                    edges.extend(
                        (
                            (max(r["start"], start), kind, 1),
                            (min(r["end"], end), kind, -1),
                        )
                    )
            active = dict.fromkeys(("kernel", "h2d", "other", "idle"), 0)
            spans = dict.fromkeys(("kernel", "h2d", "other", "overlap", "idle"), 0)
            inclusive_h2d = 0
            prev = start
            for tick, kind, change in sorted(edges):
                live = [name for name, n in active.items() if n > 0]
                bucket = "overlap" if len(live) > 1 else next(iter(live), "idle")
                spans[bucket] += tick - prev
                if active["h2d"]:
                    inclusive_h2d += tick - prev
                active[kind] += change
                prev = tick
            api_busy = sum(
                b - a for a, b in union((r["start"], r["end"]) for r in apis)
            )
            results.append(
                {
                    "step": step,
                    "global_tid": tid,
                    "device": device,
                    "gpu_step_ms": (end - start) / 1e6,
                    "kernel_only_ms": spans["kernel"] / 1e6,
                    "explicit_h2d_only_ms": spans["h2d"] / 1e6,
                    "other_gpu_only_ms": spans["other"] / 1e6,
                    "gpu_overlap_ms": spans["overlap"] / 1e6,
                    "gpu_idle_ms": spans["idle"] / 1e6,
                    "explicit_h2d_inclusive_ms": inclusive_h2d / 1e6,
                    "explicit_h2d_bytes": memcpy_bytes,
                    "explicit_h2d_copies": memcpy_count,
                    "h2d_copies_crossing_boundary": boundary_copies,
                    "cpu_cuda_api_union_ms": api_busy / 1e6,
                }
            )
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Decompose marked DSPARK steps on each rank",
    )
    args = parser.parse_args()
    report = analyze(args.sqlite)
    if args.steps:
        rows = step_breakdown(args.sqlite)
        report["decode_steps"] = rows
        with args.output.with_suffix(".steps.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    args.output.write_text(json.dumps(report, indent=2))
    with args.output.with_suffix(".csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "global_tid",
                "dependency_wall_ms",
                "all_gpu_idle_in_dependency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(report["rows"])


if __name__ == "__main__":
    main()
