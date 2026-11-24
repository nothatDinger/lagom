#!/usr/bin/env python3
"""Attribute Nsight Systems CUDA activity per device and marked DSPARK step."""

import argparse
import csv
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path

KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEMSET = "CUPTI_ACTIVITY_KIND_MEMSET"
API = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
FUSED = {
    "load_cache_to_device_buffer_spec_gather_kernel": "swap_in_gather",
    "reread_resident_hits_kernel": "swap_in_reread",
    "load_cache_to_device_buffer_spec_commit_kernel": "swap_in_commit",
}
REPORTS = "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum"


def open_database(path):
    # sqlite3.connect(path) would silently create a missing export.
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def tables(db):
    return {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def require_kernels(db):
    if (
        KERNEL not in tables(db)
        or not db.execute(
            f"SELECT 1 FROM {KERNEL} WHERE end > start LIMIT 1"
        ).fetchone()
    ):
        raise ValueError("Nsight export contains no CUDA kernel activities")


def export_report(report, database, log, *, timeout=None):
    """Force a fresh export beside a private, writable copy of the report."""
    report, database = Path(report), Path(database)
    if not report.is_file() or not report.stat().st_size:
        raise ValueError(f"Missing or empty Nsight report: {report}")
    with tempfile.TemporaryDirectory(dir=database.parent) as scratch:
        copied = Path(scratch) / "capture.nsys-rep"
        shutil.copyfile(report, copied)
        with Path(log).open("w") as out:
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--force-export=true",
                    "--report",
                    REPORTS,
                    str(copied),
                ],
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        fresh = copied.with_suffix(".sqlite")
        if not fresh.is_file() or not fresh.stat().st_size:
            raise ValueError(f"Nsight stats produced no SQLite export; inspect {log}")
        with closing(open_database(fresh)) as db:
            require_kernels(db)
        fresh.replace(database)


def union(intervals):
    output = []
    for start, end in sorted(intervals):
        if output and start <= output[-1][1]:
            output[-1][1] = max(end, output[-1][1])
        else:
            output.append([start, end])
    return output


def nvtx_ranges(db, available, prefix):
    if "NVTX_EVENTS" not in available:
        raise ValueError("Nsight export has no NVTX_EVENTS; capture cuda,nvtx")
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    join, label = "", "n.text" if "text" in columns else "NULL"
    if "textId" in columns and "StringIds" in available:
        join = "LEFT JOIN StringIds s ON s.id = n.textId"
        label = f"coalesce(nullif({label}, ''), s.value)"
    return db.execute(
        f"SELECT n.globalTid, n.start, n.end, {label} FROM NVTX_EVENTS n {join} "
        f"WHERE n.end > n.start AND {label} LIKE ?",
        (prefix + "%",),
    )


def analyze(database):
    """Idle on every captured GPU during an annotated CPU dependency."""
    with closing(open_database(database)) as db:
        available = tables(db)
        require_kernels(db)
        ranges = list(nvtx_ranges(db, available, "lagom/cpu_dependency_commit/step="))
        busy_at = {}
        if ranges:
            ticks = sorted({t for _, a, b, _ in ranges for t in (a, b)})
            selects = [
                f"SELECT start,end FROM {t} WHERE end > start AND end > ? AND start < ?"
                for t in (KERNEL, MEMCPY, MEMSET)
                if t in available
            ]
            events = db.execute(
                " UNION ALL ".join(selects) + " ORDER BY start",
                (ticks[0], ticks[-1]) * len(selects),
            )
            event = next(events, None)
            occupied, through = 0, ticks[0]
            for tick in ticks:
                while event is not None and event[0] < tick:
                    a, b = event
                    occupied += max(0, b - max(a, through))
                    through = max(through, b)
                    event = next(events, None)
                busy_at[tick] = occupied - max(0, through - tick)
        rows = []
        for tid, start, end, label in ranges:
            occupied = busy_at[end] - busy_at[start]
            rows.append(
                {
                    "step": int(re.search(r"/step=(\d+)", label)[1]),
                    "global_tid": tid,
                    "dependency_wall_ms": (end - start) / 1e6,
                    "all_gpu_idle_in_dependency_ms": (end - start - occupied) / 1e6,
                }
            )
        return {
            "rows": rows,
            "definition": "Wall time in a marked CPU dependency with every captured GPU idle.",
        }


def kernel_family(name):
    for symbol, family in FUSED.items():
        if symbol in name:
            return family
    legacy = re.search(
        r"(?:\bsglang::staging::|^)(promote_kernel|copy_kernel)\s*<\s*(?:\(bool\)\s*)?(\w+)",
        name,
    )
    if legacy:
        if legacy[1] == "promote_kernel":
            return "promote"
        if legacy[2] in ("1", "true"):
            return "refill"
        if legacy[2] in ("0", "false"):
            return "stage_copy"
    if re.search(r"gemm|matmul|nvjet_sm\d+|marlin::Marlin<", name, re.IGNORECASE):
        return "gemm"
    return "other"


def kernel_totals(counts, names):
    """Share of summed, clipped kernel durations within a single device/window."""
    total = sum(ns for _, ns in counts.values())
    families = defaultdict(lambda: [0, 0])
    kernels = []
    for name_id, (calls, ns) in sorted(counts.items(), key=lambda item: -item[1][1]):
        name = names.get(name_id) or f"<unknown name ID {name_id}>"
        family = kernel_family(name)
        families[family][0] += calls
        families[family][1] += ns
        kernels.append(
            {"name": name, "family": family, "instances": calls, "kernel_ms": ns / 1e6}
        )
    groups = [
        {
            "family": family,
            "instances": calls,
            "kernel_ms": ns / 1e6,
            "pct_of_summed_kernel_time": 100 * ns / total,
        }
        for family, (calls, ns) in sorted(
            families.items(), key=lambda item: -item[1][1]
        )
    ]
    return {"summed_kernel_ms": total / 1e6, "families": groups, "kernels": kernels}


def capture_kernels(database):
    """Whole-capture attribution, including reports from before step stamps existed."""
    with closing(open_database(database)) as db:
        require_kernels(db)
        rows = db.execute(
            f"""SELECT k.globalPid, k.deviceId, k.demangledName, s.value, k.calls, k.ns
                FROM (SELECT globalPid, deviceId, demangledName, count(*) AS calls,
                             sum(end-start) AS ns FROM {KERNEL} WHERE end > start
                      GROUP BY globalPid, deviceId, demangledName) k
                LEFT JOIN StringIds s ON s.id = k.demangledName"""
        )
        totals, names = defaultdict(dict), {}
        for pid, dev, name_id, name, calls, ns in rows:
            names[name_id] = name or f"<unknown name ID {name_id}>"
            totals[pid, dev][name_id] = (calls, ns)
        return [
            dict(pid=process_id(pid), device=dev, **kernel_totals(counts, names))
            for (pid, dev), counts in sorted(totals.items())
        ]


def process_id(global_pid):
    # Nsight stores the process ID in bits 24..47 of globalPid/globalTid.
    return (global_pid >> 24) & 0xFFFFFF


def step_windows(db, available):
    markers = defaultdict(dict)
    for tid, start, end, label in nvtx_ranges(db, available, "lagom/step"):
        match = re.fullmatch(
            r"lagom/(step|step_begin|step_end)/step=(\d+)(?:/layer=None)?", label
        )
        if match:
            key = (tid, int(match[2]))
            if match[1] in markers[key]:
                raise ValueError(f"Duplicate {match[1]} marker for {key}")
            markers[key][match[1]] = (start, end)
    if not markers:
        raise ValueError("No marked DSPARK steps in Nsight export")
    # Index only the measured scheduler threads. A full kernel table can contain
    # millions of rows and the Nsight export need not have correlation indexes.
    tids = sorted({tid for tid, _ in markers})
    params = ",".join("?" for _ in tids)
    selects = [
        f"SELECT globalTid,start,end,correlationId FROM {t} WHERE globalTid IN ({params})"
        for t in API
        if t in available
    ]
    if not selects:
        raise ValueError(
            "Step breakdown requires CUDA runtime or driver API activities"
        )
    db.execute(
        "CREATE TEMP TABLE api AS " + " UNION ALL ".join(selects), tids * len(selects)
    )
    db.execute("CREATE INDEX temp.api_thread_start ON api(globalTid,start)")
    db.execute(
        """CREATE TEMP TABLE stamp(pid INTEGER, cid INTEGER, tid INTEGER, step INTEGER, edge TEXT,
                 PRIMARY KEY(pid,cid,tid,step,edge)) WITHOUT ROWID"""
    )
    for (tid, step), edges in markers.items():
        if edges.keys() != {"step", "step_begin", "step_end"}:
            raise ValueError(f"Incomplete NVTX markers for step {step} on thread {tid}")
        for edge in ("step_begin", "step_end"):
            start, end = edges[edge]
            db.execute(
                """INSERT OR IGNORE INTO stamp SELECT ?,correlationId,?,?,? FROM api
                   WHERE globalTid=? AND start BETWEEN ? AND ? AND end<=?
                     AND correlationId IS NOT NULL""",
                (tid & ~0xFFFFFF, tid, step, edge, tid, start, end, end),
            )
    stamps = defaultdict(dict)
    for tid, step, edge, start, end, dev in db.execute(
        f"""SELECT s.tid,s.step,s.edge,k.start,k.end,k.deviceId
            FROM {KERNEL} k CROSS JOIN stamp s
            WHERE s.pid=k.globalPid AND s.cid=k.correlationId"""
    ):
        pair = stamps[tid, step]
        if edge in pair:
            raise ValueError(
                f"Ambiguous GPU stamp for {edge}, step {step} on thread {tid}"
            )
        pair[edge] = (start, end, dev)
    windows = defaultdict(list)
    for (tid, step), edges in markers.items():
        pair = stamps[tid, step]
        if pair.keys() != {"step_begin", "step_end"}:
            raise ValueError(
                f"Missing correlated GPU stamps for step {step} on thread {tid}"
            )
        (_, start, device), (end, _, last_device) = pair["step_begin"], pair["step_end"]
        if device != last_device or start >= end:
            raise ValueError("Invalid GPU stamp order or device")
        windows[tid & ~0xFFFFFF, device].append((start, end, step, tid, *edges["step"]))
    return windows


def gpu_activities(db, available, pid, device, start, end):
    # One chronological cursor per rank, shared by all its steps. SQLite may
    # sort to disk. Python retains activities for only one step at a time.
    columns = {r[1] for r in db.execute(f"PRAGMA table_info({KERNEL})")}
    selects = []
    for table, kind, detail in (
        (KERNEL, "'kernel'", "demangledName" if "demangledName" in columns else "NULL"),
        (MEMCPY, "CASE WHEN copyKind=1 THEN 'h2d' ELSE 'other' END", "bytes"),
        (MEMSET, "'other'", "0"),
    ):
        if table in available:
            selects.append(
                f"SELECT start,end,{kind},{detail} FROM {table} "
                "WHERE globalPid=? AND deviceId=? AND start<? AND end>? AND end>start"
            )
    return iter(
        db.execute(
            " UNION ALL ".join(selects) + " ORDER BY start",
            (pid, device, end, start) * len(selects),
        )
    )


def partition(start, end, edges):
    active = dict.fromkeys(("kernel", "h2d", "other", "api"), 0)
    spans = defaultdict(int)
    prev = start
    for tick, kind, change in sorted(edges + [(end, "api", 0)]):
        live = [k for k in ("kernel", "h2d", "other") if active[k]]
        bucket = live[0] if live else "idle"
        if len(live) > 1:
            bucket = "overlap"
        duration = tick - prev
        spans[bucket] += duration
        if active["h2d"]:
            spans["inclusive_h2d"] += duration
        if not live and active["api"]:
            spans["idle_with_api"] += duration
        active[kind] += change
        prev = tick
    return spans


def rank_steps(db, available, pid, device, windows, names):
    windows = sorted(windows)
    events = gpu_activities(db, available, pid, device, windows[0][0], windows[-1][1])
    event = next(events, None)
    carry, rows, totals = [], [], defaultdict(lambda: [0, 0])
    last_end = None
    for start, end, step, tid, host_start, host_end in windows:
        if last_end is not None and start < last_end:
            raise ValueError(f"Overlapping GPU steps on device {device}")
        last_end = end
        # A kernel/copy can straddle stamps. Carry it forward and clip its time
        # to each intersection. Count a kernel instance once in the rank total.
        batch, carry = carry, []
        while event is not None and event[0] < end:
            if event[1] > start:
                batch.append((*event, True))
            event = next(events, None)
        counts = defaultdict(lambda: [0, 0])
        edges = []
        n_copies = n_bytes = crossing = 0
        for a, b, kind, detail, new in batch:
            if b <= start:
                continue
            lo, hi = max(start, a), min(end, b)
            edges.extend(((lo, kind, 1), (hi, kind, -1)))
            if b > end:
                carry.append((a, b, kind, detail, False))
            if kind == "kernel":
                counts[detail][0] += 1
                counts[detail][1] += hi - lo
                totals[detail][0] += new
                totals[detail][1] += hi - lo
            elif kind == "h2d":
                if a >= start and b <= end:
                    n_copies += 1
                    n_bytes += detail
                else:
                    crossing += 1
        if not counts:
            raise ValueError(f"Empty GPU kernel window on device {device}, step {step}")
        apis = union(
            db.execute(
                "SELECT start,end FROM api WHERE globalTid=? AND start BETWEEN ? AND ? AND end<=?",
                (tid, host_start, host_end, host_end),
            )
        )
        for a, b in apis:
            if a < end and b > start:
                edges.extend(((max(a, start), "api", 1), (min(b, end), "api", -1)))
        span = partition(start, end, edges)
        rows.append(
            {
                "step": step,
                "global_tid": tid,
                "device": device,
                "pid": process_id(pid),
                "gpu_step_ms": (end - start) / 1e6,
                "kernel_only_ms": span["kernel"] / 1e6,
                "explicit_h2d_only_ms": span["h2d"] / 1e6,
                "other_gpu_only_ms": span["other"] / 1e6,
                "gpu_overlap_ms": span["overlap"] / 1e6,
                "gpu_idle_ms": span["idle"] / 1e6,
                "cpu_api_during_gpu_idle_ms": span["idle_with_api"] / 1e6,
                "idle_without_observed_api_ms": (span["idle"] - span["idle_with_api"])
                / 1e6,
                "explicit_h2d_inclusive_ms": span["inclusive_h2d"] / 1e6,
                "explicit_h2d_bytes": n_bytes,
                "explicit_h2d_copies": n_copies,
                "h2d_copies_crossing_boundary": crossing,
                "cpu_cuda_api_union_ms": sum(b - a for a, b in apis) / 1e6,
                "kernel_families": kernel_totals(counts, names)["families"],
            }
        )
    return rows, kernel_totals(totals, names)


def step_report(database, *, ranks_by_pid=None, steps=None, required_families=()):
    with closing(open_database(database)) as db:
        require_kernels(db)
        available = tables(db)
        names = (
            dict(db.execute("SELECT id,value FROM StringIds"))
            if "StringIds" in available
            else {}
        )
        windows = step_windows(db, available)
        if ranks_by_pid is not None and (
            {process_id(pid) for pid, _ in windows} != set(ranks_by_pid)
            or len({dev for _, dev in windows}) != len(ranks_by_pid)
            or len(windows) != len(ranks_by_pid)
        ):
            raise ValueError(
                "Captured CUDA devices/processes do not match the decode rank traces"
            )
        rows, kernels = [], []
        for (pid, device), bounds in sorted(windows.items()):
            part, totals = rank_steps(db, available, pid, device, bounds, names)
            if steps is not None and [r["step"] for r in part] != list(range(steps)):
                raise ValueError(f"Incomplete Nsight step window on device {device}")
            identity = {"device": device, "pid": process_id(pid)}
            if ranks_by_pid is not None:
                identity["rank"] = ranks_by_pid[process_id(pid)]
            for row in part:
                missing = set(required_families) - {
                    f["family"] for f in row["kernel_families"]
                }
                if missing:
                    raise ValueError(
                        f"Missing kernel families {sorted(missing)} on device {device}, step {row['step']}"
                    )
                row.update(identity)
            rows.extend(part)
            kernels.append(dict(**identity, steps=len(part), **totals))
        return {
            "measurement": "nsys_kernel_attribution",
            "decode_steps": rows,
            "kernel_attribution": kernels,
        }


def step_breakdown(database):
    return step_report(database)["decode_steps"]


def compare_kernel_windows(baseline, forced):
    """Compare per-rank summed kernel durations in matched profiled windows."""
    result = []
    base = {row["rank"]: row for row in baseline}
    if {row["rank"] for row in forced} != base.keys():
        raise ValueError("Unmatched decode ranks in Nsight captures")
    for arm in forced:
        control = base[arm["rank"]]
        if (arm["device"], arm["steps"]) != (control["device"], control["steps"]):
            raise ValueError("Cannot compare different Nsight devices or step counts")
        a, b = (
            {f["family"]: f["kernel_ms"] for f in row["families"]}
            for row in (control, arm)
        )
        result.append(
            {
                "rank": arm["rank"],
                "device": arm["device"],
                "steps": arm["steps"],
                "families": [
                    {
                        "family": family,
                        "baseline_kernel_ms": a.get(family, 0),
                        "forced_kernel_ms": b.get(family, 0),
                        "delta_kernel_ms": b.get(family, 0) - a.get(family, 0),
                    }
                    for family in sorted(a.keys() | b.keys())
                ],
            }
        )
    return {
        "measurement": "nsys_kernel_attribution",
        "scope": "sampled_steps",
        "ranks": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture", type=Path, help="Nsight .nsys-rep or exported SQLite"
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--steps",
        action="store_true",
        help="Decompose marked DSPARK steps on each device",
    )
    mode.add_argument(
        "--kernels",
        action="store_true",
        help="Aggregate kernels per device over the entire capture",
    )
    args = parser.parse_args()
    database = args.capture
    if args.output.resolve() == args.capture.resolve():
        parser.error("Choose an output file separate from the input capture")
    if database.suffix == ".nsys-rep":
        database = args.output.with_suffix(".sqlite")
        if args.output == database:
            parser.error(
                "--output names the JSON report; the SQLite export uses the same stem"
            )
        export_report(args.capture, database, args.output.with_suffix(".stats.txt"))
    if args.kernels:
        report = {
            "measurement": "nsys_kernel_attribution",
            "scope": "entire_capture",
            "kernel_attribution": capture_kernels(database),
        }
    else:
        report = analyze(database)
        if args.steps:
            report.update(step_report(database))
            rows = report["decode_steps"]
            with args.output.with_suffix(".steps.csv").open("w", newline="") as handle:
                fields = [key for key in rows[0] if key != "kernel_families"]
                writer = csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
        with args.output.with_suffix(".csv").open("w", newline="") as handle:
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
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
