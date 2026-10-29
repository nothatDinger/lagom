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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.sqlite)
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
