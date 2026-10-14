"""Read an existing NCU smoke export and pair.json; no CUDA or subprocess calls."""

import argparse
import csv
import importlib.util
import io
import json
import math
import re
from pathlib import Path

REQUIRED = (
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "launch__grid_size",
    "launch__block_size",
)
# Availability only. These names are not assumed to exist on every NCU/GPU.
CANDIDATES = (
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum",
    "lts__t_sectors_aperture_sysmem_op_read.sum",
    "dram__bytes_read.sum",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
)


def read_target(text):
    header, rows = None, []
    columns = {"ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    for fields in csv.reader(io.StringIO(text)):
        if columns.issubset(fields):
            header = fields
        elif header is not None and len(fields) == len(header):
            rows.append(dict(zip(header, fields)))
    if not rows:
        raise ValueError("No NCU raw metric rows with the required CSV columns")
    if any(not row["ID"] for row in rows):
        raise ValueError("NCU metric row has no launch ID")
    identities = {
        tuple(
            row.get(key, "")
            for key in ("Host Name", "Process ID", "ID", "Context", "Device")
        )
        for row in rows
    }
    names = {row["Kernel Name"] for row in rows}
    if (
        len(identities) != 1
        or len(names) != 1
        or "copy_cache_planned_kernel" not in next(iter(names))
    ):
        raise ValueError("Expected exactly one copy_cache_planned_kernel in NCU export")
    metrics = {}
    for metric in REQUIRED:
        values = {
            (row["Metric Unit"], row["Metric Value"])
            for row in rows
            if row["Metric Name"] == metric
        }
        if len(values) != 1:
            raise ValueError(f"Missing or conflicting NCU metric: {metric}")
        unit, raw = values.pop()
        value = float(raw.replace(",", ""))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid NCU metric: {metric}={raw}")
        metrics[metric] = dict(value=value, unit=unit)
    if (
        metrics["launch__grid_size"]["value"] != 4
        or metrics["launch__block_size"]["value"] != 1024
    ):
        raise ValueError("Expected the unchanged CTA4/1024 gather launch")
    return dict(
        identity=next(iter(identities)), kernel=next(iter(names)), metrics=metrics
    )


def metric_availability(text):
    names = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_.]+", text))
    return dict(
        available=[name for name in CANDIDATES if name in names],
        missing=[name for name in CANDIDATES if name not in names],
    )


def pair_gate(report):
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/benchmark/hisparse_h2d_pair.py"
    )
    spec = importlib.util.spec_from_file_location("h2d_pair_checks", path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    correct = checks.correctness_summary(report)
    if not (
        report.get("success") is True
        and report.get("host_register_flags") == 1
        and not report["config"].get("profile")
        and not report.get("profile_waves")
        and correct["complete"]
        and correct["expected"] == 8
        and report["config"].get("device_source_control") is True
    ):
        raise ValueError(
            f"Incomplete paired smoke or concurrent torch profiler: {correct}"
        )
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--ncu-csv", type=Path, required=True)
    parser.add_argument("--metrics-list", type=Path)
    args = parser.parse_args()
    try:
        pair = json.loads(args.pair_json.read_text())
        correct = pair_gate(pair)
        target = read_target(args.ncu_csv.read_text())
        devices = {
            k: {f: v.get(f) for f in ("physical_index", "uuid", "pci_bus")}
            for k, v in pair["devices"].items()
        }
        availability = (
            metric_availability(args.metrics_list.read_text())
            if args.metrics_list
            else {}
        )
        print(
            "NCU_SMOKE compatibility_ok=True (one eager kernel; no performance conclusion)"
        )
        print("PAIR " + json.dumps(correct))
        print("DEVICES " + json.dumps(devices))
        print("TARGET " + json.dumps({k: target[k] for k in ("identity", "kernel")}))
        for name, value in target["metrics"].items():
            print(f"METRIC {name} {value['value']} {value['unit']}")
        for state, names in availability.items():
            print(f"METRICS_{state.upper()} " + json.dumps(names))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"NCU_SMOKE compatibility_ok=False ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
