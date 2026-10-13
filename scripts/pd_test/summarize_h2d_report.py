"""Produce short text from local H2D artifacts; original files stay on the machine.

Uses only Python's standard library. Analysis inputs are the saved --output-file
JSONs from hisparse_h2d.py, not its shortened stdout or raw profiler traces.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path


def stats(rows):
    nbytes = sum(s["h2d_bytes"] for s in rows)
    duration = sum(s["copy_us"] for s in rows)
    return dict(n=len(rows), nbytes=nbytes, us=duration, GBps=nbytes / duration / 1000)


def cycles(rows):
    """Require complete, repeated layer sequences; never infer from num_steps."""
    if not rows or any(b["index"] != a["index"] + 1 for a, b in zip(rows, rows[1:])):
        return None
    layers = {s["layer"] for s in rows}
    size = len(layers)
    template = [s["layer"] for s in rows[:size]]
    if len(set(template)) != size or len(rows) % size:
        return None
    grouped = [rows[i : i + size] for i in range(0, len(rows), size)]
    if any(
        [s["layer"] for s in group] != template
        or len({(s["real_batch"], s["verify_width"]) for s in group}) != 1
        for group in grouped
    ):
        return None
    if len(grouped) < 2:
        return None
    return dict(
        layers=size,
        iterations=len(grouped),
        copy_ms=statistics.fmean(
            sum(s["copy_us"] for s in group) / 1000 for group in grouped
        ),
    )


def validate_analysis(report):
    rows = report["samples"]
    summary = report["summary"]
    if (
        not rows
        or len(rows) != summary["samples"]
        or sum(s["h2d_bytes"] for s in rows) != summary["h2d_bytes"]
    ):
        raise ValueError("Sample count or byte sum does not match the saved summary")
    if not math.isfinite(summary["copy_gpu_us"]) or summary["copy_gpu_us"] <= 0:
        raise ValueError("Analysis needs positive, finite GPU timing")
    if report.get("pairing", {}).get("mode") != "serialized_recorders":
        return False
    if any(
        not math.isfinite(s["copy_us"]) or s["copy_us"] <= 0 or s["h2d_bytes"] < 0
        for s in rows
    ):
        raise ValueError("Invalid per-sample copy time or bytes")
    if not math.isclose(
        sum(s["copy_us"] for s in rows),
        summary["copy_gpu_us"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    ):
        raise ValueError("Sample GPU time sum does not match the saved summary")
    return True


def bucket(byte_count):
    if byte_count == 0:
        return "zero"
    for hi, label in [
        (250000, "(0,.25)"),
        (1000000, "[.25,1)"),
        (4000000, "[1,4)"),
        (16000000, "[4,16)"),
    ]:
        if byte_count < hi:
            return label
    return "[16,inf)"


def analysis_text(directory, max_layers):
    artifacts = []
    for path in sorted(directory.rglob("*.json")):
        data = path.read_bytes()
        report = json.loads(data)
        if "summary" not in report or "samples" not in report:
            continue
        timed = validate_analysis(report)
        artifacts.append(
            (report, path.name, hashlib.sha256(data).hexdigest()[:12], timed)
        )
    if not artifacts:
        raise ValueError(f"No saved analysis JSONs in {directory}")
    artifacts.sort(key=lambda a: (a[0]["device"], a[0]["pid"]))
    if len({a[0]["device"] for a in artifacts}) != len(artifacts):
        raise ValueError(
            "One directory must contain one capture per CUDA device; separate repeated captures"
        )
    lines = [
        f"\nANALYSIS {directory.name}",
        "CUDA device IDs below are logical; physical mapping is reported separately.",
        "device pid samples bytes copy_ms GB/s rec% overlap_us layers/cycles mean_copy_ms/cycle file:sha256",
    ]
    groups, totals = [], []
    for report, name, digest, timed in artifacts:
        summary, rows = report["summary"], report["samples"]
        cycle = cycles(rows) if timed else None
        cycle_text = (
            f'{cycle["layers"]}/{cycle["iterations"]} {cycle["copy_ms"]:.3f}'
            if cycle
            else "unverified NA"
        )
        lines.append(
            f'{report["device"]} {report["pid"]} {len(rows)} {summary["h2d_bytes"]} '
            f'{summary["copy_gpu_us"]/1000:.3f} {summary["h2d_bytes"]/summary["copy_gpu_us"]/1000:.2f} '
            f'{summary["recorder_over_copy_pct"]:.2f} {summary["copy_gpu_overlap_us"]:.5f} '
            f"{cycle_text} {name}:{digest}"
        )
        geometry = defaultdict(list)
        layer_groups = defaultdict(list)
        for row in rows:
            geometry[
                (row["real_batch"], row["captured_batch"], row["verify_width"])
            ].append(row)
            if timed:
                layer_groups[row["layer"]].append(row)
        lines.append(
            "  pairing="
            + report["pairing"]["mode"]
            + " streams="
            + str(report["pairing"]["recorder_streams"])
            + " B/captured/W: "
            + "; ".join(
                f"{b}/{c}/{w} n={len(rs)}"
                + (f' GB/s={stats(rs)["GBps"]:.2f}' if timed else "")
                for (b, c, w), rs in sorted(geometry.items())
            )
        )
        groups.append(layer_groups)
        totals.append(summary["copy_gpu_us"])
    if not all(a[3] for a in artifacts):
        lines.append(
            "Per-layer/payload-bin comparison omitted: at least one capture has unassigned sample timing."
        )
        return lines
    devices = [a[0]["device"] for a in artifacts]
    all_layers = sorted(set().union(*(g.keys() for g in groups)))
    # Show layers with high cost and the smallest positive payload. Reference
    # device is the first displayed device; layer selection does not sum ranks.
    ranked = sorted(
        all_layers,
        key=lambda layer: max(
            sum(s["copy_us"] for s in g.get(layer, [])) / total
            for g, total in zip(groups, totals)
        ),
        reverse=True,
    )
    selected = ranked[: max(1, max_layers - 2)]
    small = sorted(
        (
            layer
            for layer in all_layers
            if groups[0].get(layer)
            and sum(s["h2d_bytes"] for s in groups[0][layer]) > 0
        ),
        key=lambda layer: statistics.fmean(s["h2d_bytes"] for s in groups[0][layer]),
    )
    selected = sorted(set((selected + small[:2])[:max_layers]))
    if max_layers >= len(all_layers):
        selected = all_layers
    lines.append(
        "Layers: reference mean_MB/copy; each device cell=GB/s,time% (per-rank totals). selected="
        + str(selected)
    )
    lines.append("layer mean_MB " + " ".join(f"d{d}:GB/s,time%" for d in devices))
    for layer in selected:
        cells = [stats(g[layer]) if layer in g else None for g in groups]
        reference = cells[0]
        mean_mb = f'{reference["nbytes"]/reference["n"]/1e6:.3f}' if reference else "NA"
        lines.append(
            f"{layer} {mean_mb} "
            + " ".join(
                f'{c["GBps"]:.2f},{100*c["us"]/t:.1f}' if c else "NA"
                for c, t in zip(cells, totals)
            )
        )
    for i in range(1, len(groups)):
        ratios = []
        for layer in all_layers:
            if not groups[0].get(layer) or not groups[i].get(layer):
                continue
            a, b = stats(groups[0][layer]), stats(groups[i][layer])

            # Compare only the same geometry mix and per-layer mean payloads
            # within 5%. This is aggregate descriptive evidence, not a matched
            # iteration join: captures do not contain shared iteration IDs.
            def geometry(rs):
                return Counter(
                    (s["real_batch"], s["captured_batch"], s["verify_width"])
                    for s in rs
                )

            if (
                a["nbytes"]
                and b["nbytes"]
                and abs((b["nbytes"] / b["n"]) / (a["nbytes"] / a["n"]) - 1) <= 0.05
                and geometry(groups[0][layer]) == geometry(groups[i][layer])
            ):
                ratios.append((b["us"] / b["nbytes"] / (a["us"] / a["nbytes"]), layer))
        if ratios:
            worst = max(ratios)
            lines.append(
                f"d{devices[i]}/d{devices[0]} normalized_us/byte across comparable layers: n={len(ratios)} "
                f"min/median/max={min(ratios)[0]:.3f}/{statistics.median(r[0] for r in ratios):.3f}/{worst[0]:.3f} max_layer={worst[1]}"
            )
    lines.append(
        "Payload bins (decimal MB); each device cell=GB/s,time%; zero payload retains its kernel time."
    )
    for label in ["zero", "(0,.25)", "[.25,1)", "[1,4)", "[4,16)", "[16,inf)"]:
        cells = []
        for (report, *_), total in zip(artifacts, totals):
            rs = [r for r in report["samples"] if bucket(r["h2d_bytes"]) == label]
            c = stats(rs) if rs else None
            cells.append(f'{c["GBps"]:.2f},{100*c["us"]/total:.1f}' if c else "-")
        lines.append(label + " " + " ".join(cells))
    return lines


def numa_text(path):
    report = json.loads(path.read_text())
    lines = ["\nNUMA " + path.name]
    for p in report["processes"]:
        s = p.get("status", {})
        lines.append(
            f'pid={p["pid"]} stable={p["pid_stable_during_snapshot"]} identity={p["probe_identity_status"]} '
            f'cpus={s.get("Cpus_allowed_list")} allowed_mem={s.get("Mems_allowed_list")} '
            f'process_node_page_counts={p["process_pages_by_node"]} errors={len(p["errors"])}'
        )
        if p["errors"]:
            lines.append("  first_error=" + str(p["errors"][0]))
        if p.get("current_gpu_query_matches"):
            lines.append(
                "  current_GPU_PID_matches=" + str(p["current_gpu_query_matches"])
            )
        kv_groups = defaultdict(list)
        for buffer in p.get("buffer_snapshot", {}).get("buffers", []):
            regions = [
                (
                    dict(
                        policy=r["numa"].get("policy"),
                        nodes=r["numa"].get("pages_by_node"),
                        page_kB=r["numa"].get("kernelpagesize_kB"),
                        whole_vma=r["buffer_covers_whole_vma"],
                    )
                    if r.get("numa")
                    else dict(policy="unknown")
                )
                for r in buffer["regions"]
            ]
            key = json.dumps(
                dict(
                    mapped_all=buffer["mapped_bytes"] == buffer["nbytes"],
                    regions=regions,
                ),
                sort_keys=True,
            )
            kv_groups[key].append(buffer["name"])
        for placement, names in kv_groups.items():
            lines.append("  KV=" + ",".join(names) + " " + placement)
    for cmd in report["commands"]:
        if any("--query-" in arg for arg in cmd["argv"]):
            lines.extend(
                [
                    " ".join(cmd["argv"]),
                    cmd.get("stdout", "").strip(),
                    cmd.get("stderr", cmd.get("error", "")).strip(),
                ]
            )
    lines.append(
        "Process page counts do not identify KV allocations; allowed_mem is not page placement."
    )
    return lines


def copy_text(directory):
    lines = [
        "\nISOLATED_COPY " + directory.name,
        "experiment rank UUID world case CE_GB/s SM_CTA4/8/16/32/64_GB/s buffer_VMA_policy,node_counts",
    ]
    paths = sorted(directory.rglob("rank*.json"))
    if not paths:
        raise ValueError(f"No rank JSONs in {directory}")
    inventory_printed = False
    for path in paths:
        report = json.loads(path.read_text())
        if not report.get("success"):
            lines.append(
                f'{path.parent.name} rank{report.get("rank")} FAILED {report.get("error", "unknown")}'
            )
            continue
        if not inventory_printed:
            lines.extend(
                [
                    "GPU inventory (physical index/UUID/PCI bus/name):",
                    report.get("gpu_inventory", {})
                    .get("stdout", "unavailable")
                    .strip(),
                ]
            )
            inventory_printed = True
        config = report["config"]
        lines.append(
            f'{path.parent.name}/rank{report["rank"]}: CVD={report["cuda_visible_devices"]} logical={report["cuda_device"]} '
            f'source_MiB={config["source_mib"]} seed={config["seed"]} rounds={config["rounds"]} '
            f'blocks={config["blocks"]} stride/slots={config["plan_stride"]}/{config["buffer_slots"]}'
        )
        buffers = report.get("host_allocation", {}).get("buffers", [])
        regions = buffers[0].get("regions", []) if buffers else []
        placement = [
            (
                (
                    r["numa"]["policy"],
                    r["numa"]["pages_by_node"],
                    r["buffer_covers_whole_vma"],
                )
                if r.get("numa")
                else ("NUMA_unknown", {}, r["buffer_covers_whole_vma"])
            )
            for r in regions
        ]
        for case in report["cases"]:
            methods = {}
            for result in case["results"]:
                times = result["raw_copy_us"]
                if not times or any(t <= 0 or not math.isfinite(t) for t in times):
                    raise ValueError("Invalid isolated copy timing")
                if result["bytes_per_copy"] != case["payload_bytes"]:
                    raise ValueError(
                        "Isolated copy payload disagrees with case geometry"
                    )
                if not result["correctness"]["all_payload_bytes_checked"]:
                    raise ValueError("Isolated copy correctness check is missing")
                methods[result["method"]] = (
                    f'{result["bytes_per_copy"]*len(times)/sum(times)/1000:.2f}'
                )
            lines.append(
                f'{path.parent.name} {report["rank"]} {report["gpu_uuid"]} {report["world_size"]} {case["label"]} '
                f'{methods.get("ce-contiguous", "-")} '
                + "/".join(methods.get(f"sm-cta{n}", "-") for n in [4, 8, 16, 32, 64])
                + " "
                + str(placement)
            )
    return lines


def acceptance_text(path):
    samples = deque(maxlen=5)
    with path.open(errors="replace") as stream:
        for line in stream:
            if re.search(r"#running-req:\s*32(?:\D|$)", line) and "accept len:" in line:
                stamp = re.search(
                    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?", line
                )
                features = re.findall(
                    r"(?:accept len|accept rate|cap len|block accept len):\s*[\d.]+",
                    line,
                )
                samples.append(
                    (stamp.group() if stamp else "time_unknown")
                    + " "
                    + ", ".join(features)
                )
    return [
        "\nACCEPTANCE "
        + path.name
        + " (last five B32 log entries; not guaranteed to match the profile window)",
        *(samples or ["No B32 acceptance lines found."]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, action="append", default=[])
    parser.add_argument("--numa-file", type=Path)
    parser.add_argument("--copy-dir", type=Path)
    parser.add_argument("--decode-log", type=Path)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    if args.max_layers < 1 or not (
        args.analysis_dir or args.numa_file or args.copy_dir or args.decode_log
    ):
        parser.error("Provide inputs and a positive --max-layers")
    try:
        lines = [
            "H2D compact report; GB/s = decimal payload / GPU time. No TP rank times are summed.",
            "Per-cycle copy sum excludes other kernels; ratios across captures do not establish causation.",
        ]
        for directory in args.analysis_dir:
            lines.extend(analysis_text(directory, args.max_layers))
        if args.numa_file:
            lines.extend(numa_text(args.numa_file))
        if args.copy_dir:
            lines.extend(copy_text(args.copy_dir))
        if args.decode_log:
            lines.extend(acceptance_text(args.decode_log))
        text = "\n".join(line for line in lines if line != "") + "\n"
        if args.output_file:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text(text)
        print(text, end="")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
