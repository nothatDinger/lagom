"""CPU validation for the physical GPU4/5 CUDA enumeration control."""

import math
import statistics
from pathlib import Path

from sglang.benchmark.hisparse_h2d_linear import (
    EDGE_BYTES,
    LINEAR_LIBRARY,
    PAYLOAD,
)
from sglang.benchmark.hisparse_h2d_pair import clock_text

METHODS = ("linear4", "ce_bulk")
HOST_BYTES = 512 * 2**20
BINDINGS = {"shared_host", "same_inputs", "aligned_128", "pinned"}


def configuration(physical_order):
    order = list(physical_order)
    if order not in ([4, 5], [5, 4]):
        raise ValueError("Require physical order 4,5 or 5,4")
    return dict(
        physical_order=order,
        visible_devices=",".join(map(str, order)),
        first_allocation_device=0,
        register_device=order.index(5),
        register_physical=5,
        register_flags=1,
        order=[0, 1, 1, 0],
        payload=PAYLOAD,
        banks=64,
        bulk_bytes=64 * 2**20,
        bulk_banks=7,
        linear_waves=32,
        bulk_waves=12,
        host_bytes=HOST_BYTES,
        num_blocks=4,
        block_size=1024,
        seed=0,
        warm_s=2.0,
        duration_s=2.5,
        trim_s=0.3,
        rx_sampler=False,
    )


def sequence():
    return [
        (t, d, m, p)
        for t, d in enumerate([0, 1, 1, 0])
        for p, m in enumerate(METHODS if t < 2 else tuple(reversed(METHODS)))
    ]


def devices(report):
    return {int(d): row for d, row in report.get("devices", {}).items()}


def local_host(snapshot):
    buffers = snapshot.get("buffers", [])
    if snapshot.get("errors") or len(buffers) != 1:
        return False
    b = buffers[0]
    regions = b.get("regions", [])
    return (
        b.get("nbytes") == b.get("mapped_bytes") == HOST_BYTES
        and bool(regions)
        and sum(r["overlap_bytes"] for r in regions) == HOST_BYTES
        and all(
            r.get("buffer_covers_whole_vma") is True
            and (r.get("numa") or {}).get("policy") == "bind:1"
            and set(r["numa"].get("pages_by_node", {})) == {"1"}
            and r["numa"]["pages_by_node"]["1"] > 0
            and r["numa"].get("kernelpagesize_kB") == 4
            for r in regions
        )
    )


def report_status(report):
    c, meta = report["config"], devices(report)
    identity = (
        c.get("physical_order") in ([4, 5], [5, 4])
        and c == configuration(c["physical_order"])
        and set(meta) == {0, 1}
        and [meta[d]["physical_index"] for d in (0, 1)] == c["physical_order"]
        and len({r["uuid"] for r in meta.values()}) == 2
        and all(
            r["pci_bus"]
            .lower()
            .endswith("98:00.0" if r["physical_index"] == 4 else "b8:00.0")
            for r in meta.values()
        )
        and report.get("cuda_was_initialized") is False
        and report.get("allocation_order") == [0, 1]
        and report.get("actual_visible_devices") == c["visible_devices"]
        and report.get("registration", {}).get("device") == c["register_device"]
        and report.get("registration", {}).get("physical") == 5
        and report.get("registration", {}).get("flags") == 1
        and report.get("registration", {}).get("rc") == 0
    )
    expected = {
        (d, 4, n, s) for d in (0, 1) for n in EDGE_BYTES for s in ("host", "hbm")
    }
    edges = report.get("edge_checks", [])
    correct = (
        len(edges) == 16
        and {(r["device"], r["blocks"], r["size"], r["source"]) for r in edges}
        == expected
        and all(
            all(r.get(k) is True for k in ("eager", "graph", "guards")) for r in edges
        )
    )
    graphs = report.get("chunk_graphs", [])
    graph_ok = (
        len(graphs) == 4
        and {(r["device"], r["method"]) for r in graphs}
        == {(d, m) for d in (0, 1) for m in METHODS}
        and all(
            r.get("correctness") is True
            and r["capture_device"] == r["device"]
            and r["per_copy_timing_events"] == 0
            and r["launches_per_chunk"] == 1
            and r["nbytes"]
            == (PAYLOAD if r["method"] == "linear4" else c["bulk_bytes"])
            and r["banks"] == (64 if r["method"] == "linear4" else 7)
            and r["num_blocks"] == (4 if r["method"] == "linear4" else None)
            and r["block_size"] == (1024 if r["method"] == "linear4" else None)
            and r["waves"]
            == (c["linear_waves"] if r["method"] == "linear4" else c["bulk_waves"])
            and r["copy_nodes"] == r["waves"] * r["banks"]
            for r in graphs
        )
        and len({r["waves"] for r in graphs if r["method"] == "linear4"}) == 1
    )
    bindings = report.get("bindings", [])
    bound = (
        len(bindings) == 2
        and {r["device"] for r in bindings} == {0, 1}
        and all(
            set(r["checks"]) == BINDINGS
            and all(v is True for v in r["checks"].values())
            for r in bindings
        )
    )
    artifact = report.get("jit_artifact", {})
    jit = (
        artifact.get("variant") == "linear"
        and Path(artifact.get("path", "")).name == LINEAR_LIBRARY
        and len(artifact.get("sha256", "")) == 64
    )
    rows = report.get("windows", [])
    timing = [
        (r["trial"], r["device"], r["method"], r["position"]) for r in rows
    ] == sequence() and all(
        r.get("sampler_enabled") is False
        and r["metrics"].get("valid") is True
        and math.isfinite(r["metrics"].get("logical_wall_GBps", float("nan")))
        and r["metrics"]["logical_wall_GBps"] > 0
        for r in rows
    )
    drift = {}
    if timing:
        for d in (0, 1):
            for m in METHODS:
                pair = rates(report, d, m)
                drift[f"d{d}:{m}"] = pair[0] / pair[1]
        timing = all(0.9 <= r <= 1.1 for r in drift.values())
    bulk = len([r for r in rows if r["method"] == "ce_bulk"]) == 4 and all(
        r["metrics"].get("logical_wall_GBps", 0) >= 45
        for r in rows
        if r["method"] == "ce_bulk"
    )
    status = dict(
        identity=identity,
        correctness=correct,
        graphs=graph_ok,
        host=local_host(report.get("host_snapshot", {})),
        bindings=bound,
        jit=jit,
        timing=timing,
        bulk_reference=bulk,
    )
    return dict(
        **status,
        ready=report.get("success") is True and all(status.values()),
        drift=drift,
    )


def rates(report, device, method):
    return [
        r["metrics"]["logical_wall_GBps"]
        for r in report["windows"]
        if (r["device"], r["method"]) == (device, method)
    ]


def compact_report(report):
    s = report_status(report)
    lines = [
        f'success={report["success"]} results_valid={s["ready"]} gates={s}',
        f'CONFIG {report["config"]}',
    ]
    if report.get("error"):
        lines.append(f'ERROR active={report.get("active_case")} {report["error"]}')
    lines.append(
        f'INITIALIZATION cuda_was_initialized={report.get("cuda_was_initialized")} allocation_order={report.get("allocation_order")} registration={report.get("registration")}'
    )
    for d, meta in sorted(devices(report).items()):
        lines.append(
            f'IDENTITY d{d} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
        )
        for method in METHODS:
            rows = [
                r
                for r in report.get("windows", [])
                if (r["device"], r["method"]) == (d, method)
            ]
            if rows:
                quality = [
                    {
                        k: r["metrics"].get(k)
                        for k in (
                            "valid",
                            "max_enqueue_fraction",
                            "chunk_gpu_cv",
                            "gpu_wall_ratio",
                            "reason",
                        )
                    }
                    for r in rows
                ]
                lines.append(
                    f'RESULT GPU{meta["physical_index"]} d{d} {method} GBps={[r["metrics"].get("logical_wall_GBps") for r in rows]} quality={quality}'
                )
        states = [
            state
            for r in report.get("windows", [])
            if r["device"] == d
            for state in r["states"]
        ]
        lines.append(
            f'STATE GPU{meta["physical_index"]} '
            + (
                clock_text([dict(before=x) for x in states])
                if states
                else "not_collected"
            )
        )
    lines.append(
        f'HOST valid={s["host"]} buffers={[dict(nbytes=b["nbytes"], regions=b["regions"]) for b in report.get("host_snapshot", {}).get("buffers", [])]}'
    )
    lines.append(f'JIT {report.get("jit_artifact")}')
    return "\n".join(lines) + "\n"


def compare_reports(normal, reverse):
    reports = [normal, reverse]
    statuses = [report_status(r) for r in reports]
    physical = [
        {v["physical_index"]: (v["uuid"], v["pci_bus"]) for v in devices(r).values()}
        for r in reports
    ]
    checks = dict(
        runs=all(s["ready"] for s in statuses),
        order=[r["config"]["physical_order"] for r in reports] == [[4, 5], [5, 4]],
        identities=physical[0] == physical[1],
        same_jit=normal.get("jit_artifact", {}).get("sha256")
        == reverse.get("jit_artifact", {}).get("sha256"),
        same_code=bool(normal.get("code", {}).get("sha256"))
        and normal.get("code", {}).get("sha256")
        == reverse.get("code", {}).get("sha256"),
        same_runtime=bool(normal.get("runtime"))
        and normal.get("runtime") == reverse.get("runtime"),
        independent_processes=normal.get("pid") is not None
        and normal.get("pid") != reverse.get("pid"),
    )
    result = dict(
        results_valid=all(checks.values()), checks=checks, runs=statuses, comparisons=[]
    )
    if result["results_valid"]:
        for report in reports:
            by_physical = {v["physical_index"]: d for d, v in devices(report).items()}
            result["comparisons"].append(
                dict(
                    physical_order=report["config"]["physical_order"],
                    GPU5_over_GPU4_time={
                        m: [
                            a / b
                            for a, b in zip(
                                rates(report, by_physical[4], m),
                                rates(report, by_physical[5], m),
                            )
                        ]
                        for m in METHODS
                    },
                    mean_GBps={
                        f"GPU{p}": {
                            m: statistics.fmean(rates(report, by_physical[p], m))
                            for m in METHODS
                        }
                        for p in (4, 5)
                    },
                )
            )
    return result


def pair_compact(normal, reverse, result):
    lines = [
        f'DEVICE_ORDER results_valid={result["results_valid"]} checks={result["checks"]}'
    ]
    for label, report in zip(("normal", "reversed"), (normal, reverse)):
        lines.append(f"RUN {label} " + compact_report(report))
    for row in result["comparisons"]:
        lines.append(
            f'CROSS physical_order={row["physical_order"]} GPU5/GPU4_time={row["GPU5_over_GPU4_time"]} mean_GBps={row["mean_GBps"]}'
        )
    lines.append(
        "This changes logical enumeration and first explicit CUDA allocation together. It does not separate board from PCIe path or prove a particular mapping mechanism."
    )
    return "\n".join(lines) + "\n"
