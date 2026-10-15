"""Opt-in linear SM baseline and CPU reporting. Never selected by serving."""

import functools
import math
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_grid_perf import LEGACY_LIBRARY
from sglang.benchmark.hisparse_h2d_pair import clock_text

PAYLOAD = 32 * 100 * 584
METHODS = ("gather4", "linear4", "linear8", "ce_same", "ce_bulk")
BINDING_KEYS = {
    "same_sources",
    "same_target",
    "same_guard",
    "aligned_128",
    "same_bytes_banks",
    "only_linear_grid_changes",
    "same_chunk_work",
}
EDGE_BYTES = (8, 264, 584, PAYLOAD)
LINEAR_LIBRARY = "sgl_kernel_jit_hisparse_linear_probe_1024.so"


@functools.cache
def _module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "hisparse_linear_probe",
        1024,
        cuda_files=["hisparse_linear_probe.cuh"],
        cuda_wrappers=[("hisparse_linear_copy", "hisparse_linear_copy")],
    )


def copy_linear(source, output, blocks):
    import torch

    if torch.version.hip is not None or blocks not in (4, 8):
        raise ValueError("NVIDIA CTA4/CTA8 required")
    if any(
        t.dtype != torch.uint8 or t.ndim != 1 or not t.is_contiguous()
        for t in (source, output)
    ):
        raise ValueError("Require contiguous one-dimensional uint8 buffers")
    if source.numel() != output.numel() or source.numel() <= 0 or source.numel() % 8:
        raise ValueError("Require equal positive sizes divisible by 8")
    if (source.data_ptr() | output.data_ptr()) & 7 or output.device.type != "cuda":
        raise ValueError("Require aligned buffers and a CUDA output")
    if source.device.type != "cpu" and source.device != output.device:
        raise ValueError("Source must be CPU memory or on the output GPU")
    if source.device.type == "cpu" and not source.is_pinned():
        raise ValueError("Host source must be registered/pinned")
    _module().hisparse_linear_copy(source, output, blocks)


def validate_bytes(torch, guard, source_cpu, size, *, prefix=136):
    expected = torch.full((size + 2 * prefix,), 165, dtype=torch.uint8)
    expected[prefix : prefix + size] = source_cpu
    if not torch.equal(guard.cpu(), expected):
        raise ValueError("Linear copy payload or guard bytes differ")


def check_edge_case(torch, host, device, blocks, size, source_kind):
    """Check boundaries, 8-but-not-16 alignment, and a reset graph replay."""
    with torch.cuda.device(device):
        expected = host.view(-1)[8 : 8 + size]
        source = expected if source_kind == "host" else expected.to("cuda")
        guard = torch.full((size + 272,), 165, dtype=torch.uint8, device="cuda")
        output = guard[136 : 136 + size]
        copy_linear(source, output, blocks)
        torch.cuda.synchronize()
        validate_bytes(torch, guard, expected, size)
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if (
                torch.cuda.current_device() != device
                or torch.cuda.current_stream().device.index != device
            ):
                raise RuntimeError("Edge-check capture device mismatch")
            copy_linear(source, output, blocks)
        guard.fill_(165)
        graph.replay()
        torch.cuda.synchronize()
        validate_bytes(torch, guard, expected, size)
    return dict(
        device=device,
        blocks=blocks,
        size=size,
        source=source_kind,
        eager=True,
        graph=True,
        guards=True,
    )


def method_order(local_trial):
    return METHODS if local_trial % 2 == 0 else tuple(reversed(METHODS))


def loaded_linear_artifact(maps_text=None):
    text = Path("/proc/self/maps").read_text() if maps_text is None else maps_text
    paths = {
        f[5]
        for line in text.splitlines()
        if len(f := line.split(None, 5)) == 6
        and Path(f[5].removesuffix(" (deleted)")).name == LINEAR_LIBRARY
    }
    if len(paths) != 1:
        raise ValueError(f"Expected one linear JIT library, found {sorted(paths)}")
    path = next(iter(paths))
    return dict(variant="linear", path=path, sha256=sha256(path))


def report_status(report):
    config = report["config"]
    expected_edges = {
        (d, b, n, s)
        for d in (0, 1)
        for b in (4, 8)
        for n in EDGE_BYTES
        for s in ("host", "hbm")
    }
    edges = report.get("edge_checks", [])
    edges_ok = (
        len(edges) == len(expected_edges)
        and {(r["device"], r["blocks"], r["size"], r["source"]) for r in edges}
        == expected_edges
        and all(
            r["eager"] is True and r["graph"] is True and r["guards"] is True
            for r in edges
        )
    )
    graphs = report.get("chunk_graphs", [])
    expected_graphs = {(d, m) for d in (0, 1) for m in METHODS}
    graph_ok = (
        len(graphs) == len(expected_graphs)
        and {(g["device"], g["method"]) for g in graphs} == expected_graphs
        and all(
            g.get("correctness") is True
            and g["capture_device"] == g["device"]
            and g["per_copy_timing_events"] == 0
            and g["launches_per_chunk"] == 1
            and g["nbytes"] == (64 * 2**20 if g["method"] == "ce_bulk" else PAYLOAD)
            and g["banks"] == (7 if g["method"] == "ce_bulk" else 64)
            and g["num_blocks"]
            == ({"gather4": 4, "linear4": 4, "linear8": 8}.get(g["method"]))
            and g["block_size"]
            == (1024 if g["method"] in ("gather4", "linear4", "linear8") else None)
            and g["waves"] > 0
            and g["copy_nodes"] == g["waves"] * g["banks"]
            for g in graphs
        )
    )
    if graph_ok:
        graph_ok &= all(
            len(
                {
                    g["waves"]
                    for g in graphs
                    if g["device"] == d and g["method"] != "ce_bulk"
                }
            )
            == 1
            for d in (0, 1)
        )
    bindings = report.get("bindings", [])
    bound = (
        len(bindings) == 2
        and {r["device"] for r in bindings} == {0, 1}
        and all(
            r.get("valid") is True
            and set(r["checks"]) == BINDING_KEYS
            and all(value is True for value in r["checks"].values())
            for r in bindings
        )
    )
    artifacts = report.get("jit_artifacts", [])
    jit_ok = (
        len(artifacts) == 2
        and {a.get("variant") for a in artifacts} == {"legacy", "linear"}
        and all(
            Path(a.get("path", "")).name
            == (LINEAR_LIBRARY if a["variant"] == "linear" else LEGACY_LIBRARY)
            and isinstance(a.get("sha256"), str)
            and len(a["sha256"]) == 64
            for a in artifacts
        )
    )
    common = (
        report.get("success") is True and edges_ok and graph_ok and bound and jit_ok
    )
    windows = report.get("windows", [])
    expected = {(t, d, m) for t, d in enumerate(config["order"]) for m in METHODS}
    structure = config["mode"] == "main" and (
        config["order"] == [0, 1, 1, 0]
        and len(windows) == len(expected)
        and {(w["trial"], w["device"], w["method"]) for w in windows} == expected
        and all(
            w["sampler_enabled"]
            is (config["order"][: w["trial"]].count(w["device"]) % 2 == 1)
            and w["position"]
            == method_order(config["order"][: w["trial"]].count(w["device"])).index(
                w["method"]
            )
            for w in windows
        )
    )
    ce = [w for w in windows if w["method"] == "ce_bulk" and w["sampler_enabled"]]
    cal = len(ce) == 2 and all(
        w.get("calibration", {}).get("valid") is True for w in ce
    )
    on_off = {}
    method_timing = {(d, m): False for d in (0, 1) for m in METHODS}
    if structure:
        for d in (0, 1):
            for m in METHODS:
                pair = [w for w in windows if w["device"] == d and w["method"] == m]
                off, on = sorted(pair, key=lambda w: w["sampler_enabled"])
                if not all(
                    w["metrics"].get("valid") is True
                    and isinstance(w["metrics"].get("logical_wall_GBps"), (int, float))
                    and math.isfinite(w["metrics"]["logical_wall_GBps"])
                    and w["metrics"]["logical_wall_GBps"] > 0
                    for w in pair
                ):
                    continue
                ratio = (
                    off["metrics"]["logical_wall_GBps"]
                    / on["metrics"]["logical_wall_GBps"]
                )
                on_off[(d, m)] = ratio
                method_timing[(d, m)] = 0.9 <= ratio <= 1.1
    timing = structure and all(method_timing.values())
    # Keep every window and every original quality threshold. An invalid CE
    # window blocks comparisons involving CE, not independently valid SM pairs.
    # Correctness, bindings, JIT and the complete balanced sequence remain shared
    # prerequisites. Logical-byte/time SM comparisons do not need NVML calibration.
    sm = common and all(
        method_timing[d, m] for d in (0, 1) for m in ("gather4", "linear4", "linear8")
    )
    comparisons = dict(
        sm=sm,
        ce_same=sm and all(method_timing[d, "ce_same"] for d in (0, 1)),
        bulk_reference=sm and cal and all(method_timing[d, "ce_bulk"] for d in (0, 1)),
    )
    ready = common
    if config["mode"] == "main":
        ready &= timing and cal
    else:
        ready &= not windows
    return dict(
        ready=ready,
        edges=edges_ok,
        graphs=graph_ok,
        bindings=bound,
        jit=jit_ok,
        timing=timing,
        calibration=cal,
        on_off=on_off,
        comparisons=comparisons,
    )


def compact_report(report):
    status = report_status(report)
    lines = [
        "Linear SM baseline: same 1,868,800B/copy; scalar legacy helper; CTA4/CTA8 share buffers, banks, chunk work and one JIT artifact.",
        f'success={report["success"]} mode={report["config"]["mode"]} edges={len(report.get("edge_checks",[]))}/32 edge_valid={status["edges"]} '
        f'graphs={len(report.get("chunk_graphs",[]))}/10 graph_valid={status["graphs"]} bindings={status["bindings"]} JIT={status["jit"]} results_valid={status["ready"]}',
        f'CONFIG {report["config"]}',
    ]
    if report.get("error"):
        lines.append(f'ERROR active={report.get("active_case")} {report["error"]}')
    for d, meta in sorted(report.get("devices", {}).items()):
        lines.append(
            f'IDENTITY d{d} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
        )
    for b in report.get("host_snapshot", {}).get("buffers", []):
        lines.append(
            f'HOST bytes={b["nbytes"]} NUMA={[r.get("numa") for r in b["regions"]]}'
        )
    lines.append(f'JIT {report.get("jit_artifacts",[])}')
    for row in report.get("bindings", []):
        lines.append(
            f'BINDING d{row["device"]} valid={row["valid"]} checks={row["checks"]}'
        )
    if report["config"]["mode"] == "main":
        lines.append(
            f'TIMING valid={status["timing"]} windows={len(report.get("windows",[]))}/20 CE_calibration={status["calibration"]}'
        )
        lines.append(
            f'COMPARISONS SM_valid={status["comparisons"]["sm"]} '
            f'CEsame_valid={status["comparisons"]["ce_same"]} '
            f'bulk_reference_valid={status["comparisons"]["bulk_reference"]}'
        )
    for d in (0, 1):
        for m in METHODS:
            rows = [
                w
                for w in report.get("windows", [])
                if w["device"] == d and w["method"] == m
            ]
            if not rows:
                continue
            text = f"RESULT d{d} {m}"
            for w in sorted(rows, key=lambda r: r["sampler_enabled"]):
                x = w["metrics"]
                label = "ON" if w["sampler_enabled"] else "OFF"
                if x.get("logical_wall_GBps") is not None:
                    text += f' {label}_GBps={x["logical_wall_GBps"]:.4f} enqueue={x["max_enqueue_fraction"]:.4f} chunkCV={x["chunk_gpu_cv"]:.4f}'
                text += f' valid={x["valid"]}'
                if x.get("rates", {}).get("rx_kbps") is not None:
                    r = x["rates"]
                    text += f' rawRx_KBps={r["rx_kbps"]:.1f} n={r["samples"]} coverage={r["coverage"]:.3f} RxCV={r["rx_cv"]}'
                if not x["valid"]:
                    text += f' reason={x.get("reason")}'
            text += f' ON/OFF_us={status["on_off"].get((d,m))}'
            lines.append(text)
        states = [
            s
            for w in report.get("windows", [])
            if w["device"] == d
            for s in w.get("states", [])
        ]
        lines.append(
            f"STATE d{d} "
            + (
                clock_text([dict(before=s) for s in states])
                if states
                else "not_collected"
            )
        )
    if status["comparisons"]["sm"]:
        index = {
            (w["device"], w["method"], w["sampler_enabled"]): w["metrics"].get(
                "logical_wall_GBps"
            )
            for w in report["windows"]
        }
        for d in (0, 1):
            lines.append(
                f'RATIOS_SM d{d} OFF/ON CTA8/4_us={[round(index[d,"linear4",p]/index[d,"linear8",p],4) for p in (False,True)]} '
                f'linear4/gather_us={[round(index[d,"gather4",p]/index[d,"linear4",p],4) for p in (False,True)]}'
            )
        lines.append(
            "CROSS_SM GPU5/GPU4_us OFF/ON "
            + str(
                {
                    m: [
                        round(index[0, m, p] / index[1, m, p], 4) for p in (False, True)
                    ]
                    for m in ("gather4", "linear4", "linear8")
                }
            )
        )
        for d in (0, 1):
            if status["comparisons"]["ce_same"]:
                lines.append(
                    f"RATIOS_CEsame d{d} OFF/ON linear4/CEsame_GBps="
                    f'{[round(index[d,"linear4",p]/index[d,"ce_same",p],4) for p in (False,True)]}'
                )
            if status["comparisons"]["bulk_reference"]:
                lines.append(
                    f"BULK_REFERENCE d{d} OFF/ON linear4/CEbulk_GBps="
                    f'{[round(index[d,"linear4",p]/index[d,"ce_bulk",p],4) for p in (False,True)]}'
                )
    lines.append(
        "Dense vs gather changes plans, source footprint, C4 splitting and destination layout together; this is a baseline, not source-only causality or a production optimization. NVML rates are approximate."
    )
    return "\n".join(lines) + "\n"
