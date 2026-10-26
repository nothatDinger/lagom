"""Pinned host UVA copy: legacy vs tiled refill/prefetch, optionally with GEMM.

Run on the server's GPU/NUMA node. Counts are controlled inputs, not inferred
from the launch grid. CUDA graphs time copy only; selection/publication and
end-to-end serving need a separate server A/B run.
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def csv_ints(value):
    return [int(item) for item in value.split(",")]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--batch", type=int, default=8, help="Captured batch, including padding"
    )
    p.add_argument("--live-batch", type=int, default=7)
    p.add_argument("--heads", type=int, default=8, help="Local KV heads after TP")
    p.add_argument(
        "--storage", type=int, default=256, help="Padded K+V elements per token"
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--context-length", type=int, default=16384)
    p.add_argument("--top-k", type=int, default=2048)
    p.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    counts = p.add_mutually_exclusive_group()
    counts.add_argument("--misses-per-head", type=csv_ints, default=None)
    counts.add_argument(
        "--total-misses", type=csv_ints, help="Distribute exact total over live heads"
    )
    p.add_argument("--refill-ctas-per-sm", type=int, default=8)
    p.add_argument("--prefetch-ctas-per-sm", type=csv_ints, default=[1])
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--flush-mib",
        type=int,
        default=128,
        help="L2 eviction working set, outside timing",
    )
    p.add_argument(
        "--matmul-size",
        type=int,
        default=0,
        help="Optional concurrent square GEMM; zero disables",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="JSON report")
    return p


def make_counts(batch, live, heads, slots, *, per_head=None, total=None):
    """Validate the workload before allocating pinned memory."""
    if not 0 < live <= batch or heads <= 0 or slots <= 0:
        raise ValueError("Require 0 < live_batch <= batch, positive heads and slots")
    groups = live * heads
    if (per_head is None) == (total is None):
        raise ValueError("Specify exactly one miss count mode")
    total = per_head * groups if per_head is not None else total
    if total < 0 or total > groups * slots:
        raise ValueError("Miss count must fit the live heads' residency slots")
    base, remainder = divmod(total, groups)
    return [base + (i < remainder) for i in range(groups)] + [0] * (
        (batch - live) * heads
    )


def distribution(values):
    ordered = sorted(values)
    return {
        "min_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_ms": ordered[-1],
    }


def main():
    p = parser()
    args = p.parse_args()
    for name in (
        "block_size",
        "storage",
        "context_length",
        "top_k",
        "samples",
        "refill_ctas_per_sm",
    ):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.top_k % args.block_size or args.context_length < args.top_k:
        p.error("Require top_k divisible by block_size and context_length >= top_k")
    if (
        min(args.prefetch_ctas_per_sm) <= 0
        or min(args.warmup, args.flush_mib, args.matmul_size) < 0
    ):
        p.error(
            "CTA budgets must be positive; warmup, flush_mib and matmul_size non-negative"
        )
    slots = args.top_k // args.block_size
    workloads = []
    try:
        if args.total_misses is not None:
            for count in args.total_misses:
                workloads.append(
                    make_counts(
                        args.batch, args.live_batch, args.heads, slots, total=count
                    )
                )
        else:
            for count in (
                args.misses_per_head
                if args.misses_per_head is not None
                else [0, 1, 8, 32, slots]
            ):
                workloads.append(
                    make_counts(
                        args.batch, args.live_batch, args.heads, slots, per_head=count
                    )
                )
    except ValueError as exc:
        p.error(str(exc))

    import torch
    import triton
    import triton.language as tl
    from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
    from sglang.srt.mem_cache.oasiskv_graph import copy_launch_geometry

    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required; no simulated performance results are produced")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    prop = torch.cuda.get_device_properties(device)
    dtype = getattr(torch, args.dtype)
    if args.storage * dtype.itemsize % 16:
        p.error("storage must use the production layout's 16-byte padding")
    capacity = triton.cdiv(args.context_length, args.block_size)
    host = torch.empty(
        (args.live_batch, args.heads, capacity, args.block_size, args.storage),
        dtype=dtype,
        pin_memory=True,
    ).random_(0, 128)
    data = torch.full(
        (args.batch, args.heads, slots + 1, args.block_size, args.storage),
        -1,
        dtype=dtype,
        device=device,
    )
    table_cpu = torch.zeros((args.batch, kernels.FIELDS), dtype=torch.int64)
    table_cpu[:, kernels.PREFIX] = -1
    for b in range(args.live_batch):
        table_cpu[b, kernels.HOST] = host[b].data_ptr()
        table_cpu[b, kernels.DATA] = data[b].data_ptr()
        table_cpu[b, kernels.CAPACITY] = capacity
        table_cpu[b, kernels.PREFIX] = args.context_length
    table = table_cpu.to(device)
    sources_cpu = torch.stack(
        [torch.randperm(capacity)[:slots] for _ in range(args.batch * args.heads)]
    ).reshape(args.batch, args.heads, slots)
    sources = sources_cpu.to(device)
    destinations = torch.arange(slots, device=device).repeat(args.batch, args.heads, 1)
    counts = torch.zeros((args.batch, args.heads), dtype=torch.int32, device=device)
    flush = torch.zeros(args.flush_mib * 1024 * 1024, dtype=torch.uint8, device=device)
    main_stream, side = torch.cuda.Stream(), torch.cuda.Stream()
    gemm = None
    if args.matmul_size:
        mat_a = torch.randn(
            (args.matmul_size, args.matmul_size), dtype=dtype, device=device
        )
        mat_b = torch.randn_like(mat_a)
        mat_out = torch.empty_like(mat_a)

        def gemm():
            torch.mm(mat_a, mat_b, out=mat_out)

    def launch(variant):
        legacy = variant["name"] == "legacy"
        (kernels.copy_misses_legacy if legacy else kernels.copy_misses)[
            variant["grid"]
        ](
            table,
            sources,
            destinations,
            counts,
            HEADS=args.heads,
            SLOTS=slots,
            BLOCK=args.block_size,
            STORAGE=args.storage,
            DTYPE=getattr(tl, args.dtype),
            TILE=variant["tile"],
            WORKERS=variant["grid"][-1],
            num_warps=4,
            **({} if legacy else {"GROUPS": args.batch * args.heads}),
        )

    variants = [{"name": "legacy", "tile": 1024, "grid": (args.batch, args.heads, 4)}]
    for name, budget in [("refill", args.refill_ctas_per_sm)] + [
        (f"prefetch_{n}", n) for n in args.prefetch_ctas_per_sm
    ]:
        tile, grid = copy_launch_geometry(
            batch=args.batch,
            heads=args.heads,
            slots=slots,
            block=args.block_size,
            storage=args.storage,
            element_size=dtype.itemsize,
            sm_count=prop.multi_processor_count,
            ctas_per_sm=budget,
        )
        variants.append(
            {"name": name, "tile": tile, "grid": grid, "ctas_per_sm": budget}
        )

    # Warm compilation and cuBLAS before capture. Kernels read changing Counts
    # from stable addresses; every workload reuses these same graphs.
    main_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(main_stream):
        for variant in variants:
            launch(variant)
        if gemm:
            gemm()
    torch.cuda.synchronize()

    def capture(variant=None, compute=False):
        graph = torch.cuda.CUDAGraph()
        body_events = tuple(
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
        )
        compute_events = (
            tuple(torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            if compute
            else None
        )
        with torch.cuda.graph(graph, stream=main_stream):
            body_events[0].record()
            if variant is not None and compute:
                side.wait_stream(main_stream)
                with torch.cuda.stream(side):
                    launch(variant)
            elif variant is not None:
                launch(variant)
            if compute:
                compute_events[0].record()
                gemm()
                compute_events[1].record()
                if variant is not None:
                    main_stream.wait_stream(side)
            body_events[1].record()
        torch.cuda.synchronize()
        return graph, compute_events, body_events

    graphs = {v["name"]: capture(v) for v in variants}
    joint_graphs = (
        {v["name"]: capture(v, compute=True) for v in variants} if gemm else {}
    )
    compute_graph = capture(compute=True) if gemm else None

    def measure(captured):
        graph, compute_events, body_events = captured
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        end = torch.cuda.Event()
        times, compute_times = [], []
        for _ in range(args.samples):
            # Evict repeated host payload from GPU L2 before, not within, the
            # timed region. Report the working-set size so this is auditable.
            flush.add_(1)
            graph.replay()
            end.record()
            end.synchronize()
            # Timing nodes are inside the graph, excluding CPU submission gaps.
            times.append(body_events[0].elapsed_time(body_events[1]))
            if compute_events:
                compute_times.append(compute_events[0].elapsed_time(compute_events[1]))
        return distribution(times), (
            distribution(compute_times) if compute_times else None
        )

    report = {
        "device": prop.name,
        "sm_count": prop.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "host_bytes": host.nbytes,
        "variants": variants,
        "results": [],
    }
    compute_alone = measure(compute_graph)[1] if compute_graph else None
    report["compute_alone"] = compute_alone
    order_rng = random.Random(args.seed)
    for workload in workloads:
        counts.copy_(torch.tensor(workload, dtype=torch.int32).reshape_as(counts))
        total = sum(workload)
        payload_bytes = total * args.block_size * args.storage * dtype.itemsize
        expected = torch.full_like(data, -1, device="cpu")
        for b in range(args.live_batch):
            for h in range(args.heads):
                n = workload[b * args.heads + h]
                expected[b, h, :n] = host[b, h, sources_cpu[b, h, :n]]
        ordered_variants = list(variants)
        order_rng.shuffle(ordered_variants)
        for variant in ordered_variants:
            name = variant["name"]
            data.fill_(-1)
            graphs[name][0].replay()
            if not torch.equal(
                data.cpu().view(torch.uint8), expected.view(torch.uint8)
            ):
                raise AssertionError(
                    f"Copy validation failed: {name}, total_misses={total}"
                )
            with torch.cuda.nvtx.range(f"oasiskv/{name}/misses={total}"):
                copy_time, _ = measure(graphs[name])
                joint_time, concurrent_compute = (
                    measure(joint_graphs[name]) if gemm else (None, None)
                )
            record = {
                "variant": name,
                "counts": workload,
                "total_misses": total,
                "payload_bytes": payload_bytes,
                "copy": copy_time,
                "payload_GBps": payload_bytes / (copy_time["p50_ms"] * 1e6),
                "joint": joint_time,
                "concurrent_compute": concurrent_compute,
                "compute_slowdown": (
                    concurrent_compute["p50_ms"] / compute_alone["p50_ms"]
                    if gemm
                    else None
                ),
            }
            report["results"].append(record)
            print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
