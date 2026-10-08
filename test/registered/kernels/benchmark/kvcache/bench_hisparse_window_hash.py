"""Compare main's positional LRU with PR #40670's whole-window hash planner.

Example: python bench_hisparse_window_hash.py --batch-size 4 --verify-width 6
Default: plan+IO. --mode all also measures planning and copy-only replay.
Snapshot resets occur before CUDA events. With --force-h2d-each-step, residency
invalidation occurs inside the timed plan, as in the runtime. --timing eager
uses direct launches like main's bench_hisparse.py instead of CUDA graphs.
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
)
from sglang.kernels.ops.kvcache.hisparse_window_hash import (
    allocate_window_hash_states,
    initialize_window_hash_state,
    plan_cache_window_hash_mla,
)
from sglang.srt.mem_cache.hisparse_window_hash import HiSparseWindowHashLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verify-width", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--hot-slots", type=int)
    parser.add_argument("--miss-rate", type=float, default=0.2)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("plan-io", "metadata-only", "copy-only", "all"),
        default="plan-io",
    )
    parser.add_argument(
        "--timing", choices=("cuda-graph", "eager"), default="cuda-graph"
    )
    parser.add_argument("--force-h2d-each-step", action="store_true")
    parser.add_argument(
        "--skip-io",
        action="store_true",
        default=os.environ.get("SGLANG_DEBUG_HISPARSE_SKIP_IO", "0") == "1",
        help="Keep planning and copy dispatch but skip KV bytes, like the server probe.",
    )
    parser.add_argument("--num-copy-blocks", type=int, default=4)
    parser.add_argument("--output-file", type=Path)
    opts = parser.parse_args()
    if opts.metadata_only:
        if opts.mode != "plan-io":
            parser.error("--metadata-only cannot be combined with --mode")
        opts.mode = "metadata-only"
    if opts.mode in ("metadata-only", "all") and opts.skip_io:
        parser.error("--skip-io requires --mode plan-io or copy-only")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA is required")
    if not (0 <= opts.miss_rate <= 1 and 0 <= opts.overlap <= 1):
        parser.error("miss-rate and overlap must be between 0 and 1")
    if opts.batch_size <= 0 or opts.rounds <= 0 or opts.num_copy_blocks <= 0:
        parser.error("batch-size, rounds, and num-copy-blocks must be positive")
    b, w, k = opts.batch_size, opts.verify_width, opts.top_k
    # Match resolve_dspark_device_buffer_size: force-H2D still needs room for
    # every distinct token in this live verify window.
    h = (
        w * k
        if opts.force_h2d_each_step
        else (opts.hot_slots if opts.hot_slots is not None else w * k)
    )
    layout = HiSparseWindowHashLayout(h, w, k)
    host_len = h + w * k + 1
    page_bytes = ((64 * 584 + 575) // 576) * 576
    host = torch.zeros(
        ((b * host_len + 63) // 64, page_bytes), dtype=torch.uint8, pin_memory=True
    )
    device = torch.zeros(
        ((b * (h + 1) + 63) // 64, page_bytes), dtype=torch.uint8, device="cuda"
    )
    resident = torch.full((b, h + 1), -1, device="cuda", dtype=torch.int32)
    resident[:, :h] = torch.arange(h, device="cuda", dtype=torch.int32)
    tokens = resident.clone()
    physical = torch.arange(b * (h + 1), dtype=torch.int32, device="cuda").view(
        b, h + 1
    )
    lru_initial = torch.arange(h, dtype=torch.int16, device="cuda").repeat(b, 1)
    lru = lru_initial.clone()
    state = allocate_window_hash_states(
        layout, num_layers=1, num_req_slots=b, device="cuda"
    )[0]
    reqs = torch.arange(b, dtype=torch.int64, device="cuda")
    miss_per_row = round(k * opts.miss_rate)
    shared_misses = round(miss_per_row * opts.overlap)
    generator = torch.Generator().manual_seed(40670)
    shared_hits = torch.randperm(h, generator=generator)[: k - miss_per_row]
    rows = []
    for step in range(w):
        rows.append(
            torch.cat(
                [
                    shared_hits,
                    torch.arange(h, h + shared_misses),
                    torch.arange(
                        h + shared_misses + step * (miss_per_row - shared_misses),
                        h + shared_misses + (step + 1) * (miss_per_row - shared_misses),
                    ),
                ]
            )
        )
    topk = torch.stack(rows).to(device="cuda", dtype=torch.int32).repeat(b, 1, 1)
    output = torch.full_like(topk, -1)
    args = {
        "top_k_tokens": topk,
        "device_buffer_tokens": tokens,
        "host_cache_locs": torch.arange(
            b * host_len, device="cuda", dtype=torch.int64
        ).view(b, host_len),
        "device_buffer_locs": physical,
        "host_cache": host,
        "device_buffer": device,
        "top_k_device_locs": output,
        "req_pool_indices": reqs,
        "seq_lens": torch.full((b * w,), host_len, dtype=torch.int32, device="cuda"),
        "num_real_reqs": torch.tensor([b], dtype=torch.int32, device="cuda"),
        "miss_src": torch.empty((b, w * k), dtype=torch.int64, device="cuda"),
        "miss_dst": torch.empty((b, w * k), dtype=torch.int32, device="cuda"),
        "miss_count": torch.zeros(b, dtype=torch.int32, device="cuda"),
    }

    def reset(planner):
        tokens.copy_(resident)
        if planner == "window_hash":
            initialize_window_hash_state(
                device_buffer_tokens=tokens, req_pool_indices=reqs, state=state
            )
        else:
            lru.copy_(lru_initial)

    def plan(planner):
        if opts.force_h2d_each_step:
            # Forget the previous step once, preserving all physical slot IDs.
            # The planner still deduplicates repeated tokens within this window.
            tokens.index_fill_(0, reqs, -1)
            lru[reqs] = lru_initial
            if planner == "window_hash":
                initialize_window_hash_state(
                    device_buffer_tokens=tokens, req_pool_indices=reqs, state=state
                )
        if planner == "window_hash":
            plan_cache_window_hash_mla(**args, state=state)
        else:
            legacy = dict(args)
            legacy["top_k_tokens"] = topk.view(b * w, k)
            legacy["top_k_device_locs"] = output.view(b * w, k)
            load_cache_to_device_buffer_dsv4_mla(
                **legacy,
                lru_slots=lru,
                item_size_bytes=584,
                num_top_k=k,
                hot_buffer_size=h,
                page_size=1,
                block_size=960,
                skip_io=True,
                verify_width=w,
            )

    def copy():
        copy_cache_planned_mla(
            **{
                key: args[key]
                for key in (
                    "miss_src",
                    "miss_dst",
                    "miss_count",
                    "num_real_reqs",
                    "host_cache",
                    "device_buffer",
                )
            },
            item_size_bytes=584,
            is_dsv4_layout=True,
            num_blocks=opts.num_copy_blocks,
            skip_io=opts.skip_io,
        )

    results = []
    for planner in ("lru", "window_hash"):
        modes = (
            ("plan-io", "metadata-only", "copy-only")
            if opts.mode == "all"
            else (opts.mode,)
        )
        for mode in modes:

            def run():
                if mode != "copy-only":
                    plan(planner)
                if mode != "metadata-only":
                    copy()

            def prepare():
                reset(planner)
                if mode == "copy-only":
                    plan(planner)

            for _ in range(5):
                prepare()
                run()
            torch.cuda.synchronize()
            prepare()
            graph = None
            if opts.timing == "cuda-graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
            timings = []
            for _ in range(opts.rounds):
                prepare()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                if graph is None:
                    run()
                else:
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) * 1000)
            misses = args["miss_count"].tolist()
            expected_misses = (
                torch.unique(topk[0]).numel()
                if opts.force_h2d_each_step
                else torch.unique(topk[0][topk[0] >= h]).numel()
            )
            if misses != [expected_misses] * b:
                raise RuntimeError(
                    f"{planner} miss counts {misses} != expected {expected_misses}"
                )
            results.append(
                {
                    "planner": planner,
                    "mode": mode,
                    "mean_us": statistics.fmean(timings),
                    "median_us": statistics.median(timings),
                    "p95_us": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
                    "min_us": min(timings),
                    "unique_misses_per_request": misses,
                    "h2d_bytes_per_window": (
                        sum(misses) * 584
                        if mode != "metadata-only" and not opts.skip_io
                        else 0
                    ),
                    "force_invalidation_in_timing": opts.force_h2d_each_step
                    and mode != "copy-only",
                    "persistent_metadata_bytes": (
                        layout.fixed_state_bytes(num_layers=1, num_req_slots=b)
                        if planner == "window_hash"
                        else lru.numel() * lru.element_size()
                    ),
                }
            )
    config = vars(opts).copy()
    config["output_file"] = str(opts.output_file) if opts.output_file else None
    report = json.dumps(
        {
            "config": config,
            "effective_hot_slots": h,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "reset_in_timing": False,
            "cuda_graph_timing": opts.timing == "cuda-graph",
            "results": results,
        },
        indent=2,
    )
    print(report)
    if opts.output_file:
        opts.output_file.write_text(report + "\n")


if __name__ == "__main__":
    main()
