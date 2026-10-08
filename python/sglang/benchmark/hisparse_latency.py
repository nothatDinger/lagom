"""Report real HiSparse plan+IO GPU time and its share of measured TPOT.

Use one rank's decode-only Chrome trace and a matching serving benchmark JSONL.
The skip-IO ablation reports exposed IO cost separately from GPU kernel time.
This module uses only the standard library and does not instrument serving.
"""

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def read_benchmark(path: Path, index: int = -1) -> dict:
    raw = path.read_text()
    try:
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No benchmark results in {path}")
    result = rows[index]
    tpot = result.get("mean_tpot_ms")
    if not isinstance(tpot, (int, float)) or not math.isfinite(tpot) or tpot <= 0:
        raise ValueError("mean_tpot_ms must be finite and positive")
    if result.get("completed", 1) <= 0:
        raise ValueError("Benchmark has no completed requests")
    return result


def compare_skip_io(normal: dict, skip_io: dict) -> dict:
    # A different workload would give a misleading ablation percentage.
    for key in (
        "backend",
        "dataset_name",
        "request_rate",
        "max_concurrency",
        "random_input_len",
        "random_output_len",
        "random_range_ratio",
        "completed",
        "total_output_tokens",
    ):
        if normal.get(key) != skip_io.get(key):
            raise ValueError(f"Normal and skip-IO benchmarks differ in {key}")
    for key in (
        "model_path",
        "hisparse_config",
        "speculative_algorithm",
        "tp_size",
        "dp_size",
    ):
        a = (normal.get("server_info") or {}).get("server_args", {}).get(key)
        b = (skip_io.get("server_info") or {}).get("server_args", {}).get(key)
        if a != b:
            raise ValueError(
                f"Normal and skip-IO server configurations differ in {key}"
            )
    delta = normal["mean_tpot_ms"] - skip_io["mean_tpot_ms"]
    return {
        "normal_tpot_ms": normal["mean_tpot_ms"],
        "skip_io_tpot_ms": skip_io["mean_tpot_ms"],
        "exposed_io_ms_per_token": delta,
        "exposed_io_share_pct": 100 * delta / normal["mean_tpot_ms"],
        "normal_accept_length": normal.get("accept_length"),
        "skip_io_accept_length": skip_io.get("accept_length"),
        "interpretation": "End-to-end IO ablation; planning remains. Includes changes in overlap and acceptance. Negative deltas are preserved.",
    }


def _kernel_kind(name: str):
    if "initialize_hisparse_spec_state_kernel" in name:
        return "hash_initialization"
    if "load_cache_to_device_buffer_spec_gather_kernel" in name:
        return "plan"
    if "load_cache_to_device_buffer_spec_commit_kernel" in name:
        return "plan"
    if "copy_cache_planned_kernel" in name:
        return "io"
    if "load_cache_to_device_buffer_kernel" in name:
        return "fused_plan_io"
    return None


def _union_us(intervals):
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_trace(
    path: Path,
    *,
    decode_steps: int,
    tokens_per_step: float,
    tpot_ms: float,
    device: int | None = None,
    start_us: float | None = None,
    end_us: float | None = None,
) -> dict:
    if decode_steps <= 0 or not math.isfinite(tokens_per_step) or tokens_per_step <= 0:
        raise ValueError(
            "decode_steps and per-request tokens_per_step must be positive"
        )
    if not math.isfinite(tpot_ms) or tpot_ms <= 0:
        raise ValueError("tpot_ms must be finite and positive")
    if start_us is not None and end_us is not None and start_us >= end_us:
        raise ValueError("Trace start must precede trace end")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = []
    for event in events:
        if event.get("ph") != "X" or "kernel" not in event.get("cat", "").lower():
            continue
        kind = _kernel_kind(event.get("name", ""))
        if kind is None:
            continue
        start, stop = event["ts"], event["ts"] + event["dur"]
        if not math.isfinite(start) or not math.isfinite(stop):
            raise ValueError("Kernel timestamps and durations must be finite")
        start = max(start, start_us) if start_us is not None else start
        stop = min(stop, end_us) if end_us is not None else stop
        if stop <= start:
            continue
        gpu = event.get("args", {}).get("device")
        if gpu is None:
            raise ValueError("HiSparse kernel event has no device ID")
        kernels.append((int(gpu), kind, start, stop))
    devices = {gpu for gpu, _, _, _ in kernels}
    if not devices:
        raise ValueError(
            "No HiSparse kernels found. Export a per-kernel CUDA trace, "
            "including graph nodes, rather than cudaGraphLaunch alone."
        )
    if device is None:
        if len(devices) != 1:
            raise ValueError(
                "Select one GPU with --device; do not sum tensor-parallel ranks"
            )
        device = devices.pop()
    kernels = [entry for entry in kernels if entry[0] == device]
    intervals = [
        (start, stop)
        for _, kind, start, stop in kernels
        if kind != "hash_initialization"
    ]
    if not intervals:
        raise ValueError(
            "No HiSparse plan+IO kernels found. Export a per-kernel CUDA trace, including graph nodes, rather than cudaGraphLaunch alone."
        )
    by_kind = defaultdict(lambda: {"count": 0, "gpu_us": 0.0})
    for _, kind, start, stop in kernels:
        by_kind[kind]["count"] += 1
        by_kind[kind]["gpu_us"] += stop - start
    union_us = _union_us(intervals)
    per_token_ms = union_us / 1000 / decode_steps / tokens_per_step
    return {
        "trace": str(path),
        "device": device,
        "decode_steps": decode_steps,
        "tokens_per_step_per_request": tokens_per_step,
        "kernels": dict(by_kind),
        "plan_io_gpu_sum_us": sum(stop - start for start, stop in intervals),
        "plan_io_gpu_union_us": union_us,
        "plan_io_gpu_ms_per_step": union_us / 1000 / decode_steps,
        "plan_io_gpu_ms_per_output_token": per_token_ms,
        "plan_io_gpu_share_of_tpot_pct": 100 * per_token_ms / tpot_ms,
        "interpretation": "GPU time occupied by plan+IO, counting overlapping HiSparse kernels once. Excludes launch gaps and residency invalidation. This is not the marginal TPOT reduction from removing plan+IO.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-result", type=Path, required=True)
    parser.add_argument("--skip-io-result", type=Path)
    parser.add_argument("--result-index", type=int, default=-1)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument(
        "--tokens-per-step",
        type=float,
        help="Average committed output tokens per request per profiled decode step; not verify width or batch-total tokens.",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--trace-start-us", type=float)
    parser.add_argument("--trace-end-us", type=float)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    if not args.trace and not args.skip_io_result:
        parser.error("Provide --trace and/or --skip-io-result")
    if args.trace and (args.decode_steps is None or args.tokens_per_step is None):
        parser.error("--trace requires --decode-steps and --tokens-per-step")
    try:
        normal = read_benchmark(args.normal_result, args.result_index)
        report = {"normal_tpot_ms": normal["mean_tpot_ms"]}
        if args.trace:
            report["plan_io"] = summarize_trace(
                args.trace,
                decode_steps=args.decode_steps,
                tokens_per_step=args.tokens_per_step,
                tpot_ms=normal["mean_tpot_ms"],
                device=args.device,
                start_us=args.trace_start_us,
                end_us=args.trace_end_us,
            )
        if args.skip_io_result:
            report["io_ablation"] = compare_skip_io(
                normal, read_benchmark(args.skip_io_result, args.result_index)
            )
    except (ValueError, KeyError, IndexError, OSError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, allow_nan=False)
    print(output)
    if args.output_file:
        args.output_file.write_text(output + "\n")


if __name__ == "__main__":
    main()
