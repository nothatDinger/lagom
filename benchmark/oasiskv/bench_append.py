"""Compare per-block and batched committed KV writes on a real CUDA device.

No attention, draft model or server is included. Host call time may include
allocator/synchronization waits; CUDA event time includes stream starvation by
the host, not just kernel execution. Neither is an end-to-end TTFT prediction.
"""

import argparse
import json
import random
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=8, help="Local KV heads after TP")
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--scope", choices=["write", "append"], default="write")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "heads",
        "key_dim",
        "value_dim",
        "block_size",
        "top_k",
        "context_length",
        "chunk_size",
        "samples",
        "warmup",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.output.exists():
        parser.error("Use a new output file")

    import torch
    from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
    from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required; no simulated timing is produced")
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    config = OasisKVConfig(token_budget=args.top_k, block_size=args.block_size)
    key = torch.randn(
        args.context_length, args.heads, args.key_dim, dtype=dtype, device="cuda"
    )
    value = torch.randn(
        args.context_length, args.heads, args.value_dim, dtype=dtype, device="cuda"
    )
    records = []

    def trial(implementation, scope, sample):
        cache = OasisKVCache(config, "cuda")
        state = cache.get(0, 0, 0, key, value)
        if implementation == "reference":
            state._append_batched = state._append_reference
        for prefix in range(0, args.context_length, args.chunk_size):
            end = min(prefix + args.chunk_size, args.context_length)
            if scope == "write":
                state.prepare_append(end)
                packed = (
                    torch.cat((key[prefix:end], value[prefix:end]), -1)
                    .transpose(0, 1)
                    .contiguous()
                )
                packed = torch.nn.functional.pad(
                    packed, (0, state.storage_dim - args.key_dim - args.value_dim)
                )
            torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(
                enable_timing=True
            ), torch.cuda.Event(enable_timing=True)
            start_event.record()
            start = time.perf_counter()
            if scope == "write":
                state._append_batched(packed, prefix)
                state.length = end
            else:
                state.append(key[prefix:end], value[prefix:end], prefix)
            submitted = time.perf_counter()
            end_event.record()
            end_event.synchronize()
            completed = time.perf_counter()
            if sample >= 0:
                records.append(
                    {
                        "implementation": implementation,
                        "sample": sample,
                        "prefix": prefix,
                        "tokens": end - prefix,
                        "host_call_ms": (submitted - start) * 1000,
                        "completion_ms": (completed - start) * 1000,
                        "stream_elapsed_ms": start_event.elapsed_time(end_event),
                        "current_chunk_backup_bytes": (
                            0
                            if state.host is None
                            else args.heads
                            * (end - prefix)
                            * state.storage_dim
                            * dtype.itemsize
                        ),
                    }
                )
        return cache, state

    # Validate the full append path before reporting either timing scope.
    old_cache, old = trial("reference", "append", -1)
    new_cache, new = trial("batched", "append", -1)
    for name in ("host", "key_min", "key_max", "resident"):
        a, b = getattr(old, name), getattr(new, name)
        if a is not None:
            if name == "host":
                a, b = (
                    a.flatten(1, 2)[:, : args.context_length],
                    b.flatten(1, 2)[:, : args.context_length],
                )
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(old.data, new.data, rtol=0, atol=0)
    old_cache.clear()
    new_cache.clear()
    del old_cache, new_cache, old, new
    order = ["reference", "batched"]
    rng = random.Random(42)
    for sample in range(-args.warmup, args.samples):
        rng.shuffle(order)
        for implementation in order:
            cache, state = trial(implementation, args.scope, sample)
            cache.clear()
            del cache, state
    summaries = []
    for implementation in ("reference", "batched"):
        for prefix in sorted({r["prefix"] for r in records}):
            selected = [
                r
                for r in records
                if r["implementation"] == implementation and r["prefix"] == prefix
            ]
            result = {"implementation": implementation, "prefix": prefix}
            for field in ("host_call_ms", "completion_ms", "stream_elapsed_ms"):
                values = sorted(r[field] for r in selected)
                result[field + "_p50"] = statistics.median(values)
                result[field + "_max"] = values[-1]
            summaries.append(result)
    report = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu": torch.cuda.get_device_name(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "correctness_passed": True,
        "scope": "write excludes allocation/packing; append includes prepare/growth/packing; both include D2H completion",
        "samples": records,
        "summary": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
