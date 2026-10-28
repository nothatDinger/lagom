"""HBM-only OasisKV attention A/B, with identical KV, queries and masks.

Times CUDA graphs with and without foreground residency publication. No host
KV transfers, top-k prediction or model weights are included. Useful KV bytes
are unique bytes demanded, not measured DRAM traffic; collect Nsight counters
to distinguish HBM, L2 and repeated head loads.
"""

import argparse
import itertools
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path


def ints(value):
    return [int(x) for x in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=ints, default=[1, 8, 32])
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--heads", type=int, default=8, help="Local KV heads after TP")
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--width", type=int, choices=[1, 2], default=2)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--softcap", type=float, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--flush-mib",
        type=int,
        default=128,
        help="L2 eviction outside timing; 0 for warm-cache runs",
    )
    parser.add_argument(
        "--head-tiles", type=ints, help="Optional geometry sweep (1,2,4,8)"
    )
    parser.add_argument("--splits", type=ints, help="Optional geometry sweep (1..32)")
    parser.add_argument(
        "--tiles", type=ints, help="Optional geometry sweep (32,64,128)"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.batch) <= args.padding
        or args.padding < 0
        or args.samples <= 0
        or args.flush_mib < 0
    ):
        parser.error("Require batch > padding >= 0, samples > 0 and flush_mib >= 0")

    import torch
    import triton
    import triton.language as tl
    from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
    from sglang.srt.mem_cache.oasiskv_attention import (
        AttentionGeometry,
        GroupedAttentionWorkspace,
        attention_geometry,
        use_grouped_attention,
    )
    from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig

    config = OasisKVConfig(token_budget=args.top_k, block_size=args.block_size)
    dtype = getattr(torch, args.dtype)
    if args.heads <= 0 or not use_grouped_attention(
        config, dtype, args.group, args.key_dim, args.value_dim
    ):
        parser.error(
            "This benchmark requires a supported GQA geometry (group >= 2, dimensions <= 256, power-of-two block <= 64)"
        )
    if args.head_tiles and any(x not in (1, 2, 4, 8) for x in args.head_tiles):
        parser.error("head-tiles must be 1,2,4,8")
    if args.splits and any(x < 1 or x > 32 for x in args.splits):
        parser.error("splits must be in 1..32")
    if args.tiles and any(
        x not in (32, 64, 128) or x < args.block_size for x in args.tiles
    ):
        parser.error("tiles must be 32,64,128 and >= block-size")
    if not torch.cuda.is_available():
        parser.error(
            "CUDA is required; this script does not fabricate performance measurements"
        )
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )
    report = {
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "byte_definition": "unique useful KV bytes, not measured DRAM traffic",
        "results": [],
    }

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        times = []
        for _ in range(args.samples):
            if flush.numel():
                flush.fill_(1)  # ordered before begin, excluded from the interval
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
        ordered = sorted(times)
        return {
            "p50_ms": statistics.median(times),
            "p95_ms": ordered[math.ceil(0.95 * len(times)) - 1],
        }

    def run_batch(batch):
        live, heads, group = batch - args.padding, args.heads, args.group
        kd, vd, width, slots, block = (
            args.key_dim,
            args.value_dim,
            args.width,
            config.history_blocks,
            config.block_size,
        )
        storage = triton.cdiv((kd + vd) * dtype.itemsize, 16) * 16 // dtype.itemsize
        w = GroupedAttentionWorkspace(
            batch, heads, group, kd, vd, config, device, max_splits=32
        )
        data = torch.randn(
            batch * heads * (slots + 1), block, storage, device=device, dtype=dtype
        )
        # Random physical order across a common pool. Indirection is real, while
        # values and addresses are identical for the two implementations.
        page_ids = torch.randperm(batch * heads * (slots + 1), device=device)
        pointers = (
            data.data_ptr() + page_ids * block * storage * dtype.itemsize
        ).reshape(batch, heads, slots + 1)
        ids = torch.arange(slots, device=device).expand(batch, heads, slots).clone()
        ages = torch.ones_like(ids)
        selected = torch.ones_like(ids, dtype=torch.bool)
        table_cpu = torch.zeros(batch, kernels.FIELDS, dtype=torch.int64)
        table_cpu[:, kernels.PREFIX] = -1
        for b in range(live):
            table_cpu[b, kernels.PREFIX] = (slots + 1) * block - 1
            table_cpu[b, kernels.RESIDENT] = ids[b].data_ptr()
            table_cpu[b, kernels.AGES] = ages[b].data_ptr()
            table_cpu[b, kernels.PAGES] = pointers[b].data_ptr()
        table = table_cpu.to(device)
        q = torch.randn(batch * width, heads * group, kd, device=device, dtype=dtype)
        k = torch.randn(batch * width, heads, kd, device=device, dtype=dtype)
        v = torch.randn(batch * width, heads, vd, device=device, dtype=dtype)
        output = torch.empty(
            batch * width, heads * group, vd, device=device, dtype=dtype
        )
        legacy_splits = min(8, triton.cdiv((slots + 1) * block + 1, 256))
        partial = torch.empty(
            batch * width, heads * group, legacy_splits, vd + 2, device=device
        )
        tl_dtype = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

        def publish(fused):
            return kernels.publish_plan[(batch, heads)](
                table,
                ids,
                ages,
                HEADS=heads,
                SLOTS=slots,
                N=triton.next_power_of_2(slots),
                AttentionPages=w.pages if fused else None,
                Selected=selected,
                BLOCK=block,
                STORAGE=storage,
                DTYPE=tl_dtype,
            )

        def legacy_attention():
            kernel = kernels.attention_partial[(batch, heads * group, legacy_splits)](
                table,
                q,
                k,
                v,
                selected,
                partial,
                HEADS=heads,
                GROUP=group,
                KD=kd,
                VD=vd,
                STORAGE=storage,
                SLOTS=slots,
                BLOCK=block,
                WIDTH=width,
                SCALE=kd**-0.5,
                CAP=args.softcap,
                SPLITS=legacy_splits,
                PER_SPLIT=triton.cdiv((slots + 1) * block + 1, legacy_splits * 32) * 32,
                DK=triton.next_power_of_2(kd),
                DV=triton.next_power_of_2(vd),
                TILE=32,
                num_warps=8 if max(kd, vd) > 128 else 4,
            )
            kernels.attention_reduce[(batch * width, heads * group)](
                partial,
                output,
                QHEADS=heads * group,
                VD=vd,
                SPLITS=legacy_splits,
                NS=triton.next_power_of_2(legacy_splits),
                DV=triton.next_power_of_2(vd),
            )
            return kernel

        legacy_attention()
        expected = output.clone()
        default = attention_geometry(
            batch,
            heads,
            group,
            width,
            slots,
            block,
            props.multi_processor_count,
            kd,
            vd,
        )
        geometries = [default]
        if args.head_tiles or args.splits or args.tiles:
            geometries += [
                AttentionGeometry(
                    h, s, triton.cdiv(slots * block, s * t), t, default.warps
                )
                for h, s, t in itertools.product(
                    args.head_tiles or [default.head_tile],
                    args.splits or [default.splits],
                    args.tiles or [default.tile],
                )
            ]
        variants = [("legacy", None)] + [
            ("auto" if g == default else "sweep", g) for g in dict.fromkeys(geometries)
        ]
        for name, geometry in variants:

            def attention(geometry=geometry):
                if geometry is None:
                    return legacy_attention()
                return w.forward(
                    q,
                    k,
                    v,
                    output,
                    batch=batch,
                    width=width,
                    storage=storage,
                    scale=kd**-0.5,
                    cap=args.softcap,
                    geometry=geometry,
                )

            def combined(geometry=geometry, attention=attention):
                publish(geometry is not None)
                return attention()

            compiled = combined()
            torch.testing.assert_close(
                output,
                expected,
                atol=1e-2 if dtype == torch.bfloat16 else 2e-3,
                rtol=1e-2 if dtype == torch.bfloat16 else 2e-3,
            )
            unique_bytes = (
                live
                * heads
                * ((slots + 1) * block + width - 1)
                * (kd + vd)
                * dtype.itemsize
            )
            result = {
                "batch": batch,
                "live_batch": live,
                "kernel": name,
                "geometry": asdict(geometry) if geometry else None,
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
                "max_abs_error_vs_legacy": (output.float() - expected.float())
                .abs()
                .max()
                .item(),
                "useful_kv_bytes": unique_bytes,
                "metadata": measure(
                    lambda geometry=geometry: publish(geometry is not None)
                ),
                "attention_and_reduce": measure(attention),
                "metadata_attention_reduce": measure(combined),
            }
            result["useful_GBps"] = (
                unique_bytes / result["attention_and_reduce"]["p50_ms"] / 1e6
            )
            report["results"].append(result)
            print(json.dumps(result), flush=True)

    for batch in args.batch:
        run_batch(batch)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
