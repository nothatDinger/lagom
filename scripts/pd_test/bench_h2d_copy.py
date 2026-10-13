"""Isolated DSv4 production SM copy / contiguous CE copy, with rotating sources.

No model, planner, recorder or CPU submission time is included in copy timing.
CUDA timing events are explicit nodes *inside* each captured graph. Launch with
one Python process, or torch.distributed.run for a single-host four-GPU test.
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import time
from pathlib import Path

from sglang.benchmark.hisparse_h2d_diagnostics import (
    ITEM_BYTES,
    PAGE_BYTES,
    PAGE_TOKENS,
    VALUE_BYTES,
    copy_geometry,
    process_snapshot,
    timing_summary,
)


def gpu_inventory():
    try:
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return dict(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def validate_sm(torch, host, device, source, destination=None):
    """Check every value and scale byte of a bank, using an independent CPU gather."""
    actual = device.cpu().view(-1)
    flat = host.view(-1)
    source = source.reshape(-1).cpu()
    destination = (
        torch.arange(len(source))
        if destination is None
        else destination.reshape(-1).cpu()
    ).to(torch.int64)
    for begin in range(0, len(source), 1024):
        loc = source[begin : begin + 1024]
        page, offset = loc // PAGE_TOKENS, loc % PAGE_TOKENS
        vaddr = page[:, None] * PAGE_BYTES + offset[:, None] * VALUE_BYTES
        saddr = (
            page[:, None] * PAGE_BYTES + PAGE_TOKENS * VALUE_BYTES + offset[:, None] * 8
        )
        expected_v = flat[vaddr + torch.arange(VALUE_BYTES)]
        expected_s = flat[saddr + torch.arange(8)]
        dst = destination[begin : begin + len(loc)]
        dvaddr = (dst // PAGE_TOKENS)[:, None] * PAGE_BYTES + (dst % PAGE_TOKENS)[
            :, None
        ] * VALUE_BYTES
        dsaddr = (
            (dst // PAGE_TOKENS)[:, None] * PAGE_BYTES
            + PAGE_TOKENS * VALUE_BYTES
            + (dst % PAGE_TOKENS)[:, None] * 8
        )
        if not torch.equal(actual[dvaddr + torch.arange(VALUE_BYTES)], expected_v):
            raise RuntimeError(f"DSv4 value mismatch at chunk {begin}")
        if not torch.equal(actual[dsaddr + torch.arange(8)], expected_s):
            raise RuntimeError(f"DSv4 scale mismatch at chunk {begin}")


def capture_wave(torch, functions, *, stream=None):
    # Default (external=False) event handling is only a graph dependency.
    # external=True emits real event-record nodes, allowing elapsed_time().
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in functions
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for fn, (start, end) in zip(functions, events):
            start.record()
            fn()
            end.record()
    return graph, events


def run_case(
    torch, dist, args, report, label, nbytes, banks, functions, validate, reset
):
    captured = {}
    checks = {}
    for name, fns in functions.items():
        checked_banks = sorted({0, banks - 1})
        for bank in checked_banks:
            reset(name)
            fns[bank]()  # Also compile / initialize the production JIT outside capture.
            torch.cuda.synchronize()
            validate(name, bank)
        checks[name] = dict(
            verified_banks=checked_banks, all_payload_bytes_checked=True
        )
        captured[name] = capture_wave(torch, fns)
    torch.cuda.synchronize()
    for graph, _ in captured.values():
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()

    samples = {name: [] for name in functions}
    waves = {name: [] for name in functions}
    order_rng = random.Random(args.seed)  # Same test order on all worker ranks.
    for repeat in range(args.rounds):
        order = list(functions)
        order_rng.shuffle(order)
        for name in order:
            if dist.is_initialized():
                dist.barrier()  # Outside CUDA graph and outside event intervals.
            graph, events = captured[name]
            submitted = time.monotonic_ns()
            graph.replay()  # A whole bank rotation, keeping each GPU busy for a wave.
            events[-1][1].synchronize()
            completed = time.monotonic_ns()
            times = [start.elapsed_time(end) * 1000 for start, end in events]
            samples[name].extend(times)
            waves[name].append(
                dict(
                    repeat=repeat,
                    host_submitted_ns=submitted,
                    host_completed_ns=completed,
                    gpu_wave_us=events[0][0].elapsed_time(events[-1][1]) * 1000,
                )
            )
    result = dict(label=label, banks=banks, payload_bytes=nbytes, results=[])
    for name in functions:
        result["results"].append(
            dict(
                method=name,
                correctness=checks[name],
                **timing_summary(samples[name], nbytes),
                raw_copy_us=samples[name],
                waves=waves[name],
            )
        )
    report["cases"].append(result)
    print(
        json.dumps(
            dict(
                rank=report["rank"],
                label=label,
                results=[
                    {k: v for k, v in r.items() if k not in ("raw_copy_us", "waves")}
                    for r in result["results"]
                ],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--misses", type=int, nargs="+", default=[100, 1272])
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--plan-stride", type=int, default=3072)
    parser.add_argument("--buffer-slots", type=int, default=3072)
    parser.add_argument("--source-mib", type=int, default=512)
    parser.add_argument("--max-banks", type=int, default=64)
    parser.add_argument("--bulk-ce-mib", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.plan_stride, args.buffer_slots) <= 0 or max(args.misses) > min(
        args.plan_stride, args.buffer_slots
    ):
        parser.error("Plan stride and per-request buffer slots must cover every miss")
    if min(args.rounds, args.warmup, args.source_mib, args.bulk_ce_mib) <= 0:
        parser.error("Rounds, warmup and sizes must be positive")
    if (
        not args.blocks
        or any(b <= 0 for b in args.blocks)
        or len(set(args.blocks)) != len(args.blocks)
    ):
        parser.error("CTA counts must be positive and distinct")
    geometries = [
        copy_geometry(args.batch, m, args.source_mib * 2**20, args.max_banks)
        for m in args.misses
    ]
    rank, local_rank, world = (
        int(os.environ.get(k, "0" if k != "WORLD_SIZE" else "1"))
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world))) != world:
        parser.error("This benchmark only supports a single host")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"rank{rank}.json"
    if target.exists():
        parser.error(f"Refusing to replace {target}; use a new output directory")

    import torch
    import torch.distributed as dist

    from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
        alloc_with_host_register,
    )

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA required")
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=10))
    props = torch.cuda.get_device_properties(local_rank)
    report = dict(
        schema_version=1,
        success=False,
        pid=os.getpid(),
        rank=rank,
        world_size=world,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device=local_rank,
        gpu=str(props),
        gpu_uuid=str(getattr(props, "uuid", "unavailable")),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        config=vars(args).copy(),
        gpu_inventory=gpu_inventory(),
        cases=[],
        notes=[
            "GBps is decimal payload bytes divided by sum of GPU copy event time.",
            "SM production copy reads pinned host memory directly; CE is a separate contiguous copy control.",
            "Timing events are inside graphs; allocation, validation, CPU submission, Gloo and planner are excluded.",
            "Disjoint banks reduce source reuse; this does not measure PCIe wire traffic or prove absence of cache hits.",
            "Concurrent workers use barriers and multi-copy waves; barriers alone do not prove identical GPU start times.",
            "Synthetic uniform miss counts and random addresses are controls, not a replay of the service plans.",
        ],
    )
    report["config"]["output_dir"] = str(args.output_dir)
    host = None
    try:
        # Exactly the default SGLang mmap + cudaHostRegister allocator, not
        # PyTorch's reusable pinned-memory pool. NUMA policy must precede mmap.
        host_pages = geometries[0]["host_pages"]
        host = alloc_with_host_register(
            (host_pages, PAGE_BYTES), torch.uint8, "cpu", True, HostTensorAllocator()
        )
        rng = torch.Generator().manual_seed(args.seed)
        torch.randint(0, 256, host.shape, dtype=torch.uint8, generator=rng, out=host)
        report["host_allocation"] = process_snapshot(
            buffers=[
                dict(
                    name="synthetic_host_kv",
                    address=host.data_ptr(),
                    nbytes=host.numel(),
                )
            ]
        )
        report["hugepage_setting"] = os.environ.get("SGLANG_HUGEPAGE_SIZE", "")
        report["l2_cache_bytes"] = getattr(props, "L2_cache_size", None)
        for case_index, (misses, geometry) in enumerate(zip(args.misses, geometries)):
            entries, banks = geometry["entries"], geometry["banks"]
            # Without replacement, across all banks. Relative addresses are
            # identical on every worker. CTA comparisons reuse the same plans.
            plan_rng = torch.Generator().manual_seed(args.seed + case_index + 1)
            active_source_cpu = torch.randperm(
                geometry["host_tokens"], generator=plan_rng
            )[: banks * entries].reshape(banks, args.batch, misses)
            source_cpu = torch.zeros(
                (banks, args.batch, args.plan_stride), dtype=torch.int64
            )
            source_cpu[:, :, :misses] = active_source_cpu
            source = source_cpu.to(device="cuda", dtype=torch.int64)
            padded_slots = (
                (args.buffer_slots + PAGE_TOKENS - 1) // PAGE_TOKENS + 1
            ) * PAGE_TOKENS
            geometry["destination_pages"] = args.batch * padded_slots // PAGE_TOKENS
            geometry["destination_bytes"] = geometry["destination_pages"] * PAGE_BYTES
            geometry["plan_stride"] = args.plan_stride
            geometry["padded_slots_per_request"] = padded_slots
            destination_cpu = torch.full(
                (args.batch, args.plan_stride), -1, dtype=torch.int32
            )
            for req in range(args.batch):
                destination_cpu[req, :misses] = (
                    torch.randperm(args.buffer_slots, generator=plan_rng)[:misses]
                    + req * padded_slots
                )
            destination = destination_cpu.to(device="cuda")
            counts = torch.full((args.batch,), misses, device="cuda", dtype=torch.int32)
            real = torch.tensor([args.batch], device="cuda", dtype=torch.int32)
            output = torch.zeros(
                (args.batch * padded_slots // PAGE_TOKENS, PAGE_BYTES),
                device="cuda",
                dtype=torch.uint8,
            )
            ce_output = torch.empty(
                geometry["payload_bytes"], device="cuda", dtype=torch.uint8
            )
            ce_sources = [
                host.view(-1).narrow(
                    0, b * geometry["payload_bytes"], geometry["payload_bytes"]
                )
                for b in range(banks)
            ]
            functions = {}
            for blocks in args.blocks:

                def make_copy(bank, blocks):
                    def run():
                        copy_cache_planned_mla(
                            miss_src=source[bank],
                            miss_dst=destination,
                            miss_count=counts,
                            num_real_reqs=real,
                            host_cache=host,
                            device_buffer=output,
                            item_size_bytes=ITEM_BYTES,
                            num_blocks=blocks,
                            block_size=1024,
                            is_dsv4_layout=True,
                        )

                    return run

                functions[f"sm-cta{blocks}"] = [
                    make_copy(b, blocks) for b in range(banks)
                ]
            functions["ce-contiguous"] = [
                lambda b=b: ce_output.copy_(ce_sources[b], non_blocking=True)
                for b in range(banks)
            ]

            def validate(name, bank):
                if name == "ce-contiguous":
                    if not torch.equal(ce_output.cpu(), ce_sources[bank]):
                        raise RuntimeError("Contiguous CE copy mismatch")
                else:
                    validate_sm(
                        torch,
                        host,
                        output,
                        active_source_cpu[bank],
                        destination_cpu[:, :misses],
                    )

            def reset(name):
                (ce_output if name == "ce-contiguous" else output).zero_()

            run_case(
                torch,
                dist,
                args,
                report,
                f"B{args.batch}-miss{misses}",
                geometry["payload_bytes"],
                banks,
                functions,
                validate,
                reset,
            )
            report["cases"][-1]["geometry"] = geometry

        bulk_bytes = args.bulk_ce_mib * 2**20
        bulk_banks = min(args.max_banks, host.numel() // bulk_bytes)
        if bulk_banks < 2:
            raise ValueError("Host source must hold at least two bulk CE banks")
        bulk_output = torch.empty(bulk_bytes, device="cuda", dtype=torch.uint8)
        bulk_sources = [
            host.view(-1).narrow(0, b * bulk_bytes, bulk_bytes)
            for b in range(bulk_banks)
        ]
        functions = {
            "ce-contiguous": [
                lambda b=b: bulk_output.copy_(bulk_sources[b], non_blocking=True)
                for b in range(bulk_banks)
            ]
        }

        def validate_bulk(_name, bank):
            if not torch.equal(bulk_output.cpu(), bulk_sources[bank]):
                raise RuntimeError("Bulk CE copy mismatch")

        run_case(
            torch,
            dist,
            args,
            report,
            f"bulk-ce-{args.bulk_ce_mib}MiB",
            bulk_bytes,
            bulk_banks,
            functions,
            validate_bulk,
            lambda _name: bulk_output.zero_(),
        )
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")
        if host is not None:
            torch.cuda.synchronize()
            _cuda_host_unregister(host)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
