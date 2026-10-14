"""Dependent host/HBM reads on two GPUs, using one portable registered pool.

This diagnostic is separate from the production copy kernel. It reports logical
loads per second and dependent-chain cost, never PCIe GB/s or hardware queue depth.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

from bench_h2d_copy import capture_wave
from bench_h2d_pair import Observer, Warmer
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_h2d_pair import abba_order
from sglang.benchmark.hisparse_read_probe import (
    check_wave_result,
    compact_report,
    make_chains,
    time_stats,
)


def grid(text):
    try:
        blocks, warps = map(int, text.split("x"))
        if blocks not in (1, 4) or warps not in (1, 4, 16, 32):
            raise ValueError()
        return blocks, warps
    except ValueError:
        raise argparse.ArgumentTypeError("Use BxW: B in {1,4}, W in {1,4,16,32}")


def capture_on_device(torch, functions, device):
    """Never reuse torch.cuda.graph's process-wide default capture stream."""
    expected = torch.device("cuda", device)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        if stream.device != expected:
            raise RuntimeError(
                f"Capture stream is on {stream.device}, expected {expected}"
            )

        def checked_call(fn):
            # Check before the custom FFI launch, not after a possible bad access.
            if (
                torch.cuda.current_device() != device
                or torch.cuda.current_stream().device != expected
            ):
                raise RuntimeError(f"Capture launch context is not {expected}")
            fn()

        graph, events = capture_wave(
            torch,
            [lambda fn=fn: checked_call(fn) for fn in functions],
            stream=stream,
        )
        event_devices = {str(event.device) for pair in events for event in pair}
        if not events or event_devices != {str(expected)}:
            raise RuntimeError(
                f"Captured events are on {sorted(event_devices)}, expected {expected}"
            )
    return (
        graph,
        events,
        stream,
        dict(
            capture_device=str(stream.device),
            event_devices=sorted(event_devices),
            device_checked=True,
        ),
    )


class ReadCase:
    def __init__(
        self,
        torch,
        module,
        source,
        device,
        plan,
        nodes,
        stride,
        blocks,
        warps,
        samples,
        progress,
    ):
        self.torch, self.device = torch, device
        self.progress = progress
        progress["stage"] = "allocate"
        self.expected = plan["expected"]
        self.total_reads = len(plan["nodes"])
        # %smid values need not be dense, so validate against PTX %nsmid below.
        self.source = source
        self.starts = torch.tensor(plan["starts"], dtype=torch.int64, device=device)
        self.output = torch.full(
            (samples, len(self.starts) + 2, 4), -1, dtype=torch.int64, device=device
        )

        def run(output):
            module.hisparse_dependent_read(
                source,
                self.starts,
                output,
                nodes,
                stride // 8,
                plan["steps"],
                blocks,
                warps,
            )

        functions = [lambda out=out[1:-1]: run(out) for out in self.output]
        progress["stage"] = "eager_launch"
        functions[0]()  # Compile and exercise eager before capture.
        torch.cuda.synchronize(device)
        progress["stage"] = "eager_check"
        self.eager_check = self.check(first_only=True)
        progress["eager_passed"] = True
        progress["stage"] = "capture"
        self.graph, self.events, self.capture_stream, self.capture_check = (
            capture_on_device(torch, functions, device)
        )
        progress.update(self.capture_check)
        progress["stage"] = "graph_replay"
        self.output.fill_(-1)
        self.graph.replay()
        torch.cuda.synchronize(device)
        progress["stage"] = "graph_check"
        self.graph_check = self.check()
        progress["graph_passed"] = True

    def check(self, *, first_only=False):
        outputs = self.output[:1] if first_only else self.output
        return check_wave_result(outputs.cpu().tolist(), self.expected)

    def measure(self, warmer, observer, warm_ms):
        torch = self.torch
        self.progress["stage"] = "measure_warmup"
        warmer.run(warm_ms)
        # Warm translation state consistently; .cv still re-fetches host data.
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        self.output.fill_(-1)
        before = observer.sample(self.device)
        self.progress["stage"] = "measure_replay"
        self.graph.replay()
        self.events[-1][1].synchronize()
        after = observer.sample(self.device)
        samples = [a.elapsed_time(b) * 1000 for a, b in self.events]
        timing = time_stats(samples, self.total_reads, len(self.starts))
        self.progress["stage"] = "measure_check"
        return dict(
            event_us=samples,
            timing=timing,
            before=before,
            after=after,
            correctness=self.check(),
        )


def provenance():
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "python/sglang/benchmark/hisparse_read_probe.py",
        root / "python/sglang/kernels/jit/csrc/hisparse_read_probe.cuh",
        root / "scripts/pd_test/bench_h2d_copy.py",
        root / "scripts/pd_test/bench_h2d_pair.py",
    ]
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    return dict(
        git=run.stdout.strip(),
        sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--register-device", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=131072)
    parser.add_argument("--total-reads", type=int, default=32768)
    parser.add_argument("--strides", type=int, nargs="+", default=[128, 4096])
    parser.add_argument(
        "--grids",
        type=grid,
        nargs="+",
        default=[(1, 1), (1, 4), (1, 16), (1, 32), (4, 1), (4, 4), (4, 16), (4, 32)],
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warm-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=81173)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if len(set(args.devices)) != 2 or args.register_device not in args.devices:
        parser.error("Require two distinct devices, including the registration device")
    if not 1 <= args.total_reads <= args.nodes <= 2**30:
        parser.error("Require 1 <= total-reads <= nodes <= 2**30")
    if args.samples <= 0 or args.epochs <= 0 or args.warm_ms <= 0:
        parser.error("Positive samples/epochs/warm-ms required")
    if len(set(args.strides)) != len(args.strides) or any(
        s not in (128, 4096) for s in args.strides
    ):
        parser.error("Use distinct strides from 128 and 4096 bytes")
    if len(set(args.grids)) != len(args.grids) or any(
        args.total_reads % (b * w) for b, w in args.grids
    ):
        parser.error("Distinct grids must each divide total-reads")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process, without torchrun")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "reads.json"
    if target.exists():
        parser.error("Use a new output directory; existing results are not replaced")

    import torch
    from sglang.kernels.jit.utils import load_jit
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA required")
    if min(args.devices) < 0 or max(args.devices) >= torch.cuda.device_count():
        parser.error("Expose both selected GPUs with CUDA_VISIBLE_DEVICES=4,5")
    torch.set_num_threads(1)
    torch.cuda.set_device(args.register_device)
    observer = Observer(torch, args.devices)
    order = random.Random(args.seed).sample(range(args.nodes), args.total_reads)
    abba = abba_order(args.devices, args.epochs, args.reverse)
    shapes = [(s, b, w) for s in args.strides for b, w in args.grids]
    if args.reverse:
        shapes.reverse()
    report = dict(
        success=False,
        code=provenance(),
        host_register_flags=1,
        config={
            **vars(args),
            "output_dir": str(args.output_dir),
            "abba": abba,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hugepage_setting": os.environ.get("SGLANG_HUGEPAGE_SIZE", ""),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "load": "ld.global.cv.u64",
            "threads_per_block": 1024,
        },
        devices=observer.metadata,
        correctness=[],
        plans=[],
        waves=[],
        current_case=dict(stage="setup"),
        expected_correctness=len(shapes) * 2 * len(args.devices),
    )
    registered = False
    host = None
    try:
        host = HostTensorAllocator().allocate(
            (args.nodes * max(args.strides) // 8,), torch.int64, "cpu"
        )
        host.fill_(-1)  # First touch all pages under the caller's NUMA binding.
        rc = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), host.numel() * 8, 1)
        if int(rc) != 0:
            raise RuntimeError(f"Portable cudaHostRegister failed: {rc}")
        registered = True
        snapshot = process_snapshot(
            buffers=[
                dict(
                    name="shared_read_pool",
                    address=host.data_ptr(),
                    nbytes=host.numel() * 8,
                )
            ]
        )
        report["host_snapshot"] = snapshot
        report["host_buffers"] = snapshot["buffers"]
        module = load_jit(
            "hisparse_dependent_read",
            cuda_files=["hisparse_read_probe.cuh"],
            cuda_wrappers=[("hisparse_dependent_read", "hisparse_dependent_read")],
        )
        warmers, hbm = {}, {}
        for device in args.devices:
            with torch.cuda.device(device):
                warmers[device] = Warmer(torch, device)
                hbm[device] = torch.empty(host.shape, dtype=host.dtype, device=device)

        for index, (stride, blocks, warps) in enumerate(shapes):
            report["current_case"] = dict(
                stride=stride, blocks=blocks, warps=warps, stage="prepare_plan"
            )
            for device in args.devices:
                torch.cuda.synchronize(device)  # Finish readers before modifying host.
            plan = make_chains(order, blocks * warps)
            host.fill_(-1)
            host[torch.tensor(plan["nodes"], dtype=torch.int64) * (stride // 8)] = (
                torch.tensor(plan["successors"], dtype=torch.int64)
            )
            report["plans"].append(
                dict(
                    stride=stride,
                    blocks=blocks,
                    warps=warps,
                    steps=plan["steps"],
                    node_ids_sha256=plan["node_ids_sha256"],
                    virtual_4k_pages=len(
                        {(host.data_ptr() + n * stride) // 4096 for n in order}
                    ),
                    note="Virtual 4KiB footprint, not observed IOMMU/GPU translation page size",
                )
            )
            cases = {}
            for device in args.devices:
                report["current_case"] = dict(
                    stride=stride,
                    blocks=blocks,
                    warps=warps,
                    device=device,
                    stage="prepare_hbm_source",
                )
                with torch.cuda.device(device):
                    hbm[device].copy_(host)
                    torch.cuda.synchronize(device)
                    for source in ("host", "hbm"):
                        progress = dict(
                            stride=stride,
                            blocks=blocks,
                            warps=warps,
                            source=source,
                            device=device,
                            stage="construct",
                        )
                        report["current_case"] = progress
                        case = ReadCase(
                            torch,
                            module,
                            host if source == "host" else hbm[device],
                            device,
                            plan,
                            args.nodes,
                            stride,
                            blocks,
                            warps,
                            args.samples,
                            progress,
                        )
                        cases[source, device] = case
                        report["correctness"].append(
                            dict(
                                stride=stride,
                                blocks=blocks,
                                warps=warps,
                                source=source,
                                device=device,
                                eager=case.eager_check,
                                graph=case.graph_check,
                                capture=case.capture_check,
                            )
                        )
            sources = ("host", "hbm") if index % 2 == 0 else ("hbm", "host")
            for source in sources:
                for wave, device in enumerate(abba):
                    case = cases[source, device]
                    report["current_case"] = case.progress
                    case.progress["wave"] = wave
                    with torch.cuda.device(device):
                        measurement = case.measure(
                            warmers[device], observer, args.warm_ms
                        )
                    report["waves"].append(
                        dict(
                            stride=stride,
                            blocks=blocks,
                            warps=warps,
                            source=source,
                            device=device,
                            wave=wave,
                            **measurement,
                        )
                    )
            print(f"Finished stride={stride} grid={blocks}x{warps}", flush=True)
            cases.clear()
        report["success"] = len(report["correctness"]) == report[
            "expected_correctness"
        ] and len(report["waves"]) == len(shapes) * 2 * len(abba)
        report.pop("current_case")
    except Exception:
        report["failure_context"] = dict(report["current_case"])
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        for device in args.devices:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        if registered:
            with torch.cuda.device(args.register_device):
                _cuda_host_unregister(host)
        target.write_text(json.dumps(report, indent=2) + "\n")
        compact = compact_report(report)
        (args.output_dir / "reads-compact.txt").write_text(compact)
        print(compact)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
