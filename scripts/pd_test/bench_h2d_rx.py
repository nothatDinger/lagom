"""Sustained legacy host gather / bulk CE and NVML PCIe receive rates.

Use the existing venv and GPU4/5 allocation. No NCU, model or CUDA changes.
"""

import argparse
import hashlib
import json
import math
import threading
import time
from pathlib import Path

from bench_h2d_pair import CopyCase, Observer, Warmer, make_plan, vector_provenance
from sglang.benchmark.hisparse_h2d_diagnostics import (
    PAGE_BYTES,
    copy_geometry,
    process_snapshot,
)
from sglang.benchmark.hisparse_h2d_grid_perf import loaded_grid_artifact
from sglang.benchmark.hisparse_h2d_pair import abba_order
from sglang.benchmark.hisparse_h2d_rx import (
    RxSampler,
    ce_calibration,
    rx_compact_report,
    window_metrics,
)


def counter_reader(observer, device):
    nv, handle = observer.nv, observer.handles[device]

    def read():
        return dict(
            rx_kbps=int(
                nv.nvmlDeviceGetPcieThroughput(handle, nv.NVML_PCIE_UTIL_RX_BYTES)
            ),
            tx_kbps=int(
                nv.nvmlDeviceGetPcieThroughput(handle, nv.NVML_PCIE_UTIL_TX_BYTES)
            ),
        )

    return read


def sustained_window(torch, observer, warmer, case, method, trial, args):
    reader = counter_reader(observer, case.device)
    idle = RxSampler(reader, args.sample_ms / 1000).start()
    try:
        threading.Event().wait(0.5)
    finally:
        idle.stop()
    case.warm(torch, "compute", warmer, args)
    pilot = case.measure(torch, observer, "pilot", trial)
    repeats = max(1, math.ceil(args.chunk_ms * 1000 / pilot["gpu_wave_us"]))
    sampler = RxSampler(
        reader, args.sample_ms / 1000, lambda: observer.sample(case.device)
    ).start()
    chunks = []
    try:
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        began = time.monotonic_ns()
        while (time.monotonic_ns() - began) / 1e9 < args.duration_s:
            before = time.monotonic_ns()
            start.record()
            for _ in range(repeats):
                case.graph.replay()
            submitted = time.monotonic_ns()
            end.record()
            end.synchronize()
            after = time.monotonic_ns()
            chunks.append(
                dict(
                    begin_ns=before,
                    end_ns=after,
                    enqueue_ns=submitted - before,
                    replays=repeats,
                    gpu_us=start.elapsed_time(end) * 1000,
                )
            )
    finally:
        sampler.stop()
    window = dict(
        device=case.device,
        trial=trial,
        method=method,
        case=case.label,
        bytes_per_copy=case.nbytes,
        banks=case.banks,
        bytes_per_graph=case.nbytes * case.banks,
        chunks=chunks,
        samples=sampler.rows,
        idle_samples=idle.rows,
        states=sampler.states,
        pilot=pilot,
    )
    window["metrics"] = window_metrics(window, args.sample_ms / 1000, args.trim_s)
    if method == "ce_bulk":
        window["calibration"] = ce_calibration(window["metrics"])
    return window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=3)
    parser.add_argument("--trim-s", type=float, default=0.3)
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--chunk-ms", type=float, default=100)
    parser.set_defaults(
        batch=32,
        misses=[100],
        source_mib=512,
        max_banks=64,
        plan_stride=3072,
        buffer_slots=3072,
        bulk_ce_mib=64,
        warmup=2,
        warm_ms=250,
        seed=0,
        register_device=1,
        source_layout_control=False,
        pcie_rx_probe=True,
    )
    args = parser.parse_args()
    if (
        args.duration_s < 1.5
        or not (0.2 <= args.trim_s <= 0.5)
        or args.duration_s <= 2 * args.trim_s + 0.7
    ):
        parser.error(
            "Require duration >=1.5s with >=0.7s after 0.2..0.5s edge trimming"
        )
    if not (10 <= args.sample_ms <= 25 and 50 <= args.chunk_ms <= 150):
        parser.error("Require sample-ms=10..25 and chunk-ms=50..150")
    order = abba_order(args.devices, args.epochs)
    if args.devices != [0, 1]:
        parser.error("Expose physical GPU4/5 as CUDA logical 0/1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "rx.json"
    if target.exists() or (args.output_dir / "rx-compact.txt").exists():
        parser.error(
            "Use a new output directory; existing measurements are not replaced"
        )
    report = dict(
        success=False,
        counters_available=False,
        config={**vars(args), "output_dir": str(args.output_dir), "order": order},
        windows=[],
        correctness=[],
    )
    host, registered, observer = None, False, None
    try:
        import torch
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            _cuda_host_unregister,
        )

        if (
            not torch.cuda.is_available()
            or torch.version.hip is not None
            or torch.cuda.device_count() < 2
        ):
            raise RuntimeError("Expose two NVIDIA GPUs")
        torch.set_num_threads(1)
        observer = Observer(torch, args.devices)
        report["devices"] = observer.metadata
        report["torch"], report["cuda"] = torch.__version__, torch.version.cuda
        report["nvml_version"] = str(observer.nv.nvmlSystemGetNVMLVersion())
        report["driver_version"] = str(observer.nv.nvmlSystemGetDriverVersion())
        # Zero is valid in the idle probe. Exceptions are retained, not treated as zero.
        report["capabilities"] = []
        for device in args.devices:
            report["active_case"] = dict(device=device, phase="counter_probe")
            report["capabilities"].append(
                dict(device=device, **counter_reader(observer, device)())
            )
        report["counters_available"] = True
        report["code"] = vector_provenance()
        root = Path(__file__).resolve().parents[2]
        for name in (
            "scripts/pd_test/bench_h2d_rx.py",
            "python/sglang/benchmark/hisparse_h2d_rx.py",
        ):
            report["code"]["sha256"][name] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
        geometry = copy_geometry(32, 100, 512 * 2**20, 64)
        torch.cuda.set_device(args.register_device)
        host = HostTensorAllocator().allocate(
            (geometry["host_pages"], PAGE_BYTES), torch.uint8, "cpu"
        )
        rc = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), host.numel(), 1)
        if int(rc) != 0:
            raise RuntimeError(f"Portable cudaHostRegister failed: {rc}")
        registered = True
        report["register_flags"] = 1
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(0),
            out=host,
        )
        report["host_snapshot"] = process_snapshot(
            buffers=[
                dict(
                    name="shared_host_kv", address=host.data_ptr(), nbytes=host.numel()
                )
            ]
        )
        plan = make_plan(torch, args, geometry, 0, 100)
        cases = {}
        for device in args.devices:
            cases[(device, "sm")] = CopyCase(
                torch,
                host,
                device,
                "B32-miss100",
                "sm",
                geometry["payload_bytes"],
                geometry["banks"],
                args,
                plan,
                100,
            )
            cases[(device, "ce_bulk")] = CopyCase(
                torch,
                host,
                device,
                "bulk-ce-64MiB",
                "ce",
                64 * 2**20,
                geometry["host_bytes"] // (64 * 2**20),
                args,
            )
        for (device, method), case in cases.items():
            if method == "sm" and (
                case.sm_kwargs["num_blocks"] != 4
                or case.sm_kwargs["block_size"] != 1024
                or case.sm_kwargs["load_policy"] != "legacy"
            ):
                raise RuntimeError("Unexpected SM launch/policy")
            report["correctness"].append(
                dict(
                    device=device,
                    method=method,
                    payload=True,
                    graph=case.graph_verified,
                    untouched=method == "sm",
                    verified_banks=case.verified_banks,
                )
            )
        report["jit_artifacts"] = [loaded_grid_artifact()]
        warmers = {d: Warmer(torch, d) for d in args.devices}
        local_trials = {d: 0 for d in args.devices}
        for trial, device in enumerate(order):
            methods = (
                ("ce_bulk", "sm")
                if local_trials[device] % 2 == 0
                else ("sm", "ce_bulk")
            )
            for method in methods:
                report["active_case"] = dict(device=device, trial=trial, method=method)
                with torch.cuda.device(device):
                    window = sustained_window(
                        torch,
                        observer,
                        warmers[device],
                        cases[(device, method)],
                        method,
                        trial,
                        args,
                    )
                report["windows"].append(window)
                print(
                    f"Completed d{device} trial={trial} {method}: {window['metrics']}",
                    flush=True,
                )
                if not window["metrics"]["valid"]:
                    raise RuntimeError(f"Invalid sustained window: {method}")
                if method == "ce_bulk" and not window["calibration"]["valid"]:
                    raise RuntimeError(
                        f"CE counter calibration failed: {window['calibration']}"
                    )
            local_trials[device] += 1
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            try:
                compact = rx_compact_report(report)
            except Exception as exc:
                compact = f"results_valid=False\nAcquisition error: {report.get('error')}\nSummary error: {type(exc).__name__}: {exc}\n"
            (args.output_dir / "rx-compact.txt").write_text(compact)
            print(compact, flush=True)
        finally:
            if registered:
                for device in args.devices:
                    torch.cuda.synchronize(device)
                _cuda_host_unregister(host)
            if observer is not None:
                observer.nv.nvmlShutdown()
    return 0 if report["success"] and "results_valid=True" in compact else 1


if __name__ == "__main__":
    raise SystemExit(main())
