"""Run or summarize the GPU4/5 CUDA enumeration reversal control."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from sglang.benchmark.hisparse_h2d_device_order import (
    HOST_BYTES,
    METHODS,
    compact_report,
    compare_reports,
    configuration,
    local_host,
    pair_compact,
    report_status,
    sequence,
)
from sglang.benchmark.hisparse_h2d_linear import (
    EDGE_BYTES,
    PAYLOAD,
    check_edge_case,
    loaded_linear_artifact,
)


def acquire(observer, graph, trial, device, method, position):
    from sglang.benchmark.hisparse_h2d_rx import chunk_metrics

    before = observer.sample(device)
    graph.run(2.0)
    chunks = graph.run(2.5)
    row = dict(
        trial=trial,
        device=device,
        method=method,
        position=position,
        sampler_enabled=False,
        bytes_per_graph=graph.case.nbytes * graph.case.banks,
        chunks=chunks,
        states=[before, observer.sample(device)],
    )
    row["metrics"] = chunk_metrics(row, 0.3, require_steady=True)
    return row


def run(cli):
    config = configuration(map(int, cli.physical_order.split(",")))
    report = dict(
        success=False,
        pid=os.getpid(),
        config=config,
        actual_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allocation_order=[],
        edge_checks=[],
        chunk_graphs=[],
        bindings=[],
        windows=[],
    )
    observer, host, registered = None, None, False
    try:
        import torch

        report["cuda_was_initialized"] = torch.cuda.is_initialized()
        if report["cuda_was_initialized"]:
            raise RuntimeError(
                "CUDA was initialized before the explicit CUDA0 allocation"
            )
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 2
            or torch.version.hip is not None
        ):
            raise RuntimeError("Expose exactly physical GPU4/5")
        if report["actual_visible_devices"] != config["visible_devices"]:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES differs from expected physical order"
            )
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        first = torch.empty(1, device="cuda:0")
        torch.cuda.synchronize(0)
        report["allocation_order"].append(0)
        # Keep benchmark and allocator imports after the first explicit CUDA0
        # allocation: service import side effects must not precede this control.
        from bench_h2d_linear import ContiguousCase
        from bench_h2d_pair import CopyCase, Observer, vector_provenance
        from bench_h2d_rx import SustainedGraph
        from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot

        observer = Observer(torch, [0, 1])
        report["devices"] = observer.metadata
        if [observer.metadata[d]["physical_index"] for d in (0, 1)] != config[
            "physical_order"
        ]:
            raise RuntimeError(
                "Actual UUID/physical mapping differs from expected order"
            )
        with torch.cuda.device(1):
            second = torch.empty(1, device="cuda:1")
            torch.cuda.synchronize(1)
        report["allocation_order"].append(1)
        # Keep these allocations alive; merely selecting a device is not the
        # initialization control claimed by this experiment.
        report["initial_allocation_pointers"] = [first.data_ptr(), second.data_ptr()]
        report["runtime"] = dict(
            python=sys.executable,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        )
        report["code"] = vector_provenance()
        root = Path(__file__).resolve().parents[2]
        for name in (
            "scripts/pd_test/bench_h2d_device_order.py",
            "scripts/pd_test/bench_h2d_linear.py",
            "scripts/pd_test/bench_h2d_rx.py",
            "python/sglang/benchmark/hisparse_h2d_device_order.py",
            "python/sglang/benchmark/hisparse_h2d_linear.py",
            "python/sglang/benchmark/hisparse_h2d_rx.py",
            "python/sglang/kernels/jit/csrc/hisparse_linear_probe.cuh",
        ):
            report["code"]["sha256"][name] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
        torch.cuda.set_device(config["register_device"])
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            _cuda_host_unregister,
        )

        host = HostTensorAllocator().allocate((HOST_BYTES,), torch.uint8, "cpu")
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(0),
            out=host,
        )
        torch.cuda.set_device(config["register_device"])
        rc = int(torch.cuda.cudart().cudaHostRegister(host.data_ptr(), host.numel(), 1))
        report["registration"] = dict(
            device=torch.cuda.current_device(),
            physical=observer.metadata[torch.cuda.current_device()]["physical_index"],
            flags=1,
            rc=rc,
        )
        if rc != 0:
            raise RuntimeError(f"Portable registration failed: {rc}")
        registered = True
        report["host_snapshot"] = process_snapshot(
            buffers=[
                dict(name="shared_host", address=host.data_ptr(), nbytes=host.numel())
            ]
        )
        if not local_host(report["host_snapshot"]):
            raise RuntimeError("Require fully mapped 4KiB host pages bound to NUMA1")
        inputs = [host.narrow(0, b * PAYLOAD, PAYLOAD) for b in range(64)]
        graphs = {}
        for device in (0, 1):
            report["active_case"] = dict(phase="correctness", device=device)
            for size in EDGE_BYTES:
                for source in ("host", "hbm"):
                    report["edge_checks"].append(
                        check_edge_case(torch, host, device, 4, size, source)
                    )
            cases = dict(
                linear4=ContiguousCase(torch, device, "linear4", inputs),
                ce_bulk=CopyCase(
                    torch,
                    host,
                    device,
                    "bulk-ce-64MiB",
                    "ce",
                    64 * 2**20,
                    7,
                    SimpleNamespace(source_layout_control=False, pcie_rx_probe=True),
                ),
            )
            cases["ce_bulk"].probe_method = "ce_bulk"
            report["bindings"].append(
                dict(
                    device=device,
                    checks=dict(
                        shared_host=all(
                            t.untyped_storage().data_ptr() == host.data_ptr()
                            for t in cases["linear4"].inputs
                        )
                        and cases["ce_bulk"].host is host,
                        same_inputs=[t.data_ptr() for t in cases["linear4"].inputs]
                        == [t.data_ptr() for t in inputs],
                        aligned_128=all(t.data_ptr() % 128 == 0 for t in inputs)
                        and cases["linear4"].output.data_ptr() % 128 == 0,
                        pinned=host.is_pinned(),
                    ),
                )
            )
            for method in METHODS:
                report["active_case"] = dict(
                    phase="capture", device=device, method=method
                )
                graph = SustainedGraph(
                    torch,
                    cases[method],
                    100,
                    waves=(
                        config["linear_waves"]
                        if method == "linear4"
                        else config["bulk_waves"]
                    ),
                )
                graphs[device, method] = graph
                report["chunk_graphs"].append(graph.metadata)
        report["jit_artifact"] = loaded_linear_artifact()
        preflight = report_status(report)
        if not all(
            preflight[k]
            for k in ("identity", "correctness", "graphs", "host", "bindings", "jit")
        ):
            raise RuntimeError(f"Pre-measurement validation failed: {preflight}")
        for trial, device, method, position in sequence():
            report["active_case"] = dict(
                phase="measurement", device=device, method=method, trial=trial
            )
            with torch.cuda.device(device):
                row = acquire(
                    observer, graphs[device, method], trial, device, method, position
                )
            report["windows"].append(row)
            print(f"WINDOW {report['active_case']} {row['metrics']}", flush=True)
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if registered:
            try:
                for device in (0, 1):
                    torch.cuda.synchronize(device)
                with torch.cuda.device(config["register_device"]):
                    _cuda_host_unregister(host)
            except Exception as exc:
                report["success"] = False
                report["cleanup_error"] = str(exc)
        if observer is not None:
            try:
                observer.nv.nvmlShutdown()
            except Exception as exc:
                report["success"] = False
                report["observer_cleanup_error"] = str(exc)
        (cli.output_dir / "order.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        compact = compact_report(report)
        (cli.output_dir / "order-compact.txt").write_text(compact)
        print(compact, flush=True)
    return 0 if report_status(report)["ready"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--physical-order", choices=("4,5", "5,4"), required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--normal", type=Path, required=True)
    summary.add_argument("--reversed", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (cli.output_dir / name).exists() for name in ("order.json", "order-compact.txt")
    ):
        parser.error("Use a fresh output directory")
    if cli.command == "run":
        return run(cli)
    normal, reverse = (json.loads(p.read_text()) for p in (cli.normal, cli.reversed))
    result = compare_reports(normal, reverse)
    (cli.output_dir / "order.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    compact = pair_compact(normal, reverse, result)
    (cli.output_dir / "order-compact.txt").write_text(compact)
    print(compact)
    return 0 if result["results_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
