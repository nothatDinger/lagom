"""GPU4 only: same-source/target 64MiB continuous SM and CE copies."""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

from sglang.benchmark.hisparse_h2d_bulk_compare import (
    CONFIG,
    COPY_BYTES,
    ORDER,
    compact_report,
    report_status,
)
from sglang.benchmark.hisparse_h2d_device_order import HOST_BYTES, local_host
from sglang.benchmark.hisparse_h2d_linear import loaded_linear_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if any((cli.output_dir / n).exists() for n in ("bulk.json", "bulk-compact.txt")):
        parser.error("Use a fresh output directory")
    report = dict(
        success=False,
        config=CONFIG,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        correctness=[],
        chunk_graphs=[],
        windows=[],
    )
    observer, host, registered = None, None, False
    try:
        import torch

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.version.hip is not None
        ):
            raise RuntimeError("Expose only physical GPU4 as CUDA0")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        from bench_h2d_device_order import acquire
        from bench_h2d_linear import ContiguousCase
        from bench_h2d_pair import Observer, vector_provenance
        from bench_h2d_rx import SustainedGraph
        from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            _cuda_host_unregister,
        )

        observer = Observer(torch, [0])
        report["devices"] = observer.metadata
        if (
            observer.metadata[0]["physical_index"] != 4
            or report["visible_devices"] != "4"
        ):
            raise RuntimeError("Actual physical GPU mapping differs from GPU4")
        report["code"] = vector_provenance()
        root = Path(__file__).resolve().parents[2]
        for n in (
            "scripts/pd_test/bench_h2d_bulk_compare.py",
            "scripts/pd_test/bench_h2d_device_order.py",
            "scripts/pd_test/bench_h2d_linear.py",
            "scripts/pd_test/bench_h2d_rx.py",
            "python/sglang/benchmark/hisparse_h2d_bulk_compare.py",
            "python/sglang/benchmark/hisparse_h2d_rx.py",
            "python/sglang/benchmark/hisparse_h2d_linear.py",
            "python/sglang/kernels/jit/csrc/hisparse_linear_probe.cuh",
        ):
            report["code"]["sha256"][n] = hashlib.sha256(
                (root / n).read_bytes()
            ).hexdigest()
        host = HostTensorAllocator().allocate((HOST_BYTES,), torch.uint8, "cpu")
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(0),
            out=host,
        )
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
            raise RuntimeError(
                "Require fully mapped 512MiB host pool on NUMA1, bind:1, 4KiB pages"
            )
        inputs = [host.narrow(0, b * COPY_BYTES, COPY_BYTES) for b in range(7)]
        report["active_case"] = dict(phase="construct")
        sm = ContiguousCase(torch, 0, "linear4", inputs, nbytes=COPY_BYTES)
        ce = ContiguousCase(torch, 0, "ce_same", inputs, shared=sm, nbytes=COPY_BYTES)
        cases = dict(sm64=sm, ce64=ce)
        graphs = {}
        for method, case in cases.items():
            case.probe_method = method
            for bank in range(7):
                report["active_case"] = dict(
                    phase="eager_check", method=method, bank=bank
                )
                case.rx_reset()
                case.functions[bank]()
                torch.cuda.synchronize(0)
                case.rx_validate(bank)
                report["correctness"].append(
                    dict(method=method, bank=bank, all_bytes=True, guards=True)
                )
            report["active_case"] = dict(phase="capture", method=method)
            graph = SustainedGraph(torch, case, 100, waves=12)
            graphs[method] = graph
            report["chunk_graphs"].append(graph.metadata)
        report["binding"] = dict(
            same_sources=[v.data_ptr() for v in sm.inputs]
            == [v.data_ptr() for v in ce.inputs]
            == [v.data_ptr() for v in inputs],
            same_target=sm.output.data_ptr() == ce.output.data_ptr(),
            same_guard=sm.rx_output.data_ptr() == ce.rx_output.data_ptr(),
            same_bytes_banks=all(
                (c.nbytes, c.banks) == (COPY_BYTES, 7) for c in cases.values()
            ),
            same_chunk_work=all(g.waves == 12 for g in graphs.values()),
            aligned_128=sm.output.data_ptr() % 128 == 0
            and all(v.data_ptr() % 128 == 0 for v in inputs),
            host_pinned=host.is_pinned(),
        )
        report["jit_artifact"] = loaded_linear_artifact()
        preflight = report_status(report)
        if not all(
            preflight[k]
            for k in ("identity", "correctness", "graphs", "host", "bindings", "jit")
        ):
            raise RuntimeError(f"Pre-measurement checks failed: {preflight}")
        for trial, method in enumerate(ORDER):
            report["active_case"] = dict(
                phase="measurement", method=method, trial=trial
            )
            with torch.cuda.device(0):
                row = acquire(observer, graphs[method], trial, 0, method, 0)
            report["windows"].append(row)
            print(f"WINDOW {method} trial={trial} {row['metrics']}", flush=True)
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            if registered:
                torch.cuda.synchronize(0)
                _cuda_host_unregister(host)
        except Exception as exc:
            report["success"] = False
            report["cleanup_error"] = str(exc)
        try:
            if observer is not None:
                observer.nv.nvmlShutdown()
        except Exception as exc:
            report["success"] = False
            report["cleanup_error"] = str(exc)
        (cli.output_dir / "bulk.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        compact = compact_report(report)
        (cli.output_dir / "bulk-compact.txt").write_text(compact)
        print(compact, flush=True)
    return 0 if report_status(report)["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
