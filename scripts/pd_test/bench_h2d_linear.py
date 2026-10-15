"""Same-payload linear SM CTA4/CTA8 baseline alongside sparse SM and CE."""

import argparse
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_pair import CopyCase, Observer, make_plan, vector_provenance
from bench_h2d_rx import SustainedGraph, counter_reader
from sglang.benchmark.hisparse_h2d_diagnostics import (
    PAGE_BYTES,
    copy_geometry,
    process_snapshot,
)
from sglang.benchmark.hisparse_h2d_grid_perf import loaded_grid_artifact
from sglang.benchmark.hisparse_h2d_linear import (
    EDGE_BYTES,
    METHODS,
    PAYLOAD,
    check_edge_case,
    compact_report,
    copy_linear,
    loaded_linear_artifact,
    method_order,
    validate_bytes,
)
from sglang.benchmark.hisparse_h2d_rx import (
    RxSampler,
    ce_calibration,
    chunk_metrics,
    window_metrics,
)


class ContiguousCase:
    def __init__(self, torch, device, method, inputs, shared=None):
        self.device, self.method, self.probe_method = device, method, method
        self.nbytes, self.banks = PAYLOAD, len(inputs)
        self.num_blocks = {"linear4": 4, "linear8": 8}.get(method)
        self.inputs = inputs
        with torch.cuda.device(device):
            self.rx_output = (
                shared.rx_output
                if shared
                else torch.empty(PAYLOAD + 256, dtype=torch.uint8, device="cuda")
            )
            self.output = (
                shared.output if shared else self.rx_output[128 : 128 + PAYLOAD]
            )
            if method == "ce_same":
                self.functions = [
                    lambda bank=b: self.output.copy_(inputs[bank], non_blocking=True)
                    for b in range(self.banks)
                ]
            else:
                self.functions = [
                    lambda bank=b: copy_linear(
                        inputs[bank], self.output, self.num_blocks
                    )
                    for b in range(self.banks)
                ]
            for bank in (0, self.banks - 1):
                self.rx_reset()
                self.functions[bank]()
                torch.cuda.synchronize()
                self.rx_validate(bank)
        self.torch = torch

    def rx_reset(self):
        self.rx_output.fill_(165)

    def rx_validate(self, bank):
        # All guarded bytes, not just a checksum or the endpoints.
        import torch

        validate_bytes(torch, self.rx_output, self.inputs[bank], PAYLOAD, prefix=128)


def binding_record(device, cases, graphs):
    selected = [cases[m] for m in ("linear4", "linear8", "ce_same")]
    root = selected[0]
    checks = dict(
        same_sources=all(
            [v.data_ptr() for v in c.inputs] == [v.data_ptr() for v in root.inputs]
            for c in selected
        ),
        same_target=all(
            c.output.data_ptr() == root.output.data_ptr() for c in selected
        ),
        same_guard=all(
            c.rx_output.data_ptr() == root.rx_output.data_ptr() for c in selected
        ),
        aligned_128=all(
            c.output.data_ptr() % 128 == 0
            and all(v.data_ptr() % 128 == 0 for v in c.inputs)
            for c in selected
        ),
        same_bytes_banks=all((c.nbytes, c.banks) == (PAYLOAD, 64) for c in selected),
        only_linear_grid_changes=cases["linear4"].num_blocks == 4
        and cases["linear8"].num_blocks == 8,
        same_chunk_work=len({graphs[m].waves for m in METHODS if m != "ce_bulk"}) == 1,
    )
    return dict(device=device, valid=all(checks.values()), checks=checks)


def acquire(torch, observer, graph, method, device, trial, position, sampled, args):
    reader = counter_reader(observer, device, rx_only=True)
    idle = RxSampler(reader, 0.05).start() if sampled else None
    if idle is not None:
        try:
            threading.Event().wait(0.7)
        finally:
            idle.stop()
    before = observer.sample(device)
    graph.run(2.0)
    sampler = RxSampler(reader, 0.05).start() if sampled else None
    try:
        chunks = graph.run(2.5)
    finally:
        if sampler is not None:
            sampler.stop()
    after = observer.sample(device)
    case = graph.case
    row = dict(
        device=device,
        method=method,
        trial=trial,
        position=position,
        sampler_enabled=sampled,
        bytes_per_graph=case.nbytes * case.banks,
        chunks=chunks,
        samples=sampler.rows if sampler else [],
        idle_samples=idle.rows if idle else [],
        states=[before, after],
    )
    row["metrics"] = (
        window_metrics(row, 0.05, 0.3, require_steady=True)
        if sampled
        else chunk_metrics(row, 0.3, require_steady=True)
    )
    if method == "ce_bulk" and sampled:
        row["calibration"] = ce_calibration(row["metrics"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "main"))
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    target = cli.output_dir / "linear.json"
    if target.exists() or (cli.output_dir / "linear-compact.txt").exists():
        parser.error("Use a fresh output directory")
    args = SimpleNamespace(
        batch=32,
        misses=[100],
        source_mib=512,
        max_banks=64,
        plan_stride=3072,
        buffer_slots=3072,
        source_layout_control=False,
        pcie_rx_probe=True,
        seed=0,
    )
    report = dict(
        success=False,
        config=dict(
            mode=cli.mode,
            order=[0, 1, 1, 0] if cli.mode == "main" else [],
            payload=PAYLOAD,
            banks=64,
            block_size=1024,
            sample_ms=50,
            duration_s=2.5,
            trim_s=0.3,
            register_device=1,
            register_flags=1,
        ),
        edge_checks=[],
        chunk_graphs=[],
        bindings=[],
        windows=[],
    )
    host, registered, observer = None, False, None
    try:
        import torch

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() < 2
            or torch.version.hip is not None
        ):
            raise RuntimeError("Expose physical GPU4/5 as CUDA0/1")
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            _cuda_host_unregister,
        )

        torch.set_num_threads(1)
        observer = Observer(torch, [0, 1])
        report["devices"] = observer.metadata
        if [observer.metadata[d]["physical_index"] for d in (0, 1)] != [4, 5]:
            raise RuntimeError("Expected physical GPU4/5; allocation mapping differs")
        if cli.mode == "main":
            report["counter_probe"] = [
                dict(device=d, **counter_reader(observer, d, rx_only=True)())
                for d in (0, 1)
            ]
        report["code"] = vector_provenance()
        root = Path(__file__).resolve().parents[2]
        for name in (
            "scripts/pd_test/bench_h2d_linear.py",
            "scripts/pd_test/bench_h2d_rx.py",
            "python/sglang/benchmark/hisparse_h2d_linear.py",
            "python/sglang/benchmark/hisparse_h2d_rx.py",
            "python/sglang/kernels/jit/csrc/hisparse_linear_probe.cuh",
        ):
            report["code"]["sha256"][name] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
        geometry = copy_geometry(32, 100, 512 * 2**20, 64)
        torch.cuda.set_device(1)
        host = HostTensorAllocator().allocate(
            (geometry["host_pages"], PAGE_BYTES), torch.uint8, "cpu"
        )
        rc = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), host.numel(), 1)
        if int(rc) != 0:
            raise RuntimeError(f"Portable registration failed: {rc}")
        registered = True
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
        inputs = [host.view(-1).narrow(0, b * PAYLOAD, PAYLOAD) for b in range(64)]
        plan = make_plan(torch, args, geometry, 0, 100)
        cases, graphs = {}, {}
        for device in (0, 1):
            for blocks in (4, 8):
                for n in EDGE_BYTES:
                    for source in ("host", "hbm"):
                        report["active_case"] = dict(
                            phase="edge",
                            device=device,
                            blocks=blocks,
                            size=n,
                            source=source,
                        )
                        report["edge_checks"].append(
                            check_edge_case(torch, host, device, blocks, n, source)
                        )
            group = {}
            group["gather4"] = CopyCase(
                torch, host, device, "B32-miss100", "sm", PAYLOAD, 64, args, plan, 100
            )
            group["gather4"].probe_method = "gather4"
            group["gather4"].num_blocks = 4
            group["linear4"] = ContiguousCase(torch, device, "linear4", inputs)
            group["linear8"] = ContiguousCase(
                torch, device, "linear8", inputs, group["linear4"]
            )
            group["ce_same"] = ContiguousCase(
                torch, device, "ce_same", inputs, group["linear4"]
            )
            group["ce_bulk"] = CopyCase(
                torch, host, device, "bulk-ce-64MiB", "ce", 64 * 2**20, 7, args
            )
            group["ce_bulk"].probe_method = "ce_bulk"
            captured = {}
            for method in METHODS:
                report["active_case"] = dict(
                    phase="chunk_capture", device=device, method=method
                )
                captured[method] = SustainedGraph(
                    torch,
                    group[method],
                    100,
                    waves=(
                        None
                        if method in ("gather4", "ce_bulk")
                        else captured["gather4"].waves
                    ),
                )
                report["chunk_graphs"].append(captured[method].metadata)
            report["active_case"] = dict(phase="bindings", device=device)
            binding = binding_record(device, group, captured)
            report["bindings"].append(binding)
            if not binding["valid"]:
                raise ValueError(f"Source/target/chunk bindings failed: {binding}")
            cases[device], graphs[device] = group, captured
        report["jit_artifacts"] = [loaded_grid_artifact(), loaded_linear_artifact()]
        if cli.mode == "main":
            local = {0: 0, 1: 0}
            for trial, device in enumerate(report["config"]["order"]):
                for position, method in enumerate(method_order(local[device])):
                    report["active_case"] = dict(
                        phase="measurement", device=device, method=method, trial=trial
                    )
                    with torch.cuda.device(device):
                        row = acquire(
                            torch,
                            observer,
                            graphs[device][method],
                            method,
                            device,
                            trial,
                            position,
                            local[device] % 2 == 1,
                            args,
                        )
                    report["windows"].append(row)
                    print(
                        f"Completed d{device} {method} trial={trial}: {row['metrics']}",
                        flush=True,
                    )
                local[device] += 1
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            try:
                compact = compact_report(report)
            except Exception as exc:
                compact = f"results_valid=False\nAcquisition error: {report.get('error')}\nSummary error: {type(exc).__name__}: {exc}\n"
            (cli.output_dir / "linear-compact.txt").write_text(compact)
            print(compact, flush=True)
        finally:
            if registered:
                for device in (0, 1):
                    torch.cuda.synchronize(device)
                _cuda_host_unregister(host)
            if observer is not None:
                observer.nv.nvmlShutdown()
    return 0 if report["success"] and "results_valid=True" in compact else 1


if __name__ == "__main__":
    raise SystemExit(main())
