"""B32/miss100: original gather, adjacent C4 gather, byte-linear SM; CTA4 only."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_linear import ContiguousCase, acquire
from bench_h2d_pair import CopyCase, Observer, make_plan, vector_provenance
from bench_h2d_rx import SustainedGraph
from sglang.benchmark.hisparse_h2d_diagnostics import PAGE_BYTES, process_snapshot
from sglang.benchmark.hisparse_h2d_grid_perf import loaded_grid_artifact
from sglang.benchmark.hisparse_h2d_kv_bridge import (
    METHODS,
    bridge_geometry,
    compact_report,
    method_order,
    prepare_bridge,
    report_status,
)
from sglang.benchmark.hisparse_h2d_linear import PAYLOAD, loaded_linear_artifact


def binding_record(torch, device, host, cases, graphs, plan, adjacent, inputs):
    original, packed = (cases[m] for m in METHODS[:2])
    a, b = original.sm_kwargs, packed.sm_kwargs
    option_values = dict(
        num_blocks=4,
        block_size=1024,
        load_policy="legacy",
        item_size_bytes=584,
        is_dsv4_layout=True,
    )
    checks = dict(
        same_host_allocation=original.source.data_ptr()
        == packed.source.data_ptr()
        == host.data_ptr(),
        same_gather_output=a["device_buffer"].data_ptr()
        == b["device_buffer"].data_ptr(),
        same_counts_and_real=all(
            a[k].data_ptr() == b[k].data_ptr() for k in ("miss_count", "num_real_reqs")
        )
        and a["miss_count"].cpu().tolist() == [100] * 32
        and a["num_real_reqs"].cpu().tolist() == [32],
        same_gather_options=all(a[k] == b[k] == v for k, v in option_values.items()),
        same_plan_shape_stride=original.plan_src.shape == packed.plan_src.shape
        and original.plan_src.stride() == packed.plan_src.stride()
        and a["miss_dst"].shape == b["miss_dst"].shape
        and a["miss_dst"].stride() == b["miss_dst"].stride(),
        gpu_plans_match_cpu=torch.equal(original.plan_src.cpu(), plan["src"])
        and torch.equal(packed.plan_src.cpu(), adjacent["src"])
        and torch.equal(a["miss_dst"].cpu(), plan["dst"])
        and torch.equal(b["miss_dst"].cpu(), adjacent["dst"]),
        adjacent_indices=bool(
            (adjacent["active_src"].reshape(64, -1).diff(dim=1) == 1).all()
        )
        and adjacent["dst"][:, :100].reshape(-1).tolist() == list(range(3200)),
        linear_inputs_match_reservation=[t.data_ptr() for t in cases["linear4"].inputs]
        == [t.data_ptr() for t in inputs]
        and all(t.numel() == PAYLOAD and t.data_ptr() % 128 == 0 for t in inputs),
        same_bytes_banks_chunk=all(
            (c.nbytes, c.banks) == (PAYLOAD, 64) for c in cases.values()
        )
        and len({g.waves for g in graphs.values()}) == 1,
        host_pinned=host.is_pinned(),
    )
    return dict(device=device, valid=all(checks.values()), checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "main"))
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (cli.output_dir / name).exists()
        for name in ("bridge.json", "bridge-compact.txt")
    ):
        parser.error("Use a fresh output directory")
    args = SimpleNamespace(
        batch=32,
        seed=0,
        plan_stride=3072,
        buffer_slots=3072,
        source_layout_control=False,
        pcie_rx_probe=True,
    )
    report = dict(
        success=False,
        config=dict(
            mode=cli.mode,
            batch=32,
            misses=100,
            order=[0, 1, 1, 0],
            banks=64,
            payload=PAYLOAD,
            num_blocks=4,
            block_size=1024,
            plan_stride=3072,
            register_device=1,
            register_flags=1,
            warm_s=2.0,
            duration_s=2.5,
            trim_s=0.3,
            rx_sampler=False,
        ),
        correctness=[],
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
        report["code"] = vector_provenance()
        root = Path(__file__).resolve().parents[2]
        for name in (
            "scripts/pd_test/bench_h2d_kv_bridge.py",
            "scripts/pd_test/bench_h2d_linear.py",
            "scripts/pd_test/bench_h2d_rx.py",
            "python/sglang/benchmark/hisparse_h2d_kv_bridge.py",
            "python/sglang/benchmark/hisparse_h2d_layout.py",
            "python/sglang/benchmark/hisparse_h2d_linear.py",
            "python/sglang/benchmark/hisparse_h2d_rx.py",
            "python/sglang/kernels/jit/csrc/hisparse_linear_probe.cuh",
        ):
            report["code"]["sha256"][name] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
        geometry = bridge_geometry()
        report["geometry"] = geometry
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
                    name="shared_bridge_pool",
                    address=host.data_ptr(),
                    nbytes=host.numel(),
                )
            ]
        )
        report["active_case"] = "CPU packing"
        started = time.monotonic_ns()
        plan = make_plan(torch, args, geometry, 0, 100)
        adjacent, oracle, inputs, report["packing"] = prepare_bridge(
            torch, host, plan, geometry
        )
        report["packing_us"] = (time.monotonic_ns() - started) / 1000
        cases, graphs = {}, {}
        for device in (0, 1):
            report["active_case"] = dict(
                device=device, method="gather4", phase="eager/bank_graph"
            )
            original = CopyCase(
                torch, host, device, "B32-miss100", "sm", PAYLOAD, 64, args, plan, 100
            )
            # Reuse the existing two-plan replacement path. No sorting occurs:
            # the supplied CPU plans already contain adjacent absolute indices.
            report["active_case"] = dict(
                device=device, method="gather_adjacent4", phase="eager/bank_graph"
            )
            packed = CopyCase(
                torch,
                host,
                device,
                "B32-miss100",
                "sm_ordered",
                PAYLOAD,
                64,
                args,
                adjacent,
                100,
                shared_sm=original,
                oracle_plan=oracle,
            )
            report["active_case"] = dict(device=device, method="linear4", phase="eager")
            linear = ContiguousCase(torch, device, "linear4", inputs)
            group = dict(zip(METHODS, (original, packed, linear)))
            captured = {}
            for method, case in group.items():
                case.probe_method = method
                report["correctness"].append(
                    dict(
                        device=device,
                        method=method,
                        eager_banks=[0, 63],
                        all_bytes=True,
                        eager=True,
                    )
                )
                report["active_case"] = dict(
                    device=device, method=method, phase="chunk_graph"
                )
                captured[method] = SustainedGraph(
                    torch,
                    case,
                    100,
                    waves=None if method == "gather4" else captured["gather4"].waves,
                )
                report["chunk_graphs"].append(captured[method].metadata)
            binding = binding_record(
                torch, device, host, group, captured, plan, adjacent, inputs
            )
            report["bindings"].append(binding)
            if not binding["valid"]:
                raise ValueError(f"Bridge bindings failed: {binding}")
            cases[device], graphs[device] = group, captured
        report["jit_artifacts"] = [
            dict(a, name=Path(a["path"]).name)
            for a in (loaded_grid_artifact(), loaded_linear_artifact())
        ]
        if cli.mode == "main":
            local = {0: 0, 1: 0}
            for trial, device in enumerate(report["config"]["order"]):
                for position, method in enumerate(method_order(local[device])):
                    report["active_case"] = dict(
                        device=device, method=method, trial=trial
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
                            False,
                            args,
                        )
                    report["windows"].append(row)
                    print(
                        f"Completed {report['active_case']}: {row['metrics']}",
                        flush=True,
                    )
                local[device] += 1
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        (cli.output_dir / "bridge.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        try:
            compact = compact_report(report)
        except Exception as exc:
            compact = f"results_valid=False\nAcquisition error: {report.get('error')}\nSummary error: {exc}\n"
        (cli.output_dir / "bridge-compact.txt").write_text(compact)
        print(compact, flush=True)
        if registered:
            for device in (0, 1):
                torch.cuda.synchronize(device)
            _cuda_host_unregister(host)
        if observer is not None:
            observer.nv.nvmlShutdown()
    return 0 if report_status(report)["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
