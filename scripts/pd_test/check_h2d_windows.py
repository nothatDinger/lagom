"""Check-only segmented copy diagnostic. No performance entry point."""

import argparse
import json
import subprocess
import traceback
from pathlib import Path

from bench_h2d_latency import host_local
from bench_h2d_pair import Observer
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_h2d_iterator import (
    check_iterator_bytes,
    graph_states,
    make_iterator_case,
)
from sglang.benchmark.hisparse_h2d_windows import (
    VARIANTS,
    launch,
    loaded_artifacts,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "python/sglang/kernels/jit/csrc/hisparse_copy_windows.cuh",
    "python/sglang/kernels/jit/csrc/hisparse.cuh",
    "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh",
    "python/sglang/kernels/ops/kvcache/hisparse.py",
    "python/sglang/benchmark/hisparse_h2d_windows.py",
    "python/sglang/benchmark/hisparse_h2d_iterator.py",
    "scripts/pd_test/check_h2d_windows.py",
    "scripts/pd_test/bench_h2d_copy.py",
    "scripts/pd_test/bench_h2d_pair.py",
)


def provenance():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return dict(
        git=result.stdout.strip() if result.returncode == 0 else None,
        git_error=result.stderr.strip() if result.returncode else None,
        sha256={p: sha256(ROOT / p) for p in CODE_FILES},
    )


def check_one(torch, case, variant, blocks, kind):
    source = case.host if kind == "host" else case.host.to(case.output.device)
    guard = torch.full(
        (blocks * 32 + 2, 15), -1, dtype=torch.int64, device=case.output.device
    )
    records = guard[1:-1]

    def run():
        launch(
            variant,
            blocks,
            case.src,
            case.dst,
            case.counts,
            case.real,
            source,
            case.output,
            records,
        )

    def validate(counts, real, phase):
        check_iterator_bytes(torch, case, counts, real, variant, phase)
        rows = validate_records(guard.cpu().tolist(), blocks, counts, real, variant)
        return dict(
            phase=phase,
            counts_checked=True,
            guards=True,
            records=len(rows),
            copied_kv=sum(r["kv"] for r in rows),
            zero_span_rows=sum(r["end_ns"] == r["begin_ns"] for r in rows),
        )

    counts = case.capacities + [999]
    case.counts.copy_(torch.tensor(counts, dtype=torch.int32))
    case.real.fill_(65)
    case.output.fill_(165)
    run()
    torch.cuda.synchronize()
    eager = validate(counts, 65, "eager")
    stream = torch.cuda.Stream(device=case.output.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if torch.cuda.current_stream().device != case.output.device:
            raise ValueError("Capture device mismatch")
        run()
    checks = []
    for phase, counts, real in graph_states(case):
        case.host.bitwise_xor_(255)
        if phase == "changed_plan":
            case.src_cpu[0, :2] = case.src_cpu[0, :2].flip(0)
            case.src.copy_(case.src_cpu)
        if kind == "hbm":
            source.copy_(case.host)
        case.counts.copy_(torch.tensor(counts, dtype=torch.int32))
        case.real.fill_(real)
        case.output.fill_(165)
        guard.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        checks.append(validate(counts, real, phase))
    return dict(
        variant=variant,
        blocks=blocks,
        source=kind,
        eager=eager,
        graph=checks,
        bytes_padding=True,
        capture_device=stream.device.index,
    )


def run_check(torch, report):
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    case = make_iterator_case(torch, "cuda:0")
    allocation_bytes = (case.host.numel() + 4095) // 4096 * 4096
    allocation = HostTensorAllocator().allocate((allocation_bytes,), torch.uint8, "cpu")
    allocation.zero_()
    host = allocation[: case.host.numel()].view(case.host.shape)
    host.copy_(case.host)
    rc = torch.cuda.cudart().cudaHostRegister(
        allocation.data_ptr(), allocation.numel(), 1
    )
    if int(rc) != 0:
        raise RuntimeError(f"Portable register failed: {rc}")
    case.host = host
    report["host_register_flags"] = 1
    report["host_snapshot"] = process_snapshot(
        buffers=[
            dict(
                name="correctness_host",
                address=allocation.data_ptr(),
                nbytes=allocation.numel(),
            )
        ]
    )
    report["host_node1"] = host_local(report["host_snapshot"], allocation.numel())
    try:
        if not report["host_node1"]:
            raise ValueError("Correctness host pool placement not verified on node1")
        # Every variant/grid reuses the same plan/source/output allocations.
        report["bindings"] = dict(
            same_plan_source_target=True,
            only_grid_changes=True,
            src=case.src.data_ptr(),
            dst=case.dst.data_ptr(),
            counts=case.counts.data_ptr(),
            real=case.real.data_ptr(),
            source=case.host.data_ptr(),
            target=case.output.data_ptr(),
        )
        for variant in VARIANTS:
            for blocks in (4, 8):
                for kind in ("host", "hbm"):
                    report["active"] = dict(variant=variant, blocks=blocks, source=kind)
                    report["correctness"].append(
                        check_one(torch, case, variant, blocks, kind)
                    )
        report["jit_artifacts"] = loaded_artifacts()
        report["success"] = len(report["correctness"]) == 16
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(allocation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(mode="check", success=False, correctness=[], performance_ready=False)
    observer = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("NVIDIA CUDA required")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        observer = Observer(torch, [0])
        report["identity"] = observer.metadata[0]
        if report["identity"]["uuid"] != "GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702":
            raise ValueError("Require physical GPU4")
        report["code"] = provenance()
        run_check(torch, report)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.nv.nvmlShutdown()
        (args.output_dir / "check.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            f"CHECK success={report['success']} correctness={len(report['correctness'])}/16 performance_ready=False",
            f"IDENTITY {report.get('identity')} HOST node1={report.get('host_node1')}",
            f"BINDINGS {report.get('bindings')}",
        ]
        for r in report["correctness"]:
            lines.append(
                f"CASE {r['variant']} CTA{r['blocks']} source={r['source']} eager=True graph={len(r['graph'])}/6 bytes_padding=True record_guards=True"
            )
        if "error" in report:
            lines.append("ERROR " + report["error"])
        text = "\n".join(lines) + "\n"
        (args.output_dir / "check-compact.txt").write_text(text)
        print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
