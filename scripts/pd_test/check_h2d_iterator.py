"""Compile and check legacy/D1/D2 on two GPUs, then record exact artifacts.

No timings, profiler, NCU or model service. Results do not authorize a
performance run: inspect the offline SASS extraction first.
"""

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path

from bench_h2d_pair import Observer
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_h2d_iterator import (
    VARIANTS,
    make_iterator_case,
    run_iterator_case,
)
from sglang.benchmark.hisparse_h2d_iterator_codegen import loaded_iterator_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        success=False,
        scope="correctness_and_artifact_capture_only",
        correctness=[],
        expected=12,
    )
    observer = None
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("NVIDIA CUDA required")
        if (
            len(set(args.devices)) != 2
            or min(args.devices) < 0
            or max(args.devices) >= torch.cuda.device_count()
        ):
            raise ValueError("Expose both selected GPUs")
        torch.set_num_threads(1)
        observer = Observer(torch, args.devices)
        report.update(
            devices=observer.metadata,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        paths = [
            "python/sglang/kernels/jit/csrc/hisparse_copy_iterator.cuh",
            "python/sglang/kernels/jit/csrc/hisparse.cuh",
            "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh",
            "python/sglang/kernels/ops/kvcache/hisparse.py",
            "python/sglang/benchmark/hisparse_h2d_iterator.py",
            "python/sglang/benchmark/hisparse_h2d_iterator_codegen.py",
            "scripts/pd_test/check_h2d_iterator.py",
        ]
        report["code"] = dict(
            git=commit.stdout.strip() if commit.returncode == 0 else None,
            git_error=commit.stderr.strip() if commit.returncode else None,
            sha256={
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in paths
            },
        )
        for device in args.devices:
            with torch.cuda.device(device):
                case = make_iterator_case(torch, f"cuda:{device}")
                report.setdefault("host_snapshots", []).append(
                    dict(
                        device=device,
                        snapshot=process_snapshot(
                            buffers=[
                                dict(
                                    name="correctness_source",
                                    address=case.host.data_ptr(),
                                    nbytes=case.host.numel(),
                                )
                            ]
                        ),
                    )
                )
                for variant in VARIANTS:
                    for source in ("host", "hbm"):
                        report["active_case"] = dict(
                            device=device, variant=variant, source=source
                        )
                        result = run_iterator_case(torch, case, variant, source)
                        report["correctness"].append(result)
                        print(
                            f"CHECK d{device} {variant} {source} eager=True graph=6/6 all_bytes=True",
                            flush=True,
                        )
        report["jit_artifacts"] = loaded_iterator_artifacts()
        report["success"] = len(report["correctness"]) == report["expected"]
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.nv.nvmlShutdown()
        (args.output_dir / "iterator.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        lines = [
            "Iterator correctness only; no performance results.",
            f'success={report["success"]} correctness={len(report["correctness"])}/{report["expected"]} artifacts={len(report.get("jit_artifacts", []))}/3',
            f'CODE {report.get("code", {}).get("git")} git_error={report.get("code", {}).get("git_error")}',
        ]
        for device, meta in report.get("devices", {}).items():
            lines.append(
                f'IDENTITY d{device} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
            )
        for row in report["correctness"]:
            lines.append(
                f'CHECK d{row["device"]} {row["variant"]} {row["source"]} eager={row["eager"]} graph={len(row["graph_phases"])}/6 all_bytes={row["all_bytes_and_padding"]} capture_device={row["capture_stream_device"]} pinned={row["source_is_pinned"]} stride={row["stride"]}'
            )
        if "error" in report:
            lines.append(f'ACTIVE_CASE {report.get("active_case")}')
            lines.append("ERROR " + report["error"])
        text = "\n".join(lines) + "\n"
        (args.output_dir / "iterator-compact.txt").write_text(text)
        print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
