"""Three-shape uninstrumented throughput screen; physical GPU4 only."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_copy import capture_wave, validate_sm
from bench_h2d_latency import host_local
from bench_h2d_pair import Observer, Warmer, make_plan
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import copy_geometry, process_snapshot
from sglang.benchmark.hisparse_h2d_blockshape import (
    CONFIGS,
    ORDERS,
    summarize,
    schedule,
    loaded_artifacts,
    extract_codegen,
)
from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla

ROOT = Path(__file__).resolve().parents[2]


def provenance():
    files = (
        "scripts/pd_test/bench_h2d_blockshape.py",
        "python/sglang/benchmark/hisparse_h2d_blockshape.py",
        "python/sglang/kernels/jit/csrc/hisparse.cuh",
        "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh",
        "python/sglang/kernels/ops/kvcache/hisparse.py",
        "scripts/pd_test/bench_h2d_pair.py",
        "scripts/pd_test/bench_h2d_copy.py",
    )
    return dict(sha256={f: sha256(ROOT / f) for f in files})


def run_main(torch, args, report, observer):
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    config = SimpleNamespace(batch=32, plan_stride=3072, buffer_slots=3072, seed=12013)
    geometry = copy_geometry(32, 100, 512 * 2**20, 64)
    allocation = HostTensorAllocator().allocate((512 * 2**20,), torch.uint8, "cpu")
    allocation.zero_()
    host = allocation[: geometry["host_bytes"]].view(geometry["host_pages"], 37440)
    rc = torch.cuda.cudart().cudaHostRegister(
        allocation.data_ptr(), allocation.numel(), 1
    )
    if int(rc) != 0:
        raise RuntimeError(f"Portable register failed: {rc}")
    try:
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            out=host,
            generator=torch.Generator().manual_seed(config.seed),
        )
        report["host_snapshot"] = process_snapshot(
            buffers=[
                dict(
                    name="host_kv",
                    address=allocation.data_ptr(),
                    nbytes=allocation.numel(),
                )
            ]
        )
        report["host_node1"] = host_local(report["host_snapshot"], allocation.numel())
        report["host_register_flags"] = 1
        if not report["host_node1"]:
            raise ValueError("Host allocation placement not verified on node1")
        plan = make_plan(torch, config, geometry, 0, 100)
        src, dst = plan["src"].cuda(), plan["dst"].cuda()
        counts = torch.full((32,), 100, dtype=torch.int32, device="cuda")
        real = torch.tensor([32], dtype=torch.int32, device="cuda")
        output = torch.zeros(
            (plan["destination_pages"], 37440), dtype=torch.uint8, device="cuda"
        )
        graphs, events = {}, {}
        report["bindings"] = dict(
            same_plan_source_target=True,
            block_size_changes_codegen=True,
            src=src.data_ptr(),
            dst=dst.data_ptr(),
            counts=counts.data_ptr(),
            real=real.data_ptr(),
            source=host.data_ptr(),
            target=output.data_ptr(),
            banks=64,
            bytes_per_copy=1868800,
        )
        report["config"] = {
            **vars(config),
            "banks": 64,
            "warm_ms": 250,
            "warmup": 2,
            "order": ORDERS,
        }

        def check_output(bank):
            validate_sm(
                torch,
                host,
                output,
                plan["active_src"][bank],
                plan["dst"][:, :100],
                check_untouched=True,
            )

        for label, (blocks, threads) in CONFIGS.items():
            fns = [
                lambda bank=bank, b=blocks, t=threads: copy_cache_planned_mla(
                    miss_src=src[bank],
                    miss_dst=dst,
                    miss_count=counts,
                    num_real_reqs=real,
                    host_cache=host,
                    device_buffer=output,
                    item_size_bytes=584,
                    num_blocks=b,
                    block_size=t,
                    is_dsv4_layout=True,
                    load_policy="legacy",
                )
                for bank in range(64)
            ]
            for bank in (0, 63):
                output.zero_()
                fns[bank]()
                torch.cuda.synchronize()
                check_output(bank)
            graphs[label], events[label] = capture_wave(
                torch, fns, stream=torch.cuda.Stream(device=0)
            )
            output.zero_()
            graphs[label].replay()
            torch.cuda.synchronize()
            check_output(63)
            report["correctness"].append(
                dict(label=label, eager_banks=[0, 63], graph=True, bytes_padding=True)
            )
        report["jit_artifacts"] = loaded_artifacts()
        report["codegen"] = extract_codegen(
            report["jit_artifacts"], args.output_dir / "codegen", args.cuobjdump
        )
        if not report["codegen"]["valid"]:
            raise ValueError(
                "Unexpected load/store shape or spill; retain SASS, stop before timing"
            )
        report["schedule"] = {k: schedule(*v) for k, v in CONFIGS.items()}
        warmer = Warmer(torch, 0)
        for trial, order in enumerate(ORDERS):
            for label in order:
                blocks, threads = CONFIGS[label]
                report["active"] = dict(trial=trial, label=label)
                warmer.run(250)
                for _ in range(2):
                    graphs[label].replay()
                torch.cuda.synchronize()
                before = observer.sample(0)
                started = time.monotonic_ns()
                graphs[label].replay()
                events[label][-1][1].synchronize()
                completed = time.monotonic_ns()
                after = observer.sample(0)
                wave = dict(
                    trial=trial,
                    label=label,
                    blocks=blocks,
                    threads=threads,
                    copy_us=[a.elapsed_time(b) * 1000 for a, b in events[label]],
                    gpu_wave_us=events[label][0][0].elapsed_time(events[label][-1][1])
                    * 1000,
                    wall_us=(completed - started) / 1000,
                    state_before=before,
                    state_after=after,
                )
                report["waves"].append(wave)
                check_output(63)
                wave["bytes_padding_valid"] = True
        report["timing"] = summarize(report["waves"])
        report["telemetry_valid"] = all(
            not w[k]["errors"]
            and w[k].get("pstate") == 0
            and w[k].get("pcie_gen") == 5
            and w[k].get("pcie_width") == 16
            for w in report["waves"]
            for k in ("state_before", "state_after")
        )
        report["success"] = True
        report["results_valid"] = (
            report["timing"]["valid"] and report["telemetry_valid"]
        )
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(allocation)


def compact(report):
    lines = [
        f"success={report['success']} results_valid={report.get('results_valid', False)} "
        f"correctness={len(report['correctness'])}/3 waves={len(report['waves'])}/18",
        f"IDENTITY {report.get('identity')} HOST node1={report.get('host_node1')} flags={report.get('host_register_flags')}",
        f"BINDINGS {report.get('bindings')}",
        f"CODEGEN {report.get('codegen')}",
        "PLACEMENT measured=False; CTA count is not SM count. This is a throughput screen, not hardware localization.",
    ]
    for k, v in report.get("schedule", {}).items():
        lines.append(f"SCHEDULE {k} {v}")
    for k, v in report.get("timing", {}).get("groups", {}).items():
        lines.append(f"TOTAL {k} {v}")
    for k, v in report.get("timing", {}).get("ratios", {}).items():
        lines.append(f"RATIO {k} {v}")
    for w in report["waves"]:
        lines.append(
            f"STATE trial={w['trial']} {w['label']} before={w['state_before']} after={w['state_after']}"
        )
    if "error" in report:
        lines.append("ERROR " + report["error"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuobjdump", default="cuobjdump")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        mode="main",
        success=False,
        results_valid=False,
        correctness=[],
        waves=[],
    )
    observer = None
    try:
        report["code"] = provenance()
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("NVIDIA CUDA required")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        observer = Observer(torch, [0])
        report["identity"] = observer.metadata[0]
        report["runtime"] = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if report["identity"]["uuid"] != "GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702":
            raise ValueError("Require physical GPU4 on CUDA0")
        run_main(torch, args, report, observer)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.nv.nvmlShutdown()
        (args.output_dir / "blockshape.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        text = compact(report)
        (args.output_dir / "blockshape-compact.txt").write_text(text)
        print(text)
    return 0 if report["success"] and report["results_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
