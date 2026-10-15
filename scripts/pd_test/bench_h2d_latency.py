"""CTA4/8 per-warp region diagnostic. No production dispatch changes."""

import argparse
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_copy import capture_wave, validate_sm
from bench_h2d_pair import Observer, Warmer, make_plan
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import copy_geometry, process_snapshot
from sglang.benchmark.hisparse_h2d_iterator import (
    check_iterator_bytes,
    graph_states,
    make_iterator_case,
)
from sglang.benchmark.hisparse_h2d_latency import (
    ORDERS,
    VARIANTS,
    launch,
    loaded_artifacts,
    perturbation,
    summarize_invocation,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "python/sglang/kernels/jit/csrc/hisparse_copy_latency.cuh",
    "python/sglang/kernels/jit/csrc/hisparse.cuh",
    "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh",
    "python/sglang/kernels/ops/kvcache/hisparse.py",
    "python/sglang/benchmark/hisparse_h2d_latency.py",
    "python/sglang/benchmark/hisparse_h2d_iterator.py",
    "scripts/pd_test/bench_h2d_latency.py",
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
        (blocks * 32 + 2, 9), -1, dtype=torch.int64, device=case.output.device
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
        rows = validate_records(
            guard.cpu().tolist(), blocks, counts, real, variant == "trace"
        )
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
        report["success"] = len(report["correctness"]) == 12
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(allocation)


def verify_review(path, code):
    review = json.loads(path.read_text())
    if review.get("timing_semantics_reviewed") is not True or not review.get("notes"):
        raise ValueError("Require completed SASS review with notes")
    extraction_path = Path(review["codegen_report"])
    if sha256(extraction_path) != review["codegen_sha256"]:
        raise ValueError("Codegen report changed")
    extraction = json.loads(extraction_path.read_text())
    if not extraction.get("success") or not extraction.get("no_spill"):
        raise ValueError("Require successful extraction, no spill")
    check_path = Path(extraction["check_report"])
    if sha256(check_path) != extraction["input_sha256"]:
        raise ValueError("Correctness report changed")
    check = json.loads(check_path.read_text())
    if not check.get("success") or check["code"]["sha256"] != code["sha256"]:
        raise ValueError("Correctness code changed; rerun check/codegen")
    if {a["variant"]: a["sha256"] for a in check["jit_artifacts"]} != {
        a["variant"]: a["sha256"] for a in extraction["artifacts"]
    }:
        raise ValueError("Correctness/extraction artifact mismatch")
    return extraction, review


def host_local(snapshot, size):
    buffers = snapshot.get("buffers", [])
    if snapshot.get("errors") or len(buffers) != 1:
        return False
    b = buffers[0]
    return (
        b.get("mapped_bytes") == size
        and bool(b.get("regions"))
        and all(
            r.get("buffer_covers_whole_vma") is True
            and (r.get("numa") or {}).get("policy") == "bind:1"
            and set(r["numa"].get("pages_by_node", {})) == {"1"}
            and r["numa"]["pages_by_node"]["1"] > 0
            for r in b["regions"]
        )
    )


def run_main(torch, args, report, observer):
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    extraction, review = verify_review(args.review, report["code"])
    report["review"] = review
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
        graphs, guards, events = {}, {}, {}
        report["bindings"] = dict(
            same_plan_source_target=True,
            only_grid_changes=True,
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

        for variant in VARIANTS:
            for blocks in (4, 8):
                key = variant, blocks
                guard = torch.full(
                    (64, blocks * 32 + 2, 9), -1, dtype=torch.int64, device="cuda"
                )
                guards[key] = guard
                fns = [
                    lambda bank=bank, v=variant, b=blocks, g=guard: launch(
                        v, b, src[bank], dst, counts, real, host, output, g[bank, 1:-1]
                    )
                    for bank in range(64)
                ]
                for bank in (0, 63):
                    output.zero_()
                    fns[bank]()
                    torch.cuda.synchronize()
                    check_output(bank)
                    validate_records(
                        guard[bank].cpu().tolist(),
                        blocks,
                        [100] * 32,
                        32,
                        variant == "trace",
                    )
                graphs[key], events[key] = capture_wave(
                    torch, fns, stream=torch.cuda.Stream(device=0)
                )
                output.zero_()
                guard.fill_(-1)
                graphs[key].replay()
                torch.cuda.synchronize()
                check_output(63)
                for bank in guard.cpu().tolist():
                    validate_records(bank, blocks, [100] * 32, 32, variant == "trace")
                report["correctness"].append(
                    dict(
                        variant=variant,
                        blocks=blocks,
                        eager_banks=[0, 63],
                        graph=True,
                        records=True,
                        bytes_padding=True,
                    )
                )
        report["jit_artifacts"] = loaded_artifacts()
        expected = {a["variant"]: a["sha256"] for a in extraction["artifacts"]}
        if expected != {a["variant"]: a["sha256"] for a in report["jit_artifacts"]}:
            raise ValueError("Loaded artifacts differ from reviewed check/codegen")
        report["loaded_artifacts_match"] = True
        warmer = Warmer(torch, 0)
        for trial, variants in enumerate(ORDERS):
            for variant in variants:
                for blocks in ((4, 8) if trial % 2 == 0 else (8, 4)):
                    key = variant, blocks
                    report["active"] = dict(trial=trial, variant=variant, blocks=blocks)
                    warmer.run(250)
                    for _ in range(2):
                        graphs[key].replay()
                    torch.cuda.synchronize()
                    # Same memset/sync for all variants, outside measured graph.
                    guards[key].fill_(-1)
                    torch.cuda.synchronize()
                    before = observer.sample(0)
                    started = time.monotonic_ns()
                    graphs[key].replay()
                    events[key][-1][1].synchronize()
                    completed = time.monotonic_ns()
                    after = observer.sample(0)
                    wave = dict(
                        trial=trial,
                        variant=variant,
                        blocks=blocks,
                        copy_us=[s.elapsed_time(e) * 1000 for s, e in events[key]],
                        gpu_wave_us=events[key][0][0].elapsed_time(events[key][-1][1])
                        * 1000,
                        wall_us=(completed - started) / 1000,
                        state_before=before,
                        state_after=after,
                    )
                    if any(x <= 0 for x in wave["copy_us"]):
                        raise ValueError("Nonpositive CUDA event")
                    raw = guards[key].cpu().tolist()
                    if variant == "trace":
                        raw_path = (
                            args.output_dir / f"trace-wave{trial}-cta{blocks}.json"
                        )
                        raw_path.write_text(
                            json.dumps(
                                dict(
                                    fields=[
                                        "cta",
                                        "warp_in_cta",
                                        "warp",
                                        "sm",
                                        "begin_ns",
                                        "end_ns",
                                        "copy_ns",
                                        "kv",
                                        "visits",
                                    ],
                                    banks=raw,
                                )
                            )
                            + "\n"
                        )
                        wave.update(
                            raw_records=raw_path.name,
                            raw_sha256=sha256(raw_path),
                        )
                    report["waves"].append(wave)
                    summaries = []
                    for bank, rows in enumerate(raw):
                        parsed = validate_records(
                            rows, blocks, [100] * 32, 32, variant == "trace"
                        )
                        if parsed:
                            summaries.append(summarize_invocation(parsed))
                    wave["records_valid"] = True
                    if variant == "trace":
                        wave["invocations"] = summaries
        report["perturbation"] = perturbation(report["waves"])
        report["telemetry_valid"] = all(
            not w[k]["errors"]
            and w[k].get("pstate") == 0
            and w[k].get("pcie_gen") == 5
            and w[k].get("pcie_width") == 16
            for w in report["waves"]
            for k in ("state_before", "state_after")
        )
        report["success"] = True
        report["segment_results_valid"] = (
            report["perturbation"]["valid"] and report["telemetry_valid"]
        )
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(allocation)


def compact(report):
    lines = [
        f'mode={report["mode"]} success={report["success"]} correctness={len(report["correctness"])}/{12 if report["mode"]=="check" else 6} segment_results_valid={report.get("segment_results_valid",False)}',
        f'IDENTITY {report.get("identity")} HOST node1={report.get("host_node1","check uses pinned allocator")} register_flags={report.get("host_register_flags","allocator")}',
        f'BINDINGS {report.get("bindings")}',
        f'JIT artifacts={len(report.get("jit_artifacts",[]))}/3 loaded_match={report.get("loaded_artifacts_match")}',
    ]
    if report["mode"] == "check":
        for r in report["correctness"]:
            lines.append(
                f'CHECK {r["variant"]} CTA{r["blocks"]} {r["source"]} eager=True graph={len(r["graph"])}/6 bytes_padding=True records_guards=True capture_device={r["capture_device"]}'
            )
    if "perturbation" in report:
        for name, g in report["perturbation"]["gates"].items():
            lines.append(f"PERTURB {name} {g}")
        for v in VARIANTS:
            groups = {
                b: [
                    statistics.fmean(w["copy_us"])
                    for w in report["waves"]
                    if w["variant"] == v and w["blocks"] == b
                ]
                for b in (4, 8)
            }
            lines.append(
                f"TOTAL {v} CTA4/8_us={groups} ratios={[b/a for a,b in zip(groups[4],groups[8])]}"
            )
    if report.get("segment_results_valid"):
        for blocks in (4, 8):
            waves = [
                w
                for w in report["waves"]
                if w["variant"] == "trace" and w["blocks"] == blocks
            ]
            # Each entry is a median over 64 invocation summaries, six entries per metric.
            for metric in ("loop_us", "copy_us", "other_us", "copy_per_kv_us"):
                line = {
                    q: [
                        statistics.median(
                            i["metrics"][metric][q] for i in w["invocations"]
                        )
                        for w in waves
                    ]
                    for q in ("p50", "p95", "max")
                }
                lines.append(f"SEGMENT CTA{blocks} {metric} per_wave_medians={line}")
            lines.append(
                f"PLACEMENT CTA{blocks} per_wave="
                + str(
                    {
                        k: [
                            statistics.median(i[k] for i in w["invocations"])
                            for w in waves
                        ]
                        for k in (
                            "sm_count",
                            "start_spread_us",
                            "end_spread_us",
                            "envelope_us",
                        )
                    }
                )
            )
            for kv in (25,) if blocks == 4 else (12, 13):
                lines.append(
                    f"KV_GROUP CTA{blocks} copied_kv={kv} copy_us_p50_per_wave="
                    + str(
                        [
                            statistics.median(
                                i["by_kv"][str(kv)]["copy_us"]["p50"]
                                for i in w["invocations"]
                            )
                            for w in waves
                        ]
                    )
                )
            tails = [[i["last_loop_warp"] for i in w["invocations"]] for w in waves]
            lines.append(
                f"TAIL CTA{blocks} per_wave_medians="
                + str(
                    {
                        "loop_us": [
                            statistics.median(
                                (r["end_ns"] - r["begin_ns"]) / 1000 for r in rows
                            )
                            for rows in tails
                        ],
                        "copy_us": [
                            statistics.median(r["copy_ns"] / 1000 for r in rows)
                            for rows in tails
                        ],
                        "other_us": [
                            statistics.median(
                                (r["end_ns"] - r["begin_ns"] - r["copy_ns"]) / 1000
                                for r in rows
                            )
                            for rows in tails
                        ],
                        "kv": [
                            statistics.median(r["kv"] for r in rows) for rows in tails
                        ],
                    }
                )
            )
    elif report["mode"] == "main":
        lines.append(
            "SEGMENT suppressed; raw observations retained, no attribution allowed."
        )
    if report.get("waves"):
        states = [
            w[k] for w in report["waves"] for k in ("state_before", "state_after")
        ]
        lines.append(
            "STATE "
            + str(
                {
                    k: sorted({str(s.get(k)) for s in states})
                    for k in (
                        "sm_MHz",
                        "mem_MHz",
                        "pstate",
                        "pcie_gen",
                        "pcie_width",
                        "throttle_bits",
                    )
                }
            )
        )
    if "error" in report:
        lines.append("ERROR " + report["error"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["check", "main"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--review", type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    if args.mode == "main" and args.review is None:
        p.error("main requires --review after SASS review")
    args.output_dir.mkdir(parents=True)
    report = dict(
        mode=args.mode,
        success=False,
        segment_results_valid=False,
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
            raise ValueError("This round requires physical GPU4 UUID on CUDA0")
        if args.mode == "check":
            run_check(torch, report)
        else:
            run_main(torch, args, report, observer)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.nv.nvmlShutdown()
        (args.output_dir / "latency.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        text = compact(report)
        (
            args.output_dir
            / ("check-compact.txt" if args.mode == "check" else "latency-compact.txt")
        ).write_text(text)
        print(text)
    return (
        0
        if report["success"]
        and (args.mode == "check" or report["segment_results_valid"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
