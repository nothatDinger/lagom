"""Six-group segmented timing and perturbation diagnostic; physical GPU4 only."""

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_copy import capture_wave, validate_sm
from bench_h2d_latency import host_local
from bench_h2d_pair import Observer, Warmer, make_plan
from check_h2d_segments import provenance as check_provenance
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import copy_geometry, process_snapshot
from sglang.benchmark.hisparse_h2d_latency import ORDERS, perturbation
from sglang.benchmark.hisparse_h2d_segments import (
    FIELDS,
    VARIANTS,
    launch,
    loaded_artifacts,
    validate_records,
)
from sglang.benchmark.hisparse_h2d_segments_perf import (
    summarize_invocation,
    verify_codegen,
)

ROOT = Path(__file__).resolve().parents[2]


def provenance():
    result = check_provenance()
    result["perf_sha256"] = {
        name: sha256(ROOT / name)
        for name in (
            "scripts/pd_test/bench_h2d_segments.py",
            "scripts/pd_test/run_h2d_segments_perf.sh",
            "python/sglang/benchmark/hisparse_h2d_segments_perf.py",
            "python/sglang/benchmark/hisparse_h2d_latency.py",
        )
    }
    return result


def run_main(torch, args, report, observer):
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    extraction, review = verify_codegen(args.codegen)
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
                    (64, blocks * 32 + 2, len(FIELDS)),
                    -1,
                    dtype=torch.int64,
                    device="cuda",
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
                for blocks in (4, 8) if trial % 2 == 0 else (8, 4):
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
                    check_output(63)
                    raw = guards[key].cpu().tolist()
                    if variant == "trace":
                        raw_path = (
                            args.output_dir / f"trace-wave{trial}-cta{blocks}.json"
                        )
                        raw_path.write_text(
                            json.dumps(
                                dict(
                                    fields=FIELDS,
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
                    wave["bytes_padding_valid"] = True
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
        f"success={report['success']} correctness={len(report['correctness'])}/6 "
        f"segment_results_valid={report.get('segment_results_valid', False)} "
        f"waves={len(report['waves'])}/36",
        f"IDENTITY {report.get('identity')} HOST node1={report.get('host_node1')} flags={report.get('host_register_flags')}",
        f"BINDINGS {report.get('bindings')} JIT loaded_match={report.get('loaded_artifacts_match')}",
        "SEMANTICS issue includes scheduling/backpressure; residual includes data wait and consumers; store is not HBM completion.",
        "SEGMENTS are instrumented observations; failed perturbation forbids attribution to production kernel.",
    ]
    for name, gate in report.get("perturbation", {}).get("gates", {}).items():
        lines.append(f"PERTURB {name} {gate}")
    if len(report["waves"]) == 36:
        for variant in VARIANTS:
            groups = {
                b: [
                    statistics.fmean(w["copy_us"])
                    for w in report["waves"]
                    if w["variant"] == variant and w["blocks"] == b
                ]
                for b in (4, 8)
            }
            lines.append(
                f"TOTAL {variant} CTA4/8_us={groups} mean_us="
                f"{[statistics.fmean(groups[b]) for b in (4, 8)]} "
                f"ratios={[b / a for a, b in zip(groups[4], groups[8])]}"
            )
    for w in report["waves"]:
        if w["variant"] != "trace" or not w.get("invocations"):
            continue
        inv = w["invocations"]
        metrics = {
            k: statistics.fmean(i["means"][k] for i in inv) for k in inv[0]["means"]
        }
        lines.append(
            f"SEGMENT_OBSERVATION trial={w['trial']} CTA{w['blocks']} {metrics}"
        )
        lines.append(
            f"KV_GROUP trial={w['trial']} CTA{w['blocks']} "
            + str(
                {
                    k: {
                        m: statistics.fmean(i["by_kv"][k]["means"][m] for i in inv)
                        for m in inv[0]["by_kv"][k]["means"]
                    }
                    for k in inv[0]["by_kv"]
                }
            )
        )
        lines.append(
            f"RECORDS trial={w['trial']} CTA{w['blocks']} valid={w['records_valid']} "
            f"banks={len(inv)} total_kv={sorted({i['total_kv'] for i in inv})} "
            f"sm_count={sorted({i['sm_count'] for i in inv})} "
            f"monotonic=True segment_sum=True guards=True raw={w['raw_records']}"
        )
    for w in report["waves"]:
        lines.append(
            f"STATE trial={w['trial']} {w['variant']} CTA{w['blocks']} "
            f"before={w['state_before']} after={w['state_after']}"
        )
    if "error" in report:
        lines.append("ERROR " + report["error"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codegen", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        mode="main",
        success=False,
        segment_results_valid=False,
        correctness=[],
        waves=[],
    )
    observer = None
    try:
        # Reject stale/unreviewed artifacts before allocating GPU data.
        verify_codegen(args.codegen)
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
        (args.output_dir / "segments.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        text = compact(report)
        (args.output_dir / "segments-compact.txt").write_text(text)
        print(text)
    return 0 if report["success"] and report["segment_results_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
