"""Original vs single-change retry candidate: correctness only, not timing."""

import argparse
import json
import traceback
from pathlib import Path

import check_hisparse_pr40670_w6 as prior
from bench_h2d_latency import host_local
from bench_h2d_pair import Observer
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_pr40670_check import source_manifest, validate_profile
from sglang.benchmark.hisparse_pr40670_retry import compare_results, validate_candidate
from sglang.benchmark.hisparse_pr40670_w6 import (
    HOT,
    SCRATCH,
    TAIL,
    TOP_K,
    WIDTH,
    input_tokens,
)

base = prior.base
inspect = prior.inspect


def execute(torch, args, report):
    from sglang.benchmark.hisparse_pr40670_reference import (
        load_cache_to_device_buffer_spec_mla,
    )
    from sglang.benchmark.hisparse_pr40670_retry_candidate import (
        load_cache_to_device_buffer_spec_mla as retry_swap,
    )
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

    case = base.make_case(torch, WIDTH, TOP_K, HOT, TAIL)
    try:
        snap = process_snapshot(
            buffers=[
                dict(
                    name="same_geometry_host",
                    address=case.allocation.data_ptr(),
                    nbytes=case.allocation.numel(),
                )
            ]
        )
        report["host_node1"] = host_local(snap, case.allocation.numel())
        if not report["host_node1"]:
            raise ValueError("Host placement not verified")
        report["bindings"] = {
            k: case.args[k].data_ptr()
            for k in (
                "top_k_tokens",
                "device_buffer_tokens",
                "device_buffer_locs",
                "host_cache",
                "device_buffer",
            )
        }
        report["config"] = dict(
            batch=4,
            real=3,
            width=WIDTH,
            top_k=TOP_K,
            hot=HOT,
            tail=TAIL,
            scratch=SCRATCH,
        )
        for variant, swap in (
            ("original", load_cache_to_device_buffer_spec_mla),
            ("retry_fixed", retry_swap),
        ):
            for unique in (8, 96):
                row = dict(
                    variant=variant, unique_misses=unique, checks=[], success=False
                )
                report["cases"].append(row)

                def prepare():
                    base.reset(torch, case)
                    data = (
                        torch.tensor(input_tokens(unique), dtype=torch.int32)
                        .view(1, WIDTH, TOP_K)
                        .expand(4, -1, -1)
                    )
                    case.args["top_k_tokens"].copy_(data)
                    case.args["top_k_device_locs"].fill_(-1)
                    case.args["seq_lens"].fill_(case.host_stride)
                    return case.output.cpu()

                before = prepare()
                swap(**case.args)
                torch.cuda.synchronize()
                row["eager"] = inspect(torch, case, before)
                if not row["eager"]["valid"]:
                    row["status"] = (
                        "UNRESOLVED_VALID_KV"
                        if row["eager"]["missing"]
                        else "VALIDATION_ERROR"
                    )
                    continue  # Preserve actual failure; no graph/performance claim.
                stream = torch.cuda.Stream(device=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    swap(**case.args)
                for trial in range(6):
                    before = prepare()
                    graph.replay()
                    torch.cuda.synchronize()
                    result = inspect(torch, case, before)
                    row["checks"].append(result)
                    if not result["valid"]:
                        break
                if len(row["checks"]) != 6 or not all(
                    c["valid"] for c in row["checks"]
                ):
                    row["status"] = "GRAPH_VALIDATION_ERROR"
                    continue
                before = prepare()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    graph.replay()
                    torch.cuda.synchronize()
                path = args.output_dir / f"{variant}-unique{unique}.trace.json"
                prof.export_chrome_trace(str(path))
                row["profile"] = validate_profile(
                    json.loads(path.read_text()), 4, WIDTH, TOP_K
                )
                row["profile_check"] = inspect(torch, case, before)
                row["success"] = row["profile_check"]["valid"]
                row["status"] = "PASS" if row["success"] else "PROFILE_VALIDATION_ERROR"
        paths = {
            f[5]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if len(f := line.split(None, 5)) == 6
        }
        report["jit"] = {}
        for variant, prefix in [
            ("original", "hisparse_pr40670_reference"),
            ("retry_fixed", "hisparse_pr40670_retry_candidate"),
        ]:
            libs = [
                p
                for p in paths
                if Path(p).name.startswith(f"sgl_kernel_jit_{prefix}_64_16_128_6_")
            ]
            if len(libs) != 1:
                raise ValueError(f"Expected one {variant} swap JIT: {libs}")
            report["jit"][variant] = dict(path=libs[0], sha256=sha256(libs[0]))
        report["comparison"] = compare_results(report["cases"])
        report["completed"] = True
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(case.allocation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        completed=False, cases=[], performance_ready=False, service_integrated=False
    )
    observer = None
    try:
        report["source_manifest"] = source_manifest()
        report["candidate_source"] = validate_candidate()
        report["helper_sha256"] = {
            str(f): sha256(f)
            for f in [Path(base.__file__), Path(prior.__file__), Path(__file__)]
        }
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("NVIDIA CUDA required")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        observer = Observer(torch, [0])
        report["identity"] = observer.metadata[0]
        if report["identity"]["uuid"] != "GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702":
            raise ValueError("Require GPU4")
        execute(torch, args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.nv.nvmlShutdown()
        (args.output_dir / "retry.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            f"RETRY completed={report['completed']} cases={len(report['cases'])}/4 comparison={report.get('comparison')} performance_ready=False",
            f"CONFIG {report.get('config')} HOST node1={report.get('host_node1')} JIT {report.get('jit')}",
            f"SOURCE {report.get('candidate_source')}",
        ]
        for c in report["cases"]:
            e = c.get("eager", {})
            lines.append(
                f"CASE {c['variant']} unique={c['unique_misses']} status={c.get('status')} eager_valid={e.get('valid')} graph={len(c['checks'])}/6 admitted={e.get('admitted_occurrences')} unresolved={e.get('unresolved_valid_occurrences')} profile={c.get('profile')}"
            )
            lines.append(
                f"MISSING {e.get('missing')} OTHER_ERRORS {e.get('other_errors')}"
            )
            for d in e.get("diagnostics", []):
                lines.append("FINAL_STATE " + str(d))
        if "error" in report:
            lines.append("ERROR " + report["error"])
        text = "\n".join(lines) + "\n"
        (args.output_dir / "retry-compact.txt").write_text(text)
        print(text)
    return (
        0
        if report["completed"]
        and report.get("comparison", {}).get("candidate_validated")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
