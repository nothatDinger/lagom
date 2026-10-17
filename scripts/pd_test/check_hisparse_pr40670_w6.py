"""Same W6/K16/H128 scratch13: 8 vs 96 cold misses; no performance test."""

import argparse
import json
import traceback
from pathlib import Path

import check_hisparse_pr40670 as base
from bench_h2d_latency import host_local
from bench_h2d_pair import Observer
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_pr40670_check import source_manifest, validate_profile
from sglang.benchmark.hisparse_pr40670_w6 import (
    HOT,
    SCRATCH,
    TAIL,
    TOP_K,
    WIDTH,
    decode_request,
    input_tokens,
)


def inspect(torch, case, before):
    a = case.args
    top = a["top_k_tokens"].cpu()
    loc = a["top_k_device_locs"].cpu()
    output = case.output.cpu()
    rids = a["req_pool_indices"].cpu().tolist()
    lengths = a["seq_lens"].cpu().view(4, WIDTH)
    real = int(a["num_real_reqs"].item())
    missing = []
    errors = []
    src = []
    dst = []
    admitted = 0
    invalid = 0
    for b in range(4):
        for w in range(WIDTH):
            for j in range(TOP_K):
                token = int(top[b, w, j])
                position = int(loc[b, w, j])
                if b >= real:
                    if position != 0:
                        errors.append(
                            dict(kind="padding_not_zero", b=b, w=w, j=j, dst=position)
                        )
                elif not 0 <= token < int(lengths[b, w]):
                    invalid += 1
                    if position != -1:
                        errors.append(
                            dict(
                                kind="invalid_token_location",
                                b=b,
                                w=w,
                                j=j,
                                dst=position,
                            )
                        )
                elif position == -1:
                    missing.append(dict(b=b, rid=rids[b], w=w, j=j, token=token))
                elif not 0 <= position < output.shape[0] * 64:
                    errors.append(
                        dict(kind="bad_valid_location", b=b, w=w, j=j, dst=position)
                    )
                else:
                    admitted += 1
                    src.append(rids[b] * case.host_stride + token)
                    dst.append(position)
    permitted = torch.zeros(output.numel(), dtype=torch.bool)
    if src:
        si = base.c4_indices(torch, torch.tensor(src))
        di = base.c4_indices(torch, torch.tensor(dst))
        expected = case.host.view(-1)[si]
        actual = output.view(-1)[di]
        if not torch.equal(expected, actual):
            errors.append(dict(kind="admitted_value_scale_mismatch"))
        permitted[di] = True
    if not torch.equal(output.view(-1)[~permitted], before.view(-1)[~permitted]):
        errors.append(dict(kind="unreferenced_guard_or_padding_modified"))
    locations = case.locs.cpu().tolist()
    initial = case.initial_locs.cpu().tolist()
    if sorted(x for row in locations for x in row) != sorted(
        x for row in initial for x in row
    ):
        errors.append(dict(kind="physical_mapping_not_permutation"))
    state = a["state"]
    metadata = state.scratch_state.cpu().tolist()
    policy = state.cache_policy.cpu().tolist()
    tokens = case.tokens.cpu().tolist()
    counters = metadata[0]
    diagnostics = []
    for b in range(real):
        rid = rids[b]
        diagnostics.append(
            decode_request(
                metadata,
                counters,
                policy,
                tokens[rid],
                locations[rid],
                rid,
                case.slots,
                top[b].view(-1).tolist(),
                [m["token"] for m in missing if m["b"] == b],
            )
        )
    return dict(
        valid=not missing and not errors,
        admitted_occurrences=admitted,
        unresolved_valid_occurrences=len(missing),
        invalid_occurrences=invalid,
        missing=missing,
        other_errors=errors,
        diagnostics=diagnostics,
        raw=dict(
            top_k=top.tolist(),
            device_locs=loc.tolist(),
            scratch_state=metadata,
            policy=policy,
            cache_index=state.cache_index.cpu().tolist(),
            device_tokens=tokens,
            physical_locations=locations,
        ),
    )


def execute(torch, args, report):
    from sglang.benchmark.hisparse_pr40670_reference import (
        load_cache_to_device_buffer_spec_mla,
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
        for unique in (8, 96):
            row = dict(unique_misses=unique, checks=[], success=False)
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
            load_cache_to_device_buffer_spec_mla(**case.args)
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
                load_cache_to_device_buffer_spec_mla(**case.args)
            for trial in range(6):
                before = prepare()
                graph.replay()
                torch.cuda.synchronize()
                result = inspect(torch, case, before)
                row["checks"].append(result)
                if not result["valid"]:
                    break
            if len(row["checks"]) != 6 or not all(c["valid"] for c in row["checks"]):
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
            path = args.output_dir / f"unique{unique}.trace.json"
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
        libs = [
            p
            for p in paths
            if Path(p).name.startswith(
                "sgl_kernel_jit_hisparse_pr40670_reference_64_16_128_6_"
            )
        ]
        if len(libs) != 1:
            raise ValueError(f"Expected one shared swap JIT: {libs}")
        report["jit"] = dict(path=libs[0], sha256=sha256(libs[0]), same_for_both=True)
        report["success"] = len(report["cases"]) == 2 and all(
            r["success"] for r in report["cases"]
        )
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
        success=False, cases=[], performance_ready=False, consumer_safety_verified=False
    )
    observer = None
    try:
        report["source_manifest"] = source_manifest()
        report["base_harness"] = dict(path=base.__file__, sha256=sha256(base.__file__))
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
        (args.output_dir / "w6.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            f"W6 success={report['success']} cases={len(report['cases'])}/2 performance_ready=False",
            f"CONFIG {report.get('config')} HOST node1={report.get('host_node1')} JIT {report.get('jit')}",
        ]
        for c in report["cases"]:
            e = c.get("eager", {})
            lines.append(
                f"CASE unique={c['unique_misses']} status={c.get('status')} eager_valid={e.get('valid')} graph={len(c['checks'])}/6 admitted={e.get('admitted_occurrences')} unresolved={e.get('unresolved_valid_occurrences')} profile={c.get('profile')}"
            )
            lines.append(
                f"MISSING {e.get('missing')} OTHER_ERRORS {e.get('other_errors')}"
            )
            for d in e.get("diagnostics", []):
                lines.append("FINAL_STATE " + str(d))
        if "error" in report:
            lines.append("ERROR " + report["error"])
        text = "\n".join(lines) + "\n"
        (args.output_dir / "w6-compact.txt").write_text(text)
        print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
