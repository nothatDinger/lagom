"""GPU4 full PR retry-candidate versus local metadata-plan + copy, cold inputs."""

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import check_hisparse_pr40670 as base
from bench_h2d_latency import host_local
from bench_h2d_pair import Observer
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_pr40670_pipeline import (
    METHODS,
    ORDERS,
    summarize,
    validate_pipeline_profile,
)
from sglang.benchmark.hisparse_pr40670_retry import validate_candidate
from sglang.benchmark.hisparse_pr40670_w6 import input_tokens

ROOT = Path(__file__).resolve().parents[2]


def make_pair(torch, batch, unique):
    from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
    from sglang.kernels.ops.kvcache.hisparse_window_hash import (
        allocate_window_hash_states,
        initialize_window_hash_state,
        plan_cache_window_hash_mla,
    )
    from sglang.benchmark.hisparse_pr40670_retry_candidate import (
        load_cache_to_device_buffer_spec_mla,
    )
    from sglang.srt.mem_cache.hisparse_window_hash import HiSparseWindowHashLayout

    # make_case allocates batch-1 request slots. All B real requests here;
    # padding semantics were separately checked by the reference harness.
    pr = base.make_case(torch, 6, 16, 128, 16, batch=batch + 1)
    try:
        a = pr.args
        a["req_pool_indices"] = torch.arange(
            batch, dtype=torch.int64, device="cuda"
        ).roll(1)
        a["top_k_tokens"] = (
            torch.tensor(input_tokens(unique), dtype=torch.int32, device="cuda")
            .view(1, 6, 16)
            .repeat(batch, 1, 1)
        )
        a["top_k_device_locs"] = torch.full_like(a["top_k_tokens"], -1)
        a["seq_lens"] = torch.full(
            (batch * 6,), pr.host_stride, dtype=torch.int32, device="cuda"
        )
        before = base.reset(torch, pr)
        a["top_k_device_locs"].fill_(-1)
        initial_output = pr.output.clone()
        la = {
            k: v
            for k, v in a.items()
            if k not in ("state", "transfer_kind", "item_size_bytes")
        }
        la["device_buffer"] = initial_output.clone()
        la["device_buffer_tokens"] = pr.initial_tokens[:, :129].clone()
        la["device_buffer_locs"] = pr.initial_locs[:, :129].clone()
        la["top_k_device_locs"] = torch.full_like(a["top_k_tokens"], -1)
        la["state"] = allocate_window_hash_states(
            HiSparseWindowHashLayout(128, 6, 16),
            num_layers=1,
            num_req_slots=batch,
            device="cuda",
        )[0]
        la["miss_src"] = torch.full((batch, 96), -1, device="cuda", dtype=torch.int64)
        la["miss_dst"] = torch.full((batch, 96), -1, device="cuda", dtype=torch.int32)
        la["miss_count"] = torch.zeros(batch, device="cuda", dtype=torch.int32)
        initialize_window_hash_state(
            device_buffer_tokens=la["device_buffer_tokens"],
            req_pool_indices=la["req_pool_indices"],
            state=la["state"],
        )
        torch.cuda.synchronize()

        def local():
            plan_cache_window_hash_mla(**la)
            copy_cache_planned_mla(
                **{
                    k: la[k]
                    for k in (
                        "miss_src",
                        "miss_dst",
                        "miss_count",
                        "num_real_reqs",
                        "host_cache",
                        "device_buffer",
                    )
                },
                item_size_bytes=584,
                num_blocks=4,
                block_size=1024,
                is_dsv4_layout=True,
                load_policy="legacy",
            )

        def snapshot(args, is_pr):
            tensors = [
                args[k]
                for k in (
                    "device_buffer_tokens",
                    "device_buffer_locs",
                    "device_buffer",
                    "top_k_device_locs",
                )
            ]
            if is_pr:
                # Full physical map includes rotating scratch aliases.
                tensors += [pr.locs]
            tensors += [
                args["state"].cache_index,
                args["state"].cache_policy,
                args["state"].scratch_state,
            ]
            if not is_pr:
                tensors += [args[k] for k in ("miss_src", "miss_dst", "miss_count")]
            return [(t, t.clone()) for t in tensors]

        readers = {}
        for method, args, call in [
            (METHODS[0], a, lambda: load_cache_to_device_buffer_spec_mla(**a)),
            (METHODS[1], la, local),
        ]:
            pairs = snapshot(args, method == METHODS[0])

            def reset(pairs=pairs):
                for dst, src in pairs:
                    dst.copy_(src)

            readers[method] = SimpleNamespace(
                args=args, call=call, reset=reset, snapshots=pairs
            )
        return pr, before, readers
    except Exception:
        from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

        torch.cuda.synchronize()
        _cuda_host_unregister(pr.allocation)
        raise


def oracle(torch, pr, before, reader, unique, method):
    a = reader.args
    loc = a["top_k_device_locs"].cpu()
    batch = loc.shape[0]
    output = a["device_buffer"].cpu()
    if not bool(((loc >= 0) & (loc < output.shape[0] * 64)).all()):
        raise ValueError("Unresolved/out-of-range valid KV")
    src = (
        a["req_pool_indices"].cpu()[:, None, None] * pr.host_stride
        + a["top_k_tokens"].cpu()
    ).reshape(-1)
    dst = loc.reshape(-1).long()
    si, di = base.c4_indices(torch, src), base.c4_indices(torch, dst)
    if not torch.equal(output.view(-1)[di], pr.host.view(-1)[si]):
        raise ValueError("Value/scale bytes mismatch")
    permitted = torch.zeros(output.numel(), dtype=torch.bool)
    permitted[di] = True
    if not torch.equal(output.view(-1)[~permitted], before.view(-1)[~permitted]):
        raise ValueError("Unreferenced output/guard changed")
    # Unique tokens must resolve once each, consistently across duplicate occurrences.
    tokens = a["top_k_tokens"].cpu()
    for b in range(batch):
        rid = int(a["req_pool_indices"][b].item())
        allowed = (
            pr.initial_locs[rid].cpu()
            if method == METHODS[0]
            else pr.initial_locs[rid, :129].cpu()
        )
        if not bool(torch.isin(loc[b], allowed).all()):
            raise ValueError("Output location outside owning request slots")
        if dst.view(batch, -1)[b].unique().numel() != unique:
            raise ValueError("Unexpected unique destinations")
        for token in tokens[b].unique():
            if loc[b][tokens[b] == token].unique().numel() != 1:
                raise ValueError("Duplicate token resolves inconsistently")
    if method == METHODS[0]:
        if sorted(pr.locs.cpu().reshape(-1).tolist()) != sorted(
            pr.initial_locs.cpu().reshape(-1).tolist()
        ):
            raise ValueError("PR physical location permutation lost")
    else:
        if not torch.equal(a["device_buffer_locs"], pr.initial_locs[:, :129]):
            raise ValueError("Local fixed physical locations changed")
        if a["miss_count"].cpu().tolist() != [unique] * batch:
            raise ValueError("Local miss count mismatch")
    return dict(
        valid=True,
        resolved=batch * 96,
        unique_per_request=unique,
        all_bytes=True,
        untouched=True,
    )


def capture(torch, reader, timed=False):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=0)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        if timed:
            reader.reset()  # D2D reset outside measured interval on same stream.
            start.record()
        reader.call()
        if timed:
            end.record()
    return graph, start, end


def execute(torch, args, report, observer):
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

    for batch in (4, 32):
        for unique in (8, 96):
            row = dict(batch=batch, unique=unique, checks={}, waves=[])
            report["cases"].append(row)
            pr, before, readers = make_pair(torch, batch, unique)
            try:
                snap = process_snapshot(
                    buffers=[
                        dict(
                            name="shared_host",
                            address=pr.allocation.data_ptr(),
                            nbytes=pr.allocation.numel(),
                        )
                    ]
                )
                row["host_node1"] = host_local(snap, pr.allocation.numel())
                if not row["host_node1"]:
                    raise ValueError("Host placement not NUMA1")
                row["binding"] = dict(
                    shared_host=readers[METHODS[0]].args["host_cache"].data_ptr()
                    == readers[METHODS[1]].args["host_cache"].data_ptr(),
                    independent_outputs=pr.output.data_ptr()
                    != readers[METHODS[1]].args["device_buffer"].data_ptr(),
                )
                if not all(row["binding"].values()):
                    raise ValueError("Bad bindings")
                timed = {}
                for method, reader in readers.items():
                    reader.reset()
                    reader.call()  # JIT and eager check.
                    torch.cuda.synchronize()
                    checks = [oracle(torch, pr, before, reader, unique, method)]
                    reader.reset()
                    torch.cuda.synchronize()
                    graph, _, _ = capture(torch, reader)
                    for _ in range(6):
                        reader.reset()
                        graph.replay()
                        torch.cuda.synchronize()
                        checks.append(oracle(torch, pr, before, reader, unique, method))
                    reader.reset()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        graph.replay()
                        torch.cuda.synchronize()
                    path = args.output_dir / f"B{batch}-u{unique}-{method}.trace.json"
                    prof.export_chrome_trace(str(path))
                    profile = validate_pipeline_profile(
                        json.loads(path.read_text()), method, batch
                    )
                    checks.append(oracle(torch, pr, before, reader, unique, method))
                    row["checks"][method] = dict(
                        valid=True, checks=checks, profile=profile
                    )
                    if args.mode == "main":
                        timed[method] = capture(torch, reader, timed=True)
                        tg, _, _ = timed[method]
                        for _ in range(8):
                            tg.replay()
                        torch.cuda.synchronize()
                        row["checks"][method]["timed_graph_check"] = oracle(
                            torch, pr, before, reader, unique, method
                        )
                if args.mode == "main":
                    for trial, order in enumerate(ORDERS):
                        for method in order:
                            graph, start, end = timed[method]
                            state_before = observer.sample(0)
                            values = []
                            # Each cold-reset graph contains one COMPLETE pipeline.
                            # Synchronize outside event interval before reading event values.
                            for _ in range(args.samples):
                                graph.replay()
                                end.synchronize()
                                values.append(start.elapsed_time(end) * 1000)
                            row["waves"].append(
                                dict(
                                    trial=trial,
                                    method=method,
                                    us=values,
                                    before=state_before,
                                    after=observer.sample(0),
                                )
                            )
                            oracle(torch, pr, before, readers[method], unique, method)
                    row["summary"] = summarize(row["waves"], args.samples)
                row["valid"] = True
            finally:
                torch.cuda.synchronize()
                _cuda_host_unregister(pr.allocation)
    paths = {
        f[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(f := line.split(None, 5)) == 6
    }
    report["jit"] = [
        dict(path=p, sha256=sha256(p))
        for p in sorted(paths)
        if Path(p).name.startswith(
            (
                "sgl_kernel_jit_hisparse_pr40670_retry_candidate_64_",
                "sgl_kernel_jit_hisparse_window_hash_64_",
                "sgl_kernel_jit_sparse_copy_planned_1024_",
            )
        )
    ]
    if len(report["jit"]) != 3:
        raise ValueError("Expected exactly three pipeline JIT modules")
    report["results_valid"] = len(report["cases"]) == 4 and all(
        c.get("valid") for c in report["cases"]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("check", "main"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=32)
    args = p.parse_args()
    if args.samples < 8 or args.output_dir.exists():
        p.error("Require samples>=8 and a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(mode=args.mode, results_valid=False, cases=[])
    observer = None
    try:
        report["candidate_source"] = validate_candidate()
        files = [
            Path(__file__),
            Path(base.__file__),
            ROOT / "python/sglang/kernels/jit/csrc/hisparse_window_hash.cuh",
            ROOT / "python/sglang/kernels/jit/csrc/hisparse.cuh",
            ROOT / "python/sglang/kernels/ops/kvcache/hisparse_window_hash.py",
            ROOT / "python/sglang/kernels/ops/kvcache/hisparse.py",
            ROOT / "python/sglang/benchmark/hisparse_pr40670_pipeline.py",
            ROOT / "python/sglang/benchmark/hisparse_pr40670_retry_candidate.py",
        ]
        report["code"] = {str(f.relative_to(ROOT)): sha256(f) for f in files}
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("NVIDIA CUDA required")
        report["runtime"] = dict(torch=torch.__version__, cuda=torch.version.cuda)
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        observer = Observer(torch, [0])
        report["identity"] = observer.metadata[0]
        if report["identity"]["uuid"] != "GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702":
            raise ValueError("Require physical GPU4")
        execute(torch, args, report, observer)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer:
            observer.nv.nvmlShutdown()
        (args.output_dir / "pipeline.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        lines = [
            f"PIPELINE mode={args.mode} results_valid={report['results_valid']} cases={len(report['cases'])}/4",
            f"IDENTITY {report.get('identity')} RUNTIME {report.get('runtime')}",
            f"SOURCE {report.get('candidate_source')}",
            f"CODE {report.get('code')}",
            f"JIT {report.get('jit')}",
        ]
        for c in report["cases"]:
            lines.append(
                f"CASE B{c['batch']} unique={c['unique']} valid={c.get('valid')} host_node1={c.get('host_node1')} bindings={c.get('binding')}"
            )
            for method, check in c["checks"].items():
                lines.append(
                    f"CHECK {method} eager+graph+profile={len(check['checks'])}/8 profile={check['profile']}"
                )
            if "summary" in c:
                lines.append(
                    f"RESULT B{c['batch']} unique={c['unique']} {c['summary']}"
                )
            for wave in c["waves"]:
                lines.append(
                    f"STATE B{c['batch']} unique={c['unique']} trial={wave['trial']} {wave['method']} before={wave['before']} after={wave['after']}"
                )
        if "error" in report:
            lines.append("ERROR " + report["error"])
        text = "\n".join(lines) + "\n"
        (args.output_dir / "pipeline-compact.txt").write_text(text)
        print(text)
    return 0 if report["results_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
