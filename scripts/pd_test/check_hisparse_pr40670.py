"""Pinned PR #40670 H2D reference: correctness/graph/launch check only."""

import argparse
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_latency import host_local
from bench_h2d_pair import Observer
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot
from sglang.benchmark.hisparse_pr40670_check import (
    COMMIT,
    source_manifest,
    validate_profile,
)
from sglang.benchmark.hisparse_pr40670_layout import build_hisparse_spec_layout

ROOT = Path(__file__).resolve().parents[2]


def c4_indices(torch, slots):
    page, offset = slots // 64, slots % 64
    value = page[:, None] * 37440 + offset[:, None] * 576 + torch.arange(576)
    scale = page[:, None] * 37440 + 36864 + offset[:, None] * 8 + torch.arange(8)
    return torch.cat((value, scale), dim=1)


def make_case(torch, width, k, hot, tail, batch=4):
    from sglang.benchmark.hisparse_pr40670_reference import HiSparseSpecState
    from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator

    layout = build_hisparse_spec_layout(
        hot_slots=hot,
        tail_capacity_slots=tail,
        compress_ratio=4,
        verify_width=width,
        top_k=k,
    )
    slots = batch - 1
    host_stride = hot + 4 * width * k + 64
    host_tokens = slots * host_stride
    host_bytes = math.ceil(host_tokens / 64) * 37440
    allocation = HostTensorAllocator().allocate(
        (math.ceil(host_bytes / 4096) * 4096,), torch.uint8, "cpu"
    )
    allocation.zero_()
    rc = torch.cuda.cudart().cudaHostRegister(
        allocation.data_ptr(), allocation.numel(), 1
    )
    if int(rc) != 0:
        raise RuntimeError(f"Portable cudaHostRegister failed: {rc}")
    try:
        host = allocation[:host_bytes].view(-1, 37440)
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            out=host,
            generator=torch.Generator().manual_seed(40670),
        )
        per_request = math.ceil(layout.total_buffer_slots / 64) * 64
        total_slots = 64 + slots * per_request + 64
        output = torch.full(
            (total_slots // 64, 37440), 165, dtype=torch.uint8, device="cuda"
        )
        locs = (
            64
            + torch.arange(slots)[:, None] * per_request
            + torch.arange(layout.total_buffer_slots)
        ).to("cuda", torch.int32)
        tokens = torch.full_like(locs, -1)
        tokens[:, :hot] = torch.arange(hot, device="cuda", dtype=torch.int32)
        tokens[:, hot] = host_stride - 1
        state = HiSparseSpecState(
            torch.full(
                (slots, 2, layout.hash_slots), -1, dtype=torch.int64, device="cuda"
            ),
            torch.zeros(
                (slots + 1, layout.policy_width(slots)),
                dtype=torch.int32,
                device="cuda",
            )[:, :hot],
            locs[:, layout.scratch_offset : layout.total_buffer_slots],
            torch.zeros(
                (slots + 1, layout.state_width(slots)), dtype=torch.int32, device="cuda"
            ),
        )
        args = dict(
            top_k_tokens=torch.empty(
                (batch, width, k), dtype=torch.int32, device="cuda"
            ),
            device_buffer_tokens=tokens[:, : layout.active_buffer_slots],
            host_cache_locs=torch.arange(
                host_tokens, dtype=torch.int64, device="cuda"
            ).view(slots, host_stride),
            device_buffer_locs=locs[:, : layout.active_buffer_slots],
            host_cache=host,
            device_buffer=output,
            top_k_device_locs=torch.full(
                (batch, width, k), -1, dtype=torch.int32, device="cuda"
            ),
            req_pool_indices=torch.tensor(
                [2, 0, 1, 999], dtype=torch.int64, device="cuda"
            ),
            seq_lens=torch.full(
                (batch * width,), host_stride, dtype=torch.int32, device="cuda"
            ),
            state=state,
            num_real_reqs=torch.tensor([slots], dtype=torch.int32, device="cuda"),
            transfer_kind="dsv4_paged",
            item_size_bytes=584,
        )
        return SimpleNamespace(
            layout=layout,
            args=args,
            allocation=allocation,
            tokens=tokens,
            locs=locs,
            host_stride=host_stride,
            slots=slots,
            output=output,
            host=host,
            initial_tokens=tokens.clone(),
            initial_locs=locs.clone(),
        )
    except Exception:
        from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

        torch.cuda.synchronize()
        _cuda_host_unregister(allocation)
        raise


def reset(torch, case):
    from sglang.benchmark.hisparse_pr40670_reference import (
        initialize_hisparse_spec_state,
    )

    case.tokens.copy_(case.initial_tokens)
    case.locs.copy_(case.initial_locs)
    cpu = torch.full(case.output.shape, 165, dtype=torch.uint8)
    src = []
    dst = []
    for r in range(case.slots):
        for j in range(case.layout.hot_slots + 1):
            token = int(case.initial_tokens[r, j].item())
            src.append(r * case.host_stride + token)
            dst.append(int(case.initial_locs[r, j].item()))
    cpu.view(-1)[c4_indices(torch, torch.tensor(dst))] = case.host.view(-1)[
        c4_indices(torch, torch.tensor(src))
    ]
    case.output.copy_(cpu)
    case.args["num_real_reqs"].fill_(case.slots)
    initialize_hisparse_spec_state(
        device_buffer_tokens=case.args["device_buffer_tokens"],
        req_pool_indices=case.args["req_pool_indices"][: case.slots],
        state=case.args["state"],
    )
    torch.cuda.synchronize()
    return cpu


def set_input(torch, case, kind):
    L = case.layout
    n = L.occurrences
    data = torch.empty((4, L.verify_width, L.top_k), dtype=torch.int32)
    for b in range(4):
        vals = torch.arange(n, dtype=torch.int32) + L.hot_slots
        if kind in ("duplicates", "warm", "changed_host"):
            vals = L.hot_slots + torch.arange(n, dtype=torch.int32) % max(
                1, L.top_k // 2
            )
        elif kind == "mixed":
            vals[::3] = torch.arange(0, n, 3, dtype=torch.int32) % L.hot_slots
            vals[-3] = case.host_stride - 1
        if kind in ("duplicates", "mixed"):
            vals[-2] = -1
            vals[-1] = case.host_stride + 10
        data[b] = vals.view(L.verify_width, L.top_k)
    case.args["top_k_tokens"].copy_(data)
    case.args["seq_lens"].fill_(case.host_stride)
    # One position has a shorter sequence; validity must follow that step.
    if kind == "mixed":
        case.args["seq_lens"][0] = L.hot_slots + 1
    if kind == "zero_real":
        case.args["num_real_reqs"].zero_()
    return data


def oracle(torch, case, before):
    a = case.args
    top = a["top_k_tokens"].cpu()
    loc = a["top_k_device_locs"].cpu()
    lengths = a["seq_lens"].cpu().view(4, case.layout.verify_width)
    real = int(a["num_real_reqs"].item())
    rids = a["req_pool_indices"].cpu()
    actual = case.output.cpu()
    mask = torch.zeros(actual.numel(), dtype=torch.bool)
    sources = []
    destinations = []
    valid = 0
    for b in range(4):
        for w in range(case.layout.verify_width):
            for j in range(case.layout.top_k):
                token = int(top[b, w, j])
                dst = int(loc[b, w, j])
                expected_valid = b < real and 0 <= token < int(lengths[b, w])
                if not expected_valid:
                    if dst != -1:
                        raise ValueError(
                            f"Invalid occurrence has location: {b, w, j, dst}"
                        )
                    continue
                if not 0 <= dst < case.output.shape[0] * 64:
                    raise ValueError("Missing/out-of-range output location")
                sources.append(int(rids[b]) * case.host_stride + token)
                destinations.append(dst)
                valid += 1
    if valid:
        si = c4_indices(torch, torch.tensor(sources))
        di = c4_indices(torch, torch.tensor(destinations))
        if not torch.equal(actual.view(-1)[di], case.host.view(-1)[si]):
            raise ValueError("Full value/scale oracle mismatch")
        mask[di] = True
    if not torch.equal(actual.view(-1)[~mask], before.view(-1)[~mask]):
        raise ValueError("Unreferenced destination/guard/padding modified")
    # Physical locations must remain a permutation even after scratch rotation.
    if sorted(case.locs.cpu().view(-1).tolist()) != sorted(
        case.initial_locs.cpu().view(-1).tolist()
    ):
        raise ValueError("Physical slot mapping lost/duplicated")
    return dict(
        valid_occurrences=valid,
        all_value_scale_bytes=True,
        untouched_guards_padding=True,
        physical_slot_permutation=True,
    )


def check_one(torch, args, report, width, k, hot, tail):
    from sglang.benchmark.hisparse_pr40670_reference import (
        load_cache_to_device_buffer_spec_mla,
    )
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

    case = make_case(torch, width, k, hot, tail)
    row = dict(
        width=width,
        top_k=k,
        hot=hot,
        tail=tail,
        scratch=case.layout.scratch_slots,
        checks=[],
    )
    report["cases"].append(row)
    try:
        snapshot = process_snapshot(
            buffers=[
                dict(
                    name="pr_reference_host",
                    address=case.allocation.data_ptr(),
                    nbytes=case.allocation.numel(),
                )
            ]
        )
        row["host_snapshot"] = snapshot
        if not host_local(snapshot, case.allocation.numel()):
            raise ValueError("Host pool not fully bound to NUMA1")
        before = reset(torch, case)
        set_input(torch, case, "mixed")
        load_cache_to_device_buffer_spec_mla(**case.args)
        torch.cuda.synchronize()
        row["eager"] = oracle(torch, case, before)
        stream = torch.cuda.Stream(device=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if torch.cuda.current_stream().device.index != 0:
                raise ValueError("Capture device mismatch")
            load_cache_to_device_buffer_spec_mla(**case.args)
        for kind in (
            "cold",
            "duplicates",
            "warm",
            "mixed",
            "changed_host",
            "zero_real",
        ):
            if kind == "changed_host":
                case.host.bitwise_xor_(255)
            if kind != "warm":
                reset(torch, case)
            before = case.output.cpu()
            set_input(torch, case, kind)
            graph.replay()
            torch.cuda.synchronize()
            row["checks"].append(dict(phase=kind, **oracle(torch, case, before)))
        reset(torch, case)
        set_input(torch, case, "cold")
        before = case.output.cpu()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            graph.replay()
            torch.cuda.synchronize()
        trace_path = (
            args.output_dir
            / f"w{width}-k{k}-scratch{case.layout.scratch_slots}.trace.json"
        )
        prof.export_chrome_trace(str(trace_path))
        row["profile"] = validate_profile(
            json.loads(trace_path.read_text()), 4, width, k
        )
        row["profile_oracle"] = oracle(torch, case, before)
        row["success"] = True
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(case.allocation)


def collect_codegen(args, report):
    import subprocess

    paths = {
        f[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(f := line.split(None, 5)) == 6
    }
    found = sorted(
        p
        for p in paths
        if "sgl_kernel_jit_hisparse_pr40670_reference_" in Path(p).name
        and "_reference_state_" not in Path(p).name
    )
    report["artifacts"] = []
    if len(found) != 2:
        raise ValueError(f"Expected two swap JIT specializations, got {found}")
    for i, path in enumerate(found):
        r = dict(path=path, sha256=sha256(path))
        report["artifacts"].append(r)
        for kind, option in (
            ("sass", "--dump-sass"),
            ("resources", "--dump-resource-usage"),
        ):
            run = subprocess.run(
                [args.cuobjdump, option, path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (args.output_dir / f"jit{i}.{kind}.txt").write_text(run.stdout)
            (args.output_dir / f"jit{i}.{kind}.stderr.txt").write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(run.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cuobjdump", default="cuobjdump")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        success=False,
        mode="reference_check",
        pr_commit=COMMIT,
        cases=[],
        performance_ready=False,
        service_integrated=False,
    )
    observer = None
    try:
        report["source_manifest"] = source_manifest()
        report["code_sha256"] = {
            str(f.relative_to(ROOT)): sha256(f)
            for f in [
                Path(__file__),
                ROOT / "python/sglang/benchmark/hisparse_pr40670_reference.py",
                ROOT / "python/sglang/benchmark/hisparse_pr40670_layout.py",
                ROOT / "python/sglang/benchmark/hisparse_pr40670_check.py",
            ]
        }
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("Require NVIDIA CUDA")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        observer = Observer(torch, [0])
        report["identity"] = observer.metadata[0]
        if report["identity"]["uuid"] != "GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702":
            raise ValueError("Require physical GPU4")
        for config in [(2, 4, 16, 16), (2, 4, 16, 2), (6, 16, 128, 16)]:
            report["active"] = config
            check_one(torch, args, report, *config)
        collect_codegen(args, report)
        report["state"] = observer.sample(0)
        report["success"] = (
            all(c.get("success") for c in report["cases"]) and len(report["cases"]) == 3
        )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.nv.nvmlShutdown()
        (args.output_dir / "reference.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        lines = [
            f"REFERENCE success={report['success']} cases={len(report['cases'])}/3 performance_ready=False service_integrated=False",
            f"PR commit={COMMIT} IDENTITY {report.get('identity')}",
            "CONFIG gather_threads=64 commit_threads=512 RecordMissPlan=False PDL=architecture_default; no independent copy launch",
        ]
        for c in report["cases"]:
            lines.append(
                f"CASE W{c['width']} K{c['top_k']} H{c['hot']} scratch={c['scratch']} success={c.get('success', False)} eager={bool(c.get('eager'))} graph={len(c['checks'])}/6 profile={c.get('profile')}"
            )
        lines.append(
            f"CODEGEN artifacts={len(report.get('artifacts', []))}/2 {report.get('artifacts')}"
        )
        if "error" in report:
            lines.append("ERROR " + report["error"])
        text = "\n".join(lines) + "\n"
        (args.output_dir / "reference-compact.txt").write_text(text)
        print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
