"""Independent legacy CTA4 readers or SM/CE: single, serial and concurrent execution."""

import argparse
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from bench_h2d_copy import validate_sm
from bench_h2d_latency import host_local
from bench_h2d_pair import Observer, Warmer, make_plan
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_ce_pressure import compact_pressure, execute_pressure
from sglang.benchmark.hisparse_h2d_diagnostics import (
    PAGE_BYTES,
    copy_geometry,
    process_snapshot,
)
from sglang.benchmark.hisparse_h2d_grid_perf import loaded_grid_artifact
from sglang.benchmark.hisparse_h2d_streams import activity_overlap
from sglang.benchmark.hisparse_h2d_sm_ce import CEReader, mixed_overlap

METHODS = ("A", "B", "serial_AB", "serial_BA", "parallel_AB", "parallel_BA")


class Reader:
    def __init__(self, torch, host, source, plan, banks):
        from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla

        self.stream = torch.cuda.Stream(device=0)
        self.src, self.dst = plan["src"].cuda(), plan["dst"].cuda()
        self.counts = torch.full((32,), 100, dtype=torch.int32, device="cuda")
        self.real = torch.tensor([32], dtype=torch.int32, device="cuda")
        self.output = torch.zeros(
            (plan["destination_pages"], PAGE_BYTES), dtype=torch.uint8, device="cuda"
        )
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.banks = banks
        self.payload_bytes = banks * 1868800

        def run(bank):
            copy_cache_planned_mla(
                miss_src=self.src[bank],
                miss_dst=self.dst,
                miss_count=self.counts,
                num_real_reqs=self.real,
                host_cache=source,
                device_buffer=self.output,
                item_size_bytes=584,
                num_blocks=4,
                block_size=1024,
                is_dsv4_layout=True,
                load_policy="legacy",
            )

        def check(bank):
            validate_sm(
                torch,
                host,
                self.output,
                plan["active_src"][bank],
                plan["dst"][:, :100],
                check_untouched=True,
            )

        # Reuse the exact production launch for flat captures of long windows.
        # Calling an existing CUDAGraph.replay() inside capture is unsupported.
        self.launch_bank = run
        self.check = check
        for bank in (0, banks - 1):
            self.output.zero_()
            run(bank)
            torch.cuda.synchronize()
            check(bank)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.start.record()
            for bank in range(banks):
                run(bank)
            self.end.record()
        self.output.zero_()
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        torch.cuda.synchronize()
        check(banks - 1)


def run(torch, readers, coordinator, method):
    names = [method] if method in ("A", "B") else list(method.rsplit("_", 1)[1])
    begin = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    started = time.monotonic_ns()
    with torch.cuda.stream(coordinator):
        begin.record()
        if method.startswith("parallel"):
            for name in names:
                r = readers[name]
                with torch.cuda.stream(r.stream):
                    r.stream.wait_event(begin)
                    r.graph.replay()
            # Both launches are enqueued before either completion is waited on.
            for name in names:
                coordinator.wait_event(readers[name].end)
        else:
            for name in names:
                readers[name].graph.replay()
        finish.record()
    finish.synchronize()
    wall = (time.monotonic_ns() - started) / 1000
    spans = {
        n: dict(
            start_us=begin.elapsed_time(readers[n].start) * 1000,
            end_us=begin.elapsed_time(readers[n].end) * 1000,
            duration_us=readers[n].start.elapsed_time(readers[n].end) * 1000,
        )
        for n in names
    }
    total = begin.elapsed_time(finish) * 1000
    if (
        not math.isfinite(total)
        or total <= 0
        or any(
            not all(math.isfinite(x) for x in s.values())
            or s["duration_us"] <= 0
            or s["start_us"] < 0
            or s["end_us"] > total + 1
            for s in spans.values()
        )
    ):
        raise ValueError(f"Invalid event intervals: {total}, {spans}")
    payload = sum(readers[n].payload_bytes for n in names)
    for n in names:
        spans[n]["payload_bytes"] = readers[n].payload_bytes
        spans[n]["GBps"] = readers[n].payload_bytes / spans[n]["duration_us"] / 1000
    return dict(
        method=method,
        total_us=total,
        wall_us=wall,
        payload_bytes=payload,
        GBps=payload / total / 1000,
        readers=spans,
        start_skew_us=abs(spans[names[0]]["start_us"] - spans[names[-1]]["start_us"]),
    )


def execute(torch, args, report):
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    observer = Observer(torch, [0])
    allocation = None
    registered = False
    try:
        report["identity"] = observer.metadata[0]
        if observer.metadata[0]["uuid"] != "GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702":
            raise ValueError("Require physical GPU4 as CUDA0")
        allocation = HostTensorAllocator().allocate((512 * 2**20,), torch.uint8, "cpu")
        allocation.zero_()
        rc = torch.cuda.cudart().cudaHostRegister(
            allocation.data_ptr(), allocation.numel(), 1
        )
        if int(rc) != 0:
            raise RuntimeError(f"Portable registration failed: {rc}")
        registered = True
        torch.randint(
            0,
            256,
            allocation.shape,
            dtype=torch.uint8,
            out=allocation,
            generator=torch.Generator().manual_seed(24001),
        )
        pages = allocation.numel() // PAGE_BYTES
        host = allocation[: pages * PAGE_BYTES].view(pages, PAGE_BYTES)
        snapshot = process_snapshot(
            buffers=[
                dict(
                    name="shared_pool",
                    address=allocation.data_ptr(),
                    nbytes=allocation.numel(),
                )
            ]
        )
        report["host_snapshot"] = snapshot
        report["host_node1"] = host_local(snapshot, allocation.numel())
        if not report["host_node1"]:
            raise ValueError("Full allocation must be bound to node1")
        banks = 4 if args.mode == "smoke" and not args.pressure else 64
        geometry = copy_geometry(32, 100, (pages // 2) * PAGE_BYTES, banks)
        config = SimpleNamespace(
            batch=32, plan_stride=3072, buffer_slots=3072, seed=24001
        )
        plans = {}
        for i, name in enumerate(("A", "B")):
            g = dict(geometry, random_token_start=i * (pages // 2) * 64)
            plans[name] = make_plan(torch, config, g, i, 100)
        if int(plans["A"]["active_src"].max()) >= int(plans["B"]["active_src"].min()):
            raise ValueError("Source ranges overlap")
        # Prepare device source outside every capture and timing interval.
        source = host if args.source == "host" else host.to("cuda:0")
        torch.cuda.synchronize()
        if args.source == "hbm" and (
            source.device != torch.device("cuda:0")
            or source.shape != host.shape
            or source.stride() != host.stride()
            or not torch.equal(source.cpu(), host)
        ):
            raise ValueError("HBM source binding/content mismatch")
        readers = {}
        coordinator = torch.cuda.Stream(device=0)
        warmer = Warmer(torch, 0)
        ce_bytes = 16 * 2**20
        for name in ("A", "B"):
            if args.partner == "ce" and name == "B":
                # A uses the first half of the registered allocation. CE uses
                # an entirely disjoint contiguous range in the second half.
                ce_start = (pages // 2) * PAGE_BYTES
                ce_source = allocation[ce_start : ce_start + ce_bytes]
                if (
                    int(plans["A"]["active_src"].max()) >= (pages // 2) * 64
                    or ce_source.numel() != ce_bytes
                ):
                    raise ValueError("SM/CE source range binding failed")
                pilot = CEReader(torch, ce_source, 1)
                warmer.run(250)
                pilot_readers = {"A": readers["A"], "B": pilot}
                for method in ("A", "B"):
                    for _ in range(3):
                        run(torch, pilot_readers, coordinator, method)
                # Persist samples before any calibration gate can raise.
                report["pilot"] = dict(samples_us={"A": [], "B": []})
                for n in ("A", "B"):
                    for _ in range(5):
                        report["pilot"]["samples_us"][n].append(
                            run(torch, pilot_readers, coordinator, n)["readers"][n][
                                "duration_us"
                            ]
                        )
                pilot_us = {
                    n: statistics.median(v)
                    for n, v in report["pilot"]["samples_us"].items()
                }
                ce_copies = (
                    1 if args.pressure else max(1, round(pilot_us["A"] / pilot_us["B"]))
                )
                report["pilot"].update(
                    single_us=pilot_us,
                    ce_copies=ce_copies,
                    ce_single_GBps=ce_bytes / pilot_us["B"] / 1000,
                )
                if args.pressure:
                    # Fixed continuous CE calibration, not a single short pulse.
                    # No retry-until-pass and no threshold relaxation.
                    calibration = dict(
                        copies=16,
                        copy_bytes=ce_bytes,
                        warmups=3,
                        samples=[],
                        threshold_GBps=45,
                    )
                    report["ce_calibration_detail"] = calibration
                    bulk = CEReader(torch, ce_source, 16)
                    bulk_readers = {"B": bulk}
                    for _ in range(3):
                        run(torch, bulk_readers, coordinator, "B")
                    calibration["state_before"] = observer.sample(0)
                    for _ in range(5):
                        row = run(torch, bulk_readers, coordinator, "B")
                        calibration["samples"].append(row["readers"]["B"])
                    calibration["state_after"] = observer.sample(0)
                    calibration["median_GBps"] = statistics.median(
                        v["GBps"] for v in calibration["samples"]
                    )
                    report["ce_calibration"] = calibration["median_GBps"] >= 45
                    bulk.check(0)
                    del bulk_readers, bulk
                    if not report["ce_calibration"]:
                        raise ValueError(
                            f"Continuous CE calibration below 45 GB/s: median={calibration['median_GBps']:.3f}; "
                            f"samples={[v['GBps'] for v in calibration['samples']]}"
                        )
                if ce_copies > 256:
                    raise ValueError("Unexpected pilot duration ratio")
                readers[name] = CEReader(torch, ce_source, ce_copies)
                del pilot_readers, pilot
            else:
                readers[name] = Reader(torch, host, source, plans[name], banks)
            report["checks"].append(
                dict(
                    reader=name,
                    kind="CE" if args.partner == "ce" and name == "B" else "SM",
                    eager_banks=[0]
                    if args.partner == "ce" and name == "B"
                    else [0, banks - 1],
                    graph=True,
                    bytes_padding=True,
                    capture_device=readers[name].stream.device.index,
                )
            )
        if readers["A"].output.data_ptr() == readers["B"].output.data_ptr():
            raise ValueError("Outputs alias")
        report["bindings"] = dict(
            source_kind=args.source,
            source_device=str(source.device),
            source_ptr=source.data_ptr(),
            source_bytes=source.numel(),
            source_matches_host=True,
            source_disjoint=True,
            independent_plans=True if args.partner == "sm" else "CE has no plan",
            independent_outputs=True,
            readers={
                n: dict(
                    kind="CE" if args.partner == "ce" and n == "B" else "SM",
                    src=r.src.data_ptr() if hasattr(r, "src") else r.source.data_ptr(),
                    dst=r.dst.data_ptr() if hasattr(r, "dst") else None,
                    payload_bytes=r.payload_bytes,
                    output=r.output.data_ptr(),
                    stream=int(r.stream.cuda_stream),
                )
                for n, r in readers.items()
            },
        )
        report["jit"] = loaded_grid_artifact()
        report["config"] = dict(
            source=args.source,
            partner=args.partner,
            labels={"A": "SM", "B": "CE" if args.partner == "ce" else "SM"},
            ce_copy_bytes=ce_bytes if args.partner == "ce" else None,
            ce_copies=readers["B"].banks if args.partner == "ce" else None,
            h2d_bytes_per_copy=1868800 if args.source == "host" else 0,
            throughput_kind="logical payload GB/s",
            banks=banks,
            copy_bytes=1868800,
            CTA_per_reader=4,
            threads=1024,
            host_register_flags=1,
            warm_ms=250,
            source_bytes_per_reader=geometry["host_bytes"],
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda),
        )
        # Validate concurrent and serial graphs with fresh zeroed targets.
        for method in METHODS:
            for r in readers.values():
                r.output.zero_()
            torch.cuda.synchronize()
            run(torch, readers, coordinator, method)
            names = [method] if method in ("A", "B") else list(method.rsplit("_", 1)[1])
            for name in names:
                readers[name].check(banks - 1)
            report["checks"].append(dict(method=method, bytes_padding=True))
        if args.pressure:
            execute_pressure(torch, args, report, readers, warmer, observer)
            return
        for trial in range(2 if args.mode == "smoke" else 6):
            order = METHODS if trial % 2 == 0 else tuple(reversed(METHODS))
            for method in order:
                warmer.run(250)
                for _ in range(2):
                    run(torch, readers, coordinator, method)
                before = observer.sample(0)
                row = run(torch, readers, coordinator, method)
                row.update(
                    trial=trial, state_before=before, state_after=observer.sample(0)
                )
                report["waves"].append(row)
        # Separate traces: exactly one two-reader execution, no warmup in capture.
        for method in ("serial_AB", "parallel_AB", "parallel_BA"):
            report["active_profile"] = method
            warmer.run(250)
            run(torch, readers, coordinator, method)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                with torch.profiler.record_function(f"two_readers:{method}"):
                    profiled = run(torch, readers, coordinator, method)
                torch.cuda.synchronize()
            path = args.output_dir / f"{method}.trace.json"
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            metrics = (
                mixed_overlap(
                    trace,
                    banks,
                    readers["B"].banks,
                    ce_bytes,
                    method.startswith("parallel"),
                )
                if args.partner == "ce"
                else activity_overlap(trace, banks, method.startswith("parallel"))
            )
            report["profiles"].append(
                dict(
                    method=method,
                    events=profiled,
                    activity=metrics,
                    trace=path.name,
                    sha256=sha256(path),
                )
            )
        report["success"] = True
        report["telemetry_valid"] = all(
            not w[k]["errors"]
            and w[k].get("pstate") == 0
            and w[k].get("pcie_gen") == 5
            and w[k].get("pcie_width") == 16
            for w in report["waves"]
            for k in ("state_before", "state_after")
        )
        report["results_valid"] = report["telemetry_valid"]
        if args.partner == "ce":
            solo = {
                n: statistics.median(
                    w["readers"][n]["duration_us"]
                    for w in report["waves"]
                    if w["method"] == n
                )
                for n in ("A", "B")
            }
            report["duration_match"] = solo["B"] / solo["A"]
            report["ce_calibration"] = all(
                w["readers"]["B"]["GBps"] >= 45
                for w in report["waves"]
                if w["method"] == "B"
            )
            # Main must compare workloads of similar solo durations. Smoke is
            # allowed a short SM graph relative to one 16 MiB CE copy.
            report["results_valid"] &= report["ce_calibration"] and (
                args.mode == "smoke" or 0.8 <= report["duration_match"] <= 1.25
            )
    finally:
        if registered:
            torch.cuda.synchronize()
            _cuda_host_unregister(allocation)
        observer.nv.nvmlShutdown()


def compact(r):
    if r.get("pressure"):
        return compact_pressure(r)
    lines = [
        f"source={r['source']} mode={r['mode']} success={r['success']} results_valid={r['results_valid']} checks={len(r['checks'])}/8 profiles={len(r['profiles'])}/3",
        f"IDENTITY {r.get('identity')} HOST node1={r.get('host_node1')} flags=1",
        f"BINDINGS {r.get('bindings')}",
        f"JIT {r.get('jit')}",
        f"CONFIG {r.get('config')}",
    ]
    if r.get("partner") == "ce":
        lines.append(
            f"MIXED A=SM B=CE pilot={r.get('pilot')} duration_match={r.get('duration_match')} CE_calibration={r.get('ce_calibration')}"
        )
    for method in METHODS:
        rows = [w for w in r["waves"] if w["method"] == method]
        if rows:
            lines.append(
                f"TOTAL {method} us={[round(w['total_us'], 3) for w in rows]} GBps={[round(w['GBps'], 3) for w in rows]} start_skew_us={[round(w['start_skew_us'], 3) for w in rows]}"
            )
            lines.append(
                f"READERS {method} duration_us="
                + str(
                    {
                        n: [round(w["readers"][n]["duration_us"], 3) for w in rows]
                        for n in rows[0]["readers"]
                    }
                )
            )
            lines.append(
                f"READER_GBPS {method} "
                + str(
                    {
                        n: [round(w["readers"][n]["GBps"], 3) for w in rows]
                        for n in rows[0]["readers"]
                    }
                )
            )
            lines.append(
                f"STATE {method} before/after="
                + str(
                    [
                        {
                            k: [
                                w[side].get(k)
                                for side in ("state_before", "state_after")
                            ]
                            for k in (
                                "sm_MHz",
                                "power_W",
                                "temperature_C",
                                "throttle_bits",
                            )
                        }
                        for w in rows
                    ]
                )
            )
    for p in r["profiles"]:
        lines.append(f"ACTIVITY {p['method']} {p['activity']}")
    if r["results_valid"] and r["mode"] == "main":
        for order in ("AB", "BA"):
            rows = {
                m: {w["trial"]: w for w in r["waves"] if w["method"] == m}
                for m in METHODS
            }
            speedup = [
                rows[f"serial_{order}"][t]["total_us"]
                / rows[f"parallel_{order}"][t]["total_us"]
                for t in range(6)
            ]
            lines.append(
                f"RATIO serial/parallel_{order}={speedup} median={statistics.median(speedup)}"
            )
    if r["results_valid"] and r.get("partner") == "ce":
        for order in ("AB", "BA"):
            concurrent = [w for w in r["waves"] if w["method"] == f"parallel_{order}"]
            for n in ("A", "B"):
                solo = {w["trial"]: w for w in r["waves"] if w["method"] == n}
                ratios = [
                    w["readers"][n]["GBps"] / solo[w["trial"]]["readers"][n]["GBps"]
                    for w in concurrent
                ]
                lines.append(
                    f"RETENTION parallel_{order} {n} full_lifetime_GBps_ratio={ratios} median={statistics.median(ratios)}"
                )
    if "error" in r:
        lines.append("ERROR " + r["error"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "main"], required=True)
    p.add_argument("--pressure", action="store_true")
    p.add_argument("--partner", choices=["sm", "ce"], default="sm")
    p.add_argument("--source", choices=["host", "hbm"], default="host")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.pressure and (args.partner != "ce" or args.source != "host"):
        p.error("Pressure mode requires host SM/CE")
    if args.partner == "ce" and args.source != "host":
        p.error("Mixed SM/CE requires host source")
    if args.output_dir.exists():
        p.error("Use new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        mode=args.mode,
        source=args.source,
        partner=args.partner,
        pressure=args.pressure,
        success=False,
        results_valid=False,
        checks=[],
        waves=[],
        profiles=[],
    )
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("NVIDIA CUDA required")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        root = Path(__file__).resolve().parents[2]
        report["code"] = {
            s: sha256(root / s)
            for s in (
                "scripts/pd_test/bench_h2d_streams.py",
                "python/sglang/benchmark/hisparse_h2d_streams.py",
                "python/sglang/benchmark/hisparse_h2d_sm_ce.py",
                "python/sglang/benchmark/hisparse_h2d_ce_pressure.py",
                "python/sglang/kernels/jit/csrc/hisparse.cuh",
            )
        }
        execute(torch, args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        (args.output_dir / "streams.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        text = compact(report)
        (args.output_dir / "streams-compact.txt").write_text(text)
        print(text)
    return 0 if report["results_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
