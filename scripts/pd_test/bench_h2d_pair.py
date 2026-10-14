"""Two GPUs read one pinned allocation sequentially, with ABBA order and telemetry.

Only selected GPUs are used. No clock, power, driver or system policies are set.
The portable registration flag is explicit because the allocation is shared.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

from bench_h2d_copy import capture_wave, validate_sm
from sglang.benchmark.hisparse_h2d_diagnostics import (
    ITEM_BYTES,
    PAGE_BYTES,
    PAGE_TOKENS,
    copy_geometry,
    process_snapshot,
)
from sglang.benchmark.hisparse_h2d_pair import (
    SM_LOAD_POLICIES,
    abba_order,
    compact_report,
    copy_methods,
    pci_path,
    source_kind,
    validate_profile,
)


class Observer:
    def __init__(self, torch, devices):
        import pynvml as nv

        self.nv = nv
        nv.nvmlInit()
        self.handles, self.metadata = {}, {}
        for device in devices:
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            normalized = uuid.lower().removeprefix("gpu-").replace("-", "")
            matches = []
            for physical in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(physical)
                actual = nv.nvmlDeviceGetUUID(handle)
                actual = actual.decode() if isinstance(actual, bytes) else actual
                if actual.lower().removeprefix("gpu-").replace("-", "") == normalized:
                    matches.append((physical, handle, actual))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Cannot map CUDA device {device} UUID={uuid} to NVML"
                )
            physical, handle, actual = matches[0]
            self.handles[device] = handle
            bus = nv.nvmlDeviceGetPciInfo(handle).busId
            self.metadata[device] = dict(
                physical_index=physical,
                uuid=actual,
                pci_bus=bus.decode() if isinstance(bus, bytes) else bus,
            )
            self.metadata[device]["pci_path"] = pci_path(
                self.metadata[device]["pci_bus"]
            )

    def sample(self, device):
        nv, handle = self.nv, self.handles[device]
        result = dict(begin_ns=time.monotonic_ns(), errors={})
        reads = dict(
            sm_MHz=lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
            mem_MHz=lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
            pstate=lambda: nv.nvmlDeviceGetPerformanceState(handle),
            pcie_gen=lambda: nv.nvmlDeviceGetCurrPcieLinkGeneration(handle),
            pcie_width=lambda: nv.nvmlDeviceGetCurrPcieLinkWidth(handle),
            throttle_bits=lambda: hex(
                nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            ),
            power_W=lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000,
            power_limit_W=lambda: nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
            temperature_C=lambda: nv.nvmlDeviceGetTemperature(
                handle, nv.NVML_TEMPERATURE_GPU
            ),
        )
        for key, read in reads.items():
            try:
                result[key] = read()
            except Exception as exc:
                result["errors"][key] = str(exc)
        result["end_ns"] = time.monotonic_ns()
        return result


class Warmer:
    def __init__(self, torch, device):
        self.torch = torch
        with torch.cuda.device(device):
            self.a = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")
            self.b = torch.randn_like(self.a)
            self.c = torch.empty_like(self.a)
            torch.mm(self.a, self.b, out=self.c)  # Prime cuBLAS outside measurements.
            torch.cuda.synchronize()

    def run(self, milliseconds):
        deadline = time.monotonic() + milliseconds / 1000
        while time.monotonic() < deadline:
            for _ in range(8):
                self.torch.mm(self.a, self.b, out=self.c)
            self.torch.cuda.current_stream().synchronize()


def make_plan(torch, args, geometry, case_index, misses):
    rng = torch.Generator().manual_seed(args.seed + case_index + 1)
    banks, entries = geometry["banks"], geometry["entries"]
    active = torch.randperm(geometry["host_tokens"], generator=rng)[
        : banks * entries
    ].reshape(banks, args.batch, misses)
    src = torch.zeros((banks, args.batch, args.plan_stride), dtype=torch.int64)
    src[:, :, :misses] = active
    padded = ((args.buffer_slots + PAGE_TOKENS - 1) // PAGE_TOKENS + 1) * PAGE_TOKENS
    dst = torch.full((args.batch, args.plan_stride), -1, dtype=torch.int32)
    for req in range(args.batch):
        dst[req, :misses] = (
            torch.randperm(args.buffer_slots, generator=rng)[:misses] + req * padded
        )
    return dict(
        src=src,
        active_src=active,
        dst=dst,
        destination_pages=args.batch * padded // PAGE_TOKENS,
    )


def bank_destination_plan(plan):
    """Give both orders equally sized per-bank destination arrays."""
    return dict(
        plan,
        dst=plan["dst"].unsqueeze(0).expand(plan["src"].shape[0], -1, -1).clone(),
    )


def order_source_plan(torch, plan, misses):
    """Sort each bank globally, carrying absolute destinations with sources.

    The kernel consumes absolute destination slots, not request-relative ones.
    Sorting may move a pair to another request row but keeps miss counts fixed.
    Check the complete source-to-destination mapping independently by destination.
    """
    active = plan["active_src"]
    if active.ndim != 3 or active.shape[2] != misses or plan["dst"].ndim != 2:
        raise ValueError("Source ordering requires the original uniform bank plan")
    banks, batch, _ = active.shape
    flat_src = active.reshape(banks, -1)
    flat_dst = plan["dst"][:, :misses].reshape(1, -1).expand(banks, -1)
    if flat_dst[0].unique().numel() != flat_dst.shape[1]:
        raise ValueError("Source ordering requires unique destination slots")
    sorted_src, order = flat_src.sort(dim=1)
    sorted_dst = flat_dst.gather(1, order)
    # Independent of the source-sort permutation: destinations identify pairs.
    original_dst, original_order = flat_dst.sort(dim=1)
    actual_dst, actual_order = sorted_dst.sort(dim=1)
    if not torch.equal(original_dst, actual_dst) or not torch.equal(
        flat_src.gather(1, original_order), sorted_src.gather(1, actual_order)
    ):
        raise ValueError("Source ordering changed the source/destination mapping")
    src = plan["src"].clone()
    src[:, :, :misses] = sorted_src.reshape(banks, batch, misses)
    dst = plan["dst"].unsqueeze(0).expand(banks, -1, -1).clone()
    dst[:, :, :misses] = sorted_dst.reshape(banks, batch, misses)
    return dict(
        src=src,
        active_src=src[:, :, :misses],
        dst=dst,
        destination_pages=plan["destination_pages"],
        ordering_check=dict(
            all_banks_checked=True,
            pairs_preserved=True,
            entries_per_bank=flat_src.shape[1],
            banks=banks,
        ),
    )


class CopyCase:
    def __init__(
        self,
        torch,
        host,
        device,
        label,
        method,
        nbytes,
        banks,
        args,
        plan=None,
        misses=None,
        device_source=None,
        shared_sm=None,
        oracle_plan=None,
    ):
        self.device, self.label, self.method = device, label, method
        self.nbytes, self.banks = nbytes, banks
        self.host = host
        self.source_kind = source_kind(method)
        self.functions = []
        with torch.cuda.device(device):
            if method in SM_LOAD_POLICIES:
                from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla

                source = host
                if self.source_kind == "device":
                    if (
                        device_source is None
                        or device_source.device != torch.device("cuda", device)
                        or device_source.shape != host.shape
                        or device_source.dtype != host.dtype
                        or not device_source.is_contiguous()
                    ):
                        raise ValueError(
                            "SM-HBM requires an equal-shaped source on the measured GPU"
                        )
                    source = device_source
                self.source = (
                    source  # Keep the source alive through every graph replay.
                )
                if method != "sm":
                    if shared_sm is None or (
                        shared_sm.device,
                        shared_sm.label,
                        shared_sm.banks,
                        shared_sm.nbytes,
                    ) != (device, label, banks, nbytes):
                        raise ValueError(
                            "SM variant requires the matching host-SM case"
                        )
                    # Share counts and destination allocation. HBM/cache variants
                    # also share GPU plans; ordered SM replaces only paired plans.
                    src = shared_sm.plan_src
                    kwargs = dict(shared_sm.sm_kwargs, host_cache=source)
                    if method == "sm_ordered":
                        # Counts, source allocation and destination buffer stay
                        # identical to random SM; only the paired plan order changes.
                        src = plan["src"].to(device="cuda")
                        kwargs["miss_dst"] = plan["dst"].to(device="cuda")
                else:
                    src = plan["src"].to(device="cuda")
                    kwargs = dict(
                        miss_dst=plan["dst"].to(device="cuda"),
                        miss_count=torch.full(
                            (args.batch,), misses, dtype=torch.int32, device="cuda"
                        ),
                        num_real_reqs=torch.tensor(
                            [args.batch], dtype=torch.int32, device="cuda"
                        ),
                        host_cache=source,
                        device_buffer=torch.zeros(
                            (plan["destination_pages"], PAGE_BYTES),
                            dtype=torch.uint8,
                            device="cuda",
                        ),
                        item_size_bytes=ITEM_BYTES,
                        num_blocks=4,
                        block_size=1024,
                        is_dsv4_layout=True,
                    )
                kwargs["load_policy"] = SM_LOAD_POLICIES[method]
                self.plan_src, self.sm_kwargs = src, kwargs
                output = kwargs["device_buffer"]
                dst = kwargs["miss_dst"]
                copy_kwargs = {k: v for k, v in kwargs.items() if k != "miss_dst"}
                self.functions = [
                    lambda bank=bank: copy_cache_planned_mla(
                        miss_src=src[bank],
                        miss_dst=dst[bank] if dst.ndim == 3 else dst,
                        **copy_kwargs,
                    )
                    for bank in range(banks)
                ]

                def validate(bank):
                    # Ordered variants must produce the ORIGINAL mapping, not
                    # an oracle derived from the possibly faulty ordered plan.
                    reference = oracle_plan if oracle_plan is not None else plan
                    validate_sm(
                        torch,
                        host,
                        output,
                        reference["active_src"][bank],
                        reference["dst"][:, :misses],
                        check_untouched=getattr(args, "source_order_control", False),
                    )

            elif method == "ce":
                output = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                inputs = [
                    host.view(-1).narrow(0, bank * nbytes, nbytes)
                    for bank in range(banks)
                ]
                self.functions = [
                    lambda bank=bank: output.copy_(inputs[bank], non_blocking=True)
                    for bank in range(banks)
                ]

                def validate(bank):
                    if not torch.equal(output.cpu(), inputs[bank]):
                        raise RuntimeError("CE copy correctness failed")

            else:
                raise ValueError(f"Unknown copy method: {method}")

            self.verified_banks = sorted({0, banks - 1})
            for bank in self.verified_banks:
                output.zero_()
                self.functions[bank]()
                torch.cuda.synchronize()
                validate(bank)
            self.graph, self.events = capture_wave(
                torch, self.functions, stream=torch.cuda.Stream(device=device)
            )
            output.zero_()
            self.graph.replay()
            torch.cuda.synchronize()
            validate(banks - 1)
            self.graph_verified = True

    def warm(self, torch, phase, warmer, args):
        if phase == "compute":
            warmer.run(args.warm_ms)
        for _ in range(args.warmup):
            self.graph.replay()
        torch.cuda.synchronize()

    def measure(self, torch, observer, phase, trial):
        before = observer.sample(self.device)
        submitted = time.monotonic_ns()
        self.graph.replay()
        self.events[-1][1].synchronize()
        # Complete the graph before reading reusable events or switching GPUs.
        torch.cuda.current_stream().synchronize()
        completed = time.monotonic_ns()
        after = observer.sample(self.device)
        times = [start.elapsed_time(end) * 1000 for start, end in self.events]
        if any(t <= 0 or not math.isfinite(t) for t in times):
            raise RuntimeError("Non-positive graph event timing")
        return dict(
            case=self.label,
            method=self.method,
            load_policy=SM_LOAD_POLICIES.get(self.method, "CE"),
            source_kind=self.source_kind,
            h2d_bytes_per_copy=0 if self.source_kind == "device" else self.nbytes,
            device=self.device,
            phase=phase,
            trial=trial,
            payload_bytes=self.nbytes,
            banks=self.banks,
            copy_us=times,
            gpu_wave_us=self.events[0][0].elapsed_time(self.events[-1][1]) * 1000,
            before=before,
            after=after,
            host_submitted_ns=submitted,
            host_completed_ns=completed,
        )


def run_profiles(torch, args, cases, warmers, observer, report):
    report["profile_success"] = False
    path = args.output_dir / "pair.trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as profiler:
        for (label, method), devices in cases.items():
            for device in args.devices:
                case = devices[device]
                with torch.cuda.device(device):
                    case.warm(torch, "compute", warmers[device], args)
                    scope = f"h2d_pair:{label}:{method}:device{device}"
                    with torch.profiler.record_function(scope):
                        wave = case.measure(torch, observer, "profile_compute", 0)
                    wave["scope"] = scope
                    report["profile_waves"].append(wave)
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    validate_profile(report, events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--devices",
        type=int,
        nargs=2,
        default=[0, 1],
        help="CUDA logical IDs, expose physical 4,5",
    )
    parser.add_argument(
        "--register-device",
        type=int,
        default=0,
        help="CUDA context used for Portable registration",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--misses", type=int, nargs="+", default=[100, 1272])
    parser.add_argument("--plan-stride", type=int, default=3072)
    parser.add_argument("--buffer-slots", type=int, default=3072)
    parser.add_argument("--source-mib", type=int, default=512)
    parser.add_argument("--max-banks", type=int, default=64)
    parser.add_argument("--bulk-ce-mib", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warm-ms", type=float, default=250)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument(
        "--reverse-method-order",
        action="store_true",
        help="Reverse measurement case/method order; allocation and GPU ABBA order stay unchanged",
    )
    parser.add_argument(
        "--phases", nargs="+", choices=["copy", "compute"], default=["copy", "compute"]
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--source-order-control",
        action="store_true",
        help="Sort the SAME source/destination pairs by source address outside timing",
    )
    parser.add_argument(
        "--copy-load-policy-control",
        action="store_true",
        help="Compare legacy, explicit .ca and .cg source loads with shared plans and destination",
    )
    parser.add_argument(
        "--device-source-control",
        action="store_true",
        help="Also run the same SM kernel from local GPU memory; its payload rate is NOT H2D bandwidth",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.register_device not in args.devices:
        parser.error("Registration device must be one of --devices")
    order = abba_order(args.devices, args.epochs, args.reverse_order)
    if min(args.warmup, args.warm_ms, args.plan_stride, args.buffer_slots) <= 0 or max(
        args.misses
    ) > min(args.plan_stride, args.buffer_slots):
        parser.error(
            "Positive warmup/settings required; plan/buffer must cover every miss"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process, without torchrun")
    geometries = [
        copy_geometry(args.batch, m, args.source_mib * 2**20, args.max_banks)
        for m in args.misses
    ]
    host_bytes = geometries[0]["host_bytes"]
    if args.bulk_ce_mib <= 0:
        parser.error("Bulk CE size must be positive")
    bulk_bytes = args.bulk_ce_mib * 2**20
    bulk_banks = min(args.max_banks, host_bytes // bulk_bytes)
    if bulk_banks < 2:
        parser.error("Source must contain two bulk CE banks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "pair.json"
    if target.exists():
        parser.error("Use a new output directory; existing results are not replaced")

    import torch
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        _cuda_host_unregister,
    )

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA required")
    if min(args.devices) < 0 or max(args.devices) >= torch.cuda.device_count():
        parser.error("Expose both selected GPUs with CUDA_VISIBLE_DEVICES=4,5")
    torch.set_num_threads(1)
    observer = Observer(torch, args.devices)
    torch.cuda.set_device(args.register_device)
    report = dict(
        success=False,
        host_register_flags=1,
        devices=observer.metadata,
        config={
            **vars(args),
            "output_dir": str(args.output_dir),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "order": order,
        },
        waves=[],
        profile_waves=[],
        correctness=[],
    )
    host, registered = None, False
    try:
        host = HostTensorAllocator().allocate(
            (geometries[0]["host_pages"], PAGE_BYTES), torch.uint8, "cpu"
        )
        rc = torch.cuda.cudart().cudaHostRegister(
            host.data_ptr(), host.numel(), 1
        )  # cudaHostRegisterPortable
        if int(rc) != 0:
            raise RuntimeError(f"Portable cudaHostRegister failed: {rc}")
        registered = True
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(args.seed),
            out=host,
        )
        report["host_snapshot"] = process_snapshot(
            buffers=[
                dict(
                    name="shared_host_kv", address=host.data_ptr(), nbytes=host.numel()
                )
            ]
        )
        report["hugepage_setting"] = os.environ.get("SGLANG_HUGEPAGE_SIZE", "")
        device_sources = {}
        if args.device_source_control:
            # One identical full source per GPU, shared across miss shapes.
            # Setup H2D copies and synchronization are outside every timed wave.
            for device in args.devices:
                with torch.cuda.device(device):
                    device_sources[device] = host.to(device="cuda")
                    torch.cuda.synchronize()
            report["device_sources"] = [
                dict(device=device, address=source.data_ptr(), nbytes=source.numel())
                for device, source in device_sources.items()
            ]
        methods = copy_methods(vars(args))
        cases = {}
        for case_index, (misses, geometry) in enumerate(zip(args.misses, geometries)):
            plan = make_plan(torch, args, geometry, case_index, misses)
            label = f"B{args.batch}-miss{misses}"
            ordered_plan = None
            base_plan = plan
            if args.source_order_control:
                ordered_plan = order_source_plan(torch, plan, misses)
                base_plan = bank_destination_plan(plan)
                report.setdefault("ordered_plans", []).append(
                    dict(case=label, **ordered_plan["ordering_check"])
                )
            for method in methods:
                ordered = method in ("sm_ordered", "sm_hbm_ordered")
                anchor = "sm_ordered" if method == "sm_hbm_ordered" else "sm"
                cases[(label, method)] = {
                    device: CopyCase(
                        torch,
                        host,
                        device,
                        label,
                        method,
                        geometry["payload_bytes"],
                        geometry["banks"],
                        args,
                        ordered_plan if ordered else base_plan,
                        misses,
                        device_sources.get(device),
                        cases.get((label, anchor), {}).get(device),
                        plan,
                    )
                    for device in args.devices
                }
        label = f"bulk-ce-{args.bulk_ce_mib}MiB"
        cases[(label, "ce")] = {
            device: CopyCase(
                torch, host, device, label, "ce", bulk_bytes, bulk_banks, args
            )
            for device in args.devices
        }
        for (label, method), devices in cases.items():
            for device, case in devices.items():
                report["correctness"].append(
                    dict(
                        case=label,
                        method=method,
                        device=device,
                        source_kind=case.source_kind,
                        verified_banks=case.verified_banks,
                        all_payload_bytes_checked=True,
                        graph_replay_checked=case.graph_verified,
                        untouched_bytes_checked=(
                            args.source_order_control and method in SM_LOAD_POLICIES
                        ),
                    )
                )
        warmers = {device: Warmer(torch, device) for device in args.devices}
        if args.reverse_method_order:
            cases = dict(reversed(list(cases.items())))
        for phase in args.phases:
            for (label, method), devices in cases.items():
                for trial, device in enumerate(order):
                    with torch.cuda.device(device):
                        case = devices[device]
                        case.warm(torch, phase, warmers[device], args)
                        report["waves"].append(
                            case.measure(torch, observer, phase, trial)
                        )
                print(f"Completed {phase} {label} {method}", flush=True)
        if args.profile:
            run_profiles(torch, args, cases, warmers, observer, report)
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            (args.output_dir / "pair-compact.txt").write_text(compact_report(report))
        finally:
            if registered:
                for device in args.devices:
                    torch.cuda.synchronize(device)
                _cuda_host_unregister(host)
            observer.nv.nvmlShutdown()
    if args.profile and not report.get("profile_success"):
        raise SystemExit(
            "GPU activity validation incomplete; return pair-compact.txt, keep the raw trace"
        )


if __name__ == "__main__":
    main()
