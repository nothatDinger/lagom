"""Opt-in bounded DSPARK/Lagom diagnostics, including captured CUDA graphs.

Set SGLANG_LAGOM_PERF_TRACE to a JSON config path before server startup. Timing
markers are GPU kernels writing %globaltimer into a fixed per-step ring, so
replay does not overwrite the previous step's CUDA event timestamps. Never
use this instrumented server to measure throughput.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from sglang.srt.observability.lagom_decode_perf import get_meter


@functools.lru_cache(maxsize=1)
def get_trace():
    config = os.getenv("SGLANG_LAGOM_PERF_TRACE")
    return StepTrace(json.loads(Path(config).read_text())) if config else None


def phase(name, *, layer=None, slot=None):
    trace = get_trace()
    return trace.region(name, layer=layer, slot=slot) if trace else nullcontext()


def timed(name, *, layer=False):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            trace = get_trace()
            if trace is None:
                return fn(self, *args, **kwargs)
            layer_id = (
                getattr(self, "layer_id", None)
                if layer is True
                else kwargs.get("layer_id")
            )
            if isinstance(layer, str):
                import inspect

                bound = inspect.signature(fn).bind_partial(self, *args, **kwargs)
                layer_id = getattr(bound.arguments.get(layer), "layer_id", None)
            with trace.region(name, layer=layer_id):
                return fn(self, *args, **kwargs)

        return wrapped

    return decorate


def model_role(role):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            trace = get_trace()
            if trace is None:
                return fn(*args, **kwargs)
            previous, trace.role = trace.role, role
            try:
                return fn(*args, **kwargs)
            finally:
                trace.role = previous

        return wrapped

    return decorate


def verify_step(fn):
    @functools.wraps(fn)
    def wrapped(self, batch, *args, **kwargs):
        trace = get_trace()
        if trace is None:
            return fn(self, batch, *args, **kwargs)
        if batch.forward_mode.is_idle():
            trace.pause()
            return fn(self, batch, *args, **kwargs)
        trace.begin(batch)
        try:
            with trace.region("step"):
                result = fn(self, batch, *args, **kwargs)
            trace.finish(result)
            return result
        except Exception:
            trace.slot.fill_(-1)
            raise

    return wrapped


def prefetch_slot(layer_id):
    trace = get_trace()
    if trace is None:
        return None
    slot = trace.side_slots.get(layer_id)
    if slot is None:
        slot = trace.torch.full_like(trace.slot, -1)
        trace.side_slots[layer_id] = slot
    slot.copy_(trace.slot)
    return slot


def note_draft_graph(can_run_graph):
    trace = get_trace()
    if trace is not None and trace.active >= 0:
        trace.records[-1]["draft_cuda_graph"] = bool(can_run_graph)
    meter = get_meter()
    if meter is not None and meter.active is not None:
        meter.records[-1]["draft_cuda_graph"] = bool(can_run_graph)


def pause_trace():
    trace = get_trace()
    if trace is not None:
        trace.pause()


def union_ns(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative trace interval")
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def summarize(regions, steps):
    """Intervals are inclusive stream spans, never an additive GPU breakdown."""
    rows = []
    for step in range(steps):
        row = {"step": step}
        for kind in sorted({r["name"] for r in regions}):
            intervals = [
                (r["start_ns"], r["end_ns"])
                for r in regions
                if r["step"] == step and r["name"] == kind
            ]
            row[kind + "_ms"] = union_ns(intervals) / 1e6
        # Without CUPTI activity, a stream timestamp gap cannot distinguish
        # host starvation, D2H, launch latency, NCCL or uninstrumented kernels.
        row["gpu_wait_cpu_ms"] = None
        current = [r for r in regions if r["step"] == step and r["name"] == "step"]
        previous = [r for r in regions if r["step"] == step - 1 and r["name"] == "step"]
        row["inter_step_gap_ms"] = (
            max(0, current[0]["start_ns"] - previous[0]["end_ns"]) / 1e6
            if current and previous
            else None
        )
        rows.append(row)
    return rows


class StepTrace:
    MAX_REGIONS = 2048

    def __init__(self, config):
        import torch

        from sglang.srt.observability.lagom_perf_kernels import stamp

        self.torch, self.stamp = torch, stamp
        self.steps, self.skip = int(config.get("steps", 12)), int(config.get("skip", 2))
        if not 0 < self.steps <= 256 or self.skip < 0:
            raise ValueError("trace steps must be 1..256 and skip nonnegative")
        self.nsys = bool(config.get("nsys", False))
        self.required_batch_size = config.get("required_batch_size")
        if self.required_batch_size is not None and self.required_batch_size <= 0:
            raise ValueError("required_batch_size must be positive")
        self.nsys_running = False
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.cuda.current_device()
        self.slot = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        self.stamps = torch.zeros(
            (self.MAX_REGIONS, self.steps, 2), dtype=torch.int64, device="cuda"
        )
        self.names, self.side_slots, self.records, self.cpu = {}, {}, [], []
        self.accepted, self.prefix_lens_gpu, self.events = {}, {}, []
        self.role, self.seen, self.active, self.written = "target", 0, -1, False
        # Compile before graph capture. No synchronization is added to a
        # recorded verify step; collection happens after the bounded window.
        self.stamp[(1,)](self.stamps, self.slot, self.steps, 0, 0)
        self.stamp[(1,)](self.stamps, self.slot, self.steps, 0, 1)

    @contextmanager
    def region(self, name, *, layer=None, slot=None):
        full_name = (
            name
            if name
            in (
                "step",
                "draft",
                "target_verify",
                "accept",
                "cpu_dependency_commit",
                "kv_commit",
                "kv_prepare",
                "draft_inject",
            )
            else f"{self.role}.{name}"
        )
        key = (full_name, layer)
        if key not in self.names:
            if len(self.names) >= self.MAX_REGIONS:
                raise RuntimeError("Lagom trace region capacity exceeded")
            self.names[key] = len(self.names)
        region = self.names[key]
        target_slot = self.slot if slot is None else slot
        self.stamp[(1,)](self.stamps, target_slot, self.steps, region, 0)
        cpu_start = time.perf_counter_ns()
        live = self.active >= 0 and not self.torch.cuda.is_current_stream_capturing()
        if live:
            self.torch.cuda.nvtx.range_push(
                f"lagom/{full_name}/step={self.active}/layer={layer}"
            )
        try:
            yield
        finally:
            self.stamp[(1,)](self.stamps, target_slot, self.steps, region, 1)
            if live:
                self.torch.cuda.nvtx.range_pop()
                self.cpu.append(
                    {
                        "step": self.active,
                        "name": full_name,
                        "layer": layer,
                        "start_ns": cpu_start,
                        "end_ns": time.perf_counter_ns(),
                    }
                )

    def begin(self, batch):
        self.poll()
        if (
            self.required_batch_size is not None
            and len(batch.reqs) != self.required_batch_size
        ):
            self.active = -1
            self.slot.fill_(-1)
            return
        self.active = self.seen - self.skip
        self.seen += 1
        if not 0 <= self.active < self.steps:
            self.active = -1
        self.slot.fill_(self.active)
        if self.active == 0 and self.nsys and self.device == 0:
            self.torch.cuda.cudart().cudaProfilerStart()
            self.nsys_running = True
        if self.active >= 0:
            prefix_lens_cpu = batch.seq_lens_cpu
            if prefix_lens_cpu is None:
                # Plain DSPARK can keep lengths only on device. Preserve the
                # pre-forward values before the batch reuses/mutates them, but
                # defer D2H until the sampled window's completion events fire.
                self.prefix_lens_gpu[self.active] = batch.seq_lens.detach().clone()
            self.records.append(
                {
                    "step": self.active,
                    "forward_iter": int(batch.forward_iter),
                    "batch_size": len(batch.reqs),
                    "request_ids": [r.rid for r in batch.reqs],
                    "prefix_lens": (
                        prefix_lens_cpu.tolist()
                        if prefix_lens_cpu is not None
                        else None
                    ),
                }
            )

    def pause(self):
        # Prefill/idle must not overwrite the last decode's per-layer stamps.
        # Outstanding prefetches use their own stable per-layer slot scalars.
        self.poll()
        self.active = -1
        self.slot.fill_(-1)

    def finish(self, result):
        if self.active < 0:
            return
        self.records[-1]["cuda_graph"] = bool(result.can_run_cuda_graph)
        # Preserve accept counts without D2H inside the timed window.
        self.accepted[self.active] = result.accept_lens.detach().clone()
        if self.active == self.steps - 1:
            event = self.torch.cuda.Event()
            event.record()
            self.events.append(event)
            # Side streams may still write this window after the main ends.
            # Transfer streams are registered explicitly below; no global join.
            for stream in getattr(self, "streams", []):
                event = self.torch.cuda.Event()
                event.record(stream)
                self.events.append(event)

    def poll(self):
        if self.written or not self.events or not all(e.query() for e in self.events):
            return
        if self.nsys_running:
            self.torch.cuda.cudart().cudaProfilerStop()
            self.nsys_running = False
        values = self.stamps.cpu().tolist()
        regions = []
        for (name, layer), index in self.names.items():
            for step, (start, end) in enumerate(values[index]):
                if start > 0 and end >= start:
                    regions.append(
                        {
                            "name": name,
                            "layer": layer,
                            "step": step,
                            "start_ns": start,
                            "end_ns": end,
                        }
                    )
        for record in self.records:
            if record["prefix_lens"] is None:
                record["prefix_lens"] = (
                    self.prefix_lens_gpu[record["step"]].cpu().tolist()
                )
            record["accepted_tokens"] = self.accepted[record["step"]].cpu().tolist()
        self.written = True
        pid = os.getpid()
        stem = self.output / f"rank-device{self.device}-pid{pid}"
        payload = {
            "schema_version": 1,
            "device": self.device,
            "records": self.records,
            "regions": regions,
            "cpu_regions": self.cpu,
            "summary": summarize(regions, self.steps),
            "notes": [
                "GPU spans include queue gaps and nested regions; do not sum inclusive spans.",
                "CPU and GPU clocks use independent origins in Chrome export; use Nsight for precise correlation.",
                "gpu_wait_cpu_ms is unknown without CUPTI; cpu_dependency_commit is an upper-bound dependency span.",
                "prefetch_wait belongs to the consuming step; kv_prefetch belongs to the producing step.",
            ],
        }
        stem.with_suffix(".json").write_text(json.dumps(payload, indent=2))
        with stem.with_suffix(".steps.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0]))
            writer.writeheader()
            writer.writerows(payload["summary"])
        gpu_zero = min((r["start_ns"] for r in regions), default=0)
        cpu_zero = min((r["start_ns"] for r in self.cpu), default=0)
        events = []
        for track_pid, kind, source, zero in (
            (2 * self.device, "GPU", regions, gpu_zero),
            (2 * self.device + 1, "CPU (separate clock)", self.cpu, cpu_zero),
        ):
            events.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": track_pid,
                    "tid": 0,
                    "args": {"name": f"{kind} device{self.device}"},
                }
            )
            tracks = {}
            for r in source:
                side = "prefetch" if r["name"].endswith("kv_prefetch") else "main"
                track = (side, r["layer"])
                if track not in tracks:
                    tracks[track] = len(tracks) + 1
                    events.append(
                        {
                            "ph": "M",
                            "name": "thread_name",
                            "pid": track_pid,
                            "tid": tracks[track],
                            "args": {"name": f"{side}/layer{r['layer']}"},
                        }
                    )
                events.append(
                    {
                        "ph": "X",
                        "name": r["name"],
                        "cat": kind,
                        "pid": track_pid,
                        "tid": tracks[track],
                        "ts": (r["start_ns"] - zero) / 1000,
                        "dur": (r["end_ns"] - r["start_ns"]) / 1000,
                        "args": {"step": r["step"]},
                    }
                )
        stem.with_suffix(".trace.json").write_text(json.dumps({"traceEvents": events}))


def register_side_stream(stream):
    trace = get_trace()
    if trace is not None:
        if not hasattr(trace, "streams"):
            trace.streams = []
        if stream not in trace.streams:
            trace.streams.append(stream)
