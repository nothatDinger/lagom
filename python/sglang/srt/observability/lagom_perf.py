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
    kinds = sorted({r["name"] for r in regions})
    transfer_kinds = sorted({r["name"] for r in regions if "kv_entries" in r})
    for step in range(steps):
        row = {"step": step}
        step_regions = [r for r in regions if r["step"] == step]
        for kind in kinds:
            intervals = [
                (r["start_ns"], r["end_ns"])
                for r in step_regions
                if r["name"] == kind
            ]
            row[kind + "_ms"] = union_ns(intervals) / 1e6
        for kind in transfer_kinds:
            entries = [
                r["kv_entries"]
                for r in step_regions
                if r["name"] == kind and "kv_entries" in r
            ]
            row[kind + "_entries"] = sum(entries) if entries else None
        for region in step_regions:
            for field in ("resident_hits", "admissions"):
                if field in region:
                    key = region["name"] + "_" + field
                    row[key] = row.get(key, 0) + region[field]
        # Without CUPTI activity, a stream timestamp gap cannot distinguish
        # host starvation, D2H, launch latency, NCCL or uninstrumented kernels.
        row["gpu_wait_cpu_ms"] = None
        current = [r for r in step_regions if r["name"] == "step"]
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

        from sglang.srt.observability.lagom_perf_kernels import (
            count_forecast_copies,
            count_verify_copies,
            stamp,
            store_swap_in_counts,
        )

        self.torch, self.stamp = torch, stamp
        self.store_swap_in_counts = store_swap_in_counts
        self.count_verify_copies = count_verify_copies
        self.count_forecast_copies = count_forecast_copies
        self.steps, self.skip = int(config.get("steps", 12)), int(config.get("skip", 2))
        if not 0 < self.steps <= 256 or self.skip < 0:
            raise ValueError("trace steps must be 1..256 and skip nonnegative")
        self.nsys = bool(config.get("nsys", False))
        self.required_batch_size = config.get("required_batch_size")
        if self.required_batch_size is not None and self.required_batch_size <= 0:
            raise ValueError("required_batch_size must be positive")
        self.max_batch_size = config.get("max_batch_size")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.request_prefix = config.get("request_prefix")
        self.min_output_tokens = config.get("min_output_tokens")
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.cuda.current_device()
        # PD decode TP rank zero can live on physical GPU 4.
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.profile_group = None
        if self.nsys and torch.distributed.is_initialized():
            from sglang.srt.distributed.parallel_state import get_tp_group

            self.profile_group = get_tp_group()
        self.slot = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        self.stamps = torch.zeros(
            (self.MAX_REGIONS, self.steps, 2), dtype=torch.int64, device="cuda"
        )
        self.kv_entries = torch.zeros(
            (self.MAX_REGIONS, self.steps), dtype=torch.int32, device="cuda"
        )
        self.cache_work = torch.zeros_like(self.kv_entries)
        self.cache_work_regions = {}
        self.kv_entry_regions = set()
        self.names, self.side_slots, self.records, self.cpu = {}, {}, [], []
        self.accepted, self.prefix_lens_gpu, self.events = {}, {}, []
        self.role, self.seen, self.active, self.written = "target", 0, -1, False
        # Compile both stamp edges before graph capture.
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
        cpu_start = time.perf_counter_ns()
        live = self.active >= 0 and not self.torch.cuda.is_current_stream_capturing()
        if live:
            self.torch.cuda.nvtx.range_push(
                f"lagom/{full_name}/step={self.active}/layer={layer}"
            )
        mark_edges = live and self.nsys and name == "step"
        nvtx = self.torch.cuda.nvtx
        with (
            nvtx.range(f"lagom/step_begin/step={self.active}")
            if mark_edges
            else nullcontext()
        ):
            self.stamp[(1,)](self.stamps, target_slot, self.steps, region, 0)
        try:
            yield self
        finally:
            with (
                nvtx.range(f"lagom/step_end/step={self.active}")
                if mark_edges
                else nullcontext()
            ):
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

    def _kv_region(self, name, layer):
        index = self.names[(f"{self.role}.{name}", layer)]
        self.kv_entry_regions.add(index)
        return index

    def record_verify_copies(self, layer, plan, live, batch):
        capacity = plan.kinds.shape[1]
        self.count_verify_copies[(batch,)](
            self.kv_entries,
            self.slot,
            plan.kinds,
            plan.demand,
            plan.counts,
            live,
            self.steps,
            self._kv_region("kv_refill", layer),
            self._kv_region("kv_stage_copy", layer),
            capacity,
            1 << (capacity - 1).bit_length(),
        )

    def record_swap_in(self, layer, names, counts, force):
        gather, reread, commit = (self._kv_region(name, layer) for name in names)
        self.cache_work_regions[reread] = "resident_hits"
        self.cache_work_regions[commit] = "admissions"
        self.store_swap_in_counts[(1,)](
            self.kv_entries,
            self.cache_work,
            self.slot,
            counts,
            self.steps,
            gather,
            reread,
            commit,
            force,
        )

    def record_forecast_copies(self, layer, slot, counts, reqs, live):
        batch = reqs.numel()
        self.count_forecast_copies[(1,)](
            self.kv_entries,
            slot,
            counts,
            reqs,
            live,
            self.steps,
            self._kv_region("kv_prefetch", layer),
            batch,
            1 << (batch - 1).bit_length(),
        )

    def begin(self, batch):
        self.poll()
        if (
            (
                self.required_batch_size is not None
                and len(batch.reqs) != self.required_batch_size
            )
            or (
                self.max_batch_size is not None
                and not 0 < len(batch.reqs) <= self.max_batch_size
            )
            or (
                getattr(self, "min_output_tokens", None) is not None
                and any(
                    r.sampling_params.max_new_tokens < self.min_output_tokens
                    for r in batch.reqs
                )
            )
            or (
                self.request_prefix is not None
                and any(not r.rid.startswith(self.request_prefix) for r in batch.reqs)
            )
        ):
            self.active = -1
            self.slot.fill_(-1)
            return
        self.active = self.seen - self.skip
        self.seen += 1
        if not 0 <= self.active < self.steps:
            self.active = -1
        self.slot.fill_(self.active)
        if self.active == 0 and self.nsys:
            if self.rank == 0:
                self.torch.cuda.check_error(
                    self.torch.cuda.cudart().cudaProfilerStart()
                )
            if self.profile_group is not None:
                # Keep faster workers' first NVTX and GPU stamps inside capture.
                self.profile_group.barrier()
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
            # Side streams can still write this window after the main stream ends.
            for stream in getattr(self, "streams", []):
                event = self.torch.cuda.Event()
                event.record(stream)
                self.events.append(event)
            if self.nsys:
                # The final step has the same collective order on every TP rank.
                # poll() can run at different times and must never do a barrier.
                for event in self.events:
                    event.synchronize()
                if self.profile_group is not None:
                    self.profile_group.barrier()
                if self.rank == 0:
                    self.torch.cuda.check_error(
                        self.torch.cuda.cudart().cudaProfilerStop()
                    )

    def poll(self):
        if self.written or not self.events or not all(e.query() for e in self.events):
            return
        values = self.stamps.cpu().tolist()
        kv_entries = self.kv_entries.cpu().tolist() if self.kv_entry_regions else None
        cache_work = (
            self.cache_work.cpu().tolist()
            if getattr(self, "cache_work_regions", {})
            else None
        )
        regions = []
        for (name, layer), index in self.names.items():
            for step, (start, end) in enumerate(values[index]):
                if start > 0 and end >= start:
                    region = {
                        "name": name,
                        "layer": layer,
                        "step": step,
                        "start_ns": start,
                        "end_ns": end,
                    }
                    if index in self.kv_entry_regions:
                        region["kv_entries"] = kv_entries[index][step]
                    if cache_work is not None and index in self.cache_work_regions:
                        region[self.cache_work_regions[index]] = cache_work[index][step]
                    regions.append(region)
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
            "pid": pid,
            "nsys": getattr(self, "nsys", False),
            "rank": getattr(self, "rank", self.device),
            "records": self.records,
            "regions": regions,
            "cpu_regions": self.cpu,
            "summary": summarize(regions, self.steps),
            "notes": [
                "GPU spans include queue gaps and nested regions; do not sum inclusive spans.",
                "CPU and GPU clocks use independent origins in Chrome export; use Nsight for precise correlation.",
                "gpu_wait_cpu_ms is unknown without CUPTI; cpu_dependency_commit is an upper-bound dependency span.",
                "prefetch_wait belongs to the consuming step; kv_prefetch belongs to the producing step.",
                "kv_entries counts copied C4 rows per rank; each row carries 584 payload bytes.",
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
                        "args": {
                            "step": r["step"],
                            **(
                                {"kv_entries": r["kv_entries"]}
                                if "kv_entries" in r
                                else {}
                            ),
                        },
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
