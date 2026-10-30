"""Bounded, event-only measurement of consecutive full-batch DSPARK decodes.

CUDA events surround the worker call, outside captured graphs. All device
metadata is collected after a completion event; no per-step synchronization.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path


@functools.lru_cache(maxsize=1)
def get_meter():
    path = os.getenv("SGLANG_LAGOM_DECODE_PERF")
    return DecodeMeter(json.loads(Path(path).read_text())) if path else None


def measure_decode(fn):
    @functools.wraps(fn)
    def wrapped(self, batch, *args, **kwargs):
        meter = get_meter()
        if meter is None:
            return fn(self, batch, *args, **kwargs)
        meter.begin(batch)
        try:
            result = fn(self, batch, *args, **kwargs)
        except Exception:
            meter.cancel()
            raise
        meter.finish(result)
        return result

    return wrapped


class FullBatchWindow:
    """Require the same cohort and consecutive pure decode forwards."""

    def __init__(self, batch_size, steps, warmup):
        if batch_size not in (8, 16, 32, 64) or not 1 <= steps <= 256 or warmup < 0:
            raise ValueError("require batch size 8/16/32/64, steps 1..256, warmup >=0")
        self.batch_size, self.steps, self.warmup = batch_size, steps, warmup
        self.cohort, self.warmed, self.recorded, self.discarded = None, 0, 0, 0
        self.complete = False

    def select(self, request_ids, eligible):
        if self.complete:
            return None
        cohort = tuple(sorted(request_ids))
        if not eligible or len(cohort) != self.batch_size or cohort != self.cohort:
            self.discarded += self.recorded
            self.cohort, self.warmed, self.recorded = None, 0, 0
        if not eligible or len(cohort) != self.batch_size:
            return None
        self.cohort = cohort
        if self.warmed < self.warmup:
            self.warmed += 1
            return None
        slot = self.recorded
        self.recorded += 1
        return slot


class DecodeMeter:
    def __init__(self, config):
        import torch

        self.window = FullBatchWindow(
            config["batch_size"], config["steps"], config["warmup"]
        )
        self.width = int(config["verify_width"])
        self.request_prefix = config["request_prefix"]
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.cuda.current_device()
        self.stem = self.output / f"rank-device{self.device}-pid{os.getpid()}"
        self.starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.window.steps)
        ]
        self.ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.window.steps)
        ]
        self.done = torch.cuda.Event()
        # Materialize event handles during the initial, unmeasured prefill.
        for event in (*self.starts, *self.ends, self.done):
            event.record()
        self.records, self.accepted, self.prefixes = [], [], []
        self.active, self.written = None, False
        self.observed_max_batch, self.forwards = 0, 0
        self.last_progress = 0.0

    def begin(self, batch):
        self.poll()
        self.active = None
        if self.window.complete:
            return
        self.forwards += 1
        pure_decode = batch.forward_mode.is_decode() and not batch.is_extend_in_batch
        ids = [r.rid for r in batch.reqs]
        if pure_decode:
            self.observed_max_batch = max(self.observed_max_batch, len(ids))
        # The overlap scheduler's output_ids may lag by one verify. Keep two
        # complete verify windows of headroom so accepted counts cannot include
        # tokens discarded at max_new_tokens. The client also sets ignore_eos.
        remaining = [
            r.sampling_params.max_new_tokens - len(r.output_ids) for r in batch.reqs
        ]
        eligible = (
            pure_decode
            and all(rid.startswith(self.request_prefix) for rid in ids)
            and all(n > 2 * self.width for n in remaining)
        )
        slot = self.window.select(ids, eligible)
        if slot is None:
            if self.window.recorded == 0:
                self.records, self.accepted, self.prefixes = [], [], []
            self.progress()
            return
        if slot == 0:
            self.records, self.accepted, self.prefixes = [], [], []
        prefix_cpu = batch.seq_lens_cpu
        self.prefixes.append(
            batch.seq_lens.detach().clone() if prefix_cpu is None else None
        )
        self.records.append(
            {
                "step": slot,
                "forward_iter": int(batch.forward_iter),
                "batch_size": len(ids),
                "request_ids": ids,
                "prefix_lens": prefix_cpu.tolist() if prefix_cpu is not None else None,
                "remaining_output_budget_cpu": remaining,
            }
        )
        self.active = slot
        self.cpu_start = time.perf_counter_ns()
        self.starts[slot].record()

    def finish(self, result):
        if self.active is None:
            return
        self.ends[self.active].record()
        record = self.records[-1]
        record["cpu_submit_ms"] = (time.perf_counter_ns() - self.cpu_start) / 1e6
        record["cuda_graph"] = bool(result.can_run_cuda_graph)
        self.accepted.append(result.accept_lens.detach().clone())
        if self.active == self.window.steps - 1:
            self.done.record()
            self.window.complete = True
        self.active = None

    def cancel(self):
        if not self.window.complete:
            self.window.select((), False)
            self.records, self.accepted, self.prefixes = [], [], []
        self.active = None

    def progress(self):
        now = time.monotonic()
        if now - self.last_progress < 1:
            return
        self.last_progress = now
        self.stem.with_suffix(".status.json").write_text(
            json.dumps(
                {
                    "status": "waiting_for_full_decode_window",
                    "required_batch_size": self.window.batch_size,
                    "observed_max_decode_batch_size": self.observed_max_batch,
                    "forwards_observed": self.forwards,
                    "discarded_partial_steps": self.window.discarded,
                },
                indent=2,
            )
        )

    def poll(self):
        if self.written or not self.window.complete or not self.done.query():
            return
        for i, record in enumerate(self.records):
            record["start_ms"] = self.starts[0].elapsed_time(self.starts[i])
            record["end_ms"] = self.starts[0].elapsed_time(self.ends[i])
            record["step_ms"] = self.starts[i].elapsed_time(self.ends[i])
            record["accepted_tokens"] = self.accepted[i].cpu().tolist()
            if record["prefix_lens"] is None:
                record["prefix_lens"] = self.prefixes[i].cpu().tolist()
        payload = {
            "schema_version": 1,
            "device": self.device,
            "batch_size": self.window.batch_size,
            "steps": self.window.steps,
            "warmup": self.window.warmup,
            "verify_width": self.width,
            "window_ms": self.starts[0].elapsed_time(self.ends[-1]),
            "records": self.records,
            "discarded_partial_steps": self.window.discarded,
            "notes": [
                "Only consecutive pure decode forwards with the same full cohort are measured.",
                "Window duration includes inter-step scheduler/CPU gaps; step spans are not additive with those gaps.",
                "CUDA events and device snapshots add small measurement overhead; no layer trace kernels are enabled.",
                "Accepted counts include bonus tokens, exclude prefill output, and are counted once across TP ranks.",
            ],
        }
        self.stem.with_suffix(".json").write_text(json.dumps(payload, indent=2))
        self.written = True
