"""Bounded, graph-safe byte accounting for the synchronous DSv4 verify path.

There is no host read or synchronization in record(). Start/stop synchronize
only at profile boundaries. The extra recorder kernel is deliberately visible
in the trace so its cost can be assessed separately.
"""

import json
import os
from pathlib import Path

import torch

_probes = {}


class HiSparseH2DProbe:
    def __init__(self, device, capacity: int, max_batch: int):
        if capacity <= 0 or max_batch <= 0:
            raise ValueError("H2D probe capacity and max batch must be positive")
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise ValueError("HiSparse H2D probe requires NVIDIA CUDA")
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.capacity = capacity
        self.max_batch = max_batch
        self.host_buffers = {}
        self.control = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.records = torch.empty(
            (capacity, 10 + max_batch), dtype=torch.int64, device=self.device
        )

    def register_host_buffers(self, named_buffers):
        # CPU metadata only, once at coordinator creation; no reads of KV data.
        for name, buffer in named_buffers:
            self.host_buffers[buffer.data_ptr()] = dict(
                name=name,
                address=buffer.data_ptr(),
                nbytes=buffer.numel() * buffer.element_size(),
            )

    def record(
        self, counts, num_real_reqs, *, layer, width, item_bytes, copy_blocks, skip_io
    ):
        import triton

        from sglang.kernels.ops.kvcache.hisparse_h2d_probe import (
            hisparse_h2d_record_kernel,
        )

        batch = counts.numel()
        if batch > self.max_batch:
            raise ValueError(
                f"Captured batch {batch} exceeds H2D probe max batch {self.max_batch}; "
                "increase SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH before launching"
            )
        hisparse_h2d_record_kernel[(1,)](
            counts,
            num_real_reqs,
            self.control,
            self.records,
            self.capacity,
            self.records.stride(0),
            batch,
            layer,
            width,
            item_bytes,
            copy_blocks,
            skip_io,
            triton.next_power_of_2(max(batch, 1)),
            num_warps=4,
        )

    def start(self):
        torch.cuda.synchronize(self.device)
        self.control.zero_()
        self.control[0].fill_(1)
        torch.cuda.current_stream(self.device).synchronize()

    def stop(self):
        from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot

        torch.cuda.synchronize(self.device)
        self.control[0].zero_()
        torch.cuda.current_stream(self.device).synchronize()
        total = int(self.control[1].item())
        rows = self.records[: min(total, self.capacity)].cpu().tolist()
        samples = []
        for row in rows:
            index, misses, real, batch, layer, width, size, blocks, skip, stamp = row[
                :10
            ]
            samples.append(
                dict(
                    index=index,
                    layer=layer,
                    verify_width=width,
                    real_batch=real,
                    captured_batch=batch,
                    item_bytes=size,
                    copy_blocks=blocks,
                    copy_threads=1024,
                    skip_io=bool(skip),
                    gpu_stamp_ns=stamp,
                    misses_per_request=row[10 : 10 + real],
                    unique_misses=misses,
                    h2d_bytes=0 if skip else misses * size,
                )
            )
        return dict(
            schema_version=1,
            pid=os.getpid(),
            device=self.device.index,
            gpu=torch.cuda.get_device_name(self.device),
            gpu_uuid=str(
                getattr(
                    torch.cuda.get_device_properties(self.device), "uuid", "unavailable"
                )
            ),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            host_numa_snapshot=process_snapshot(
                buffers=list(self.host_buffers.values())
            ),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            capacity=self.capacity,
            max_batch=self.max_batch,
            total_records=total,
            overflow=total > self.capacity,
            samples=samples,
        )


def resolve_probe_max_batch(required, configured=None):
    required = int(required)
    reserve = int(configured) if configured is not None else required
    if required <= 0 or reserve <= 0:
        raise ValueError("H2D probe request-slot count and reserve must be positive")
    return max(required, reserve)


def create_h2d_probe(device, *, max_batch):
    capacity = int(os.environ.get("SGLANG_HISPARSE_H2D_PROBE_CAPACITY", "0"))
    if capacity == 0:
        return None
    max_batch = resolve_probe_max_batch(
        max_batch, os.environ.get("SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH")
    )
    key = str(torch.device(device))
    if key not in _probes:
        _probes[key] = HiSparseH2DProbe(
            device,
            capacity,
            max_batch,
        )
    elif _probes[key].max_batch < max_batch:
        raise ValueError(
            f"Shared H2D probe was reserved for {_probes[key].max_batch} requests; "
            f"this coordinator needs {max_batch}. Set the reserve for the largest "
            "coordinator before startup; captured probe buffers cannot be resized."
        )
    return _probes[key]


def start_h2d_probes():
    for probe in _probes.values():
        probe.start()


def stop_h2d_probes():
    return [probe.stop() for probe in _probes.values()]


def annotate_h2d_trace(profiler, capture_id):
    if _probes:
        profiler.add_metadata_json(
            "hisparse_h2d_identity",
            json.dumps(dict(capture_id=capture_id, pid=os.getpid())),
        )


def export_h2d_probes(reports, output_dir, prefix, *, capture_id=None):
    for report in reports:
        report["capture_id"] = capture_id
        path = Path(output_dir) / f"{prefix}-GPU-{report['device']}.h2d.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
