"""Segment-boundary check only; no performance runner or serving integration."""

import functools
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_grid_perf import LEGACY_LIBRARY

VARIANTS = ("legacy", "control", "trace")
LIBRARIES = dict(
    legacy=LEGACY_LIBRARY,
    control="sgl_kernel_jit_hisparse_copy_segments_1024_False.so",
    trace="sgl_kernel_jit_hisparse_copy_segments_1024_True.so",
)
FIELDS = (
    "cta",
    "warp_in_cta",
    "warp",
    "sm",
    "begin_ns",
    "end_ns",
    "copy_ns",
    "kv",
    "visits",
    "prep_ns",
    "issue_ns",
    "residual_ns",
    "store_ns",
    "checksum",
    "bad_order",
)


@functools.cache
def module(trace):
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "hisparse_copy_segments",
        1024,
        trace,
        cuda_files=["hisparse_copy_segments.cuh"],
        cuda_wrappers=[
            ("copy_cache_segments", f"copy_cache_segments<1024, {str(trace).lower()}>")
        ],
    )


def launch(variant, blocks, src, dst, counts, real, source, output, records):
    import torch

    if variant not in VARIANTS or blocks not in (4, 8):
        raise ValueError("Require legacy/control/trace, CTA4/8")
    if (
        src.dtype != torch.int64
        or dst.dtype != torch.int32
        or counts.dtype != torch.int32
        or real.dtype != torch.int32
    ):
        raise ValueError("Invalid plan dtypes")
    if (
        src.ndim != 2
        or dst.shape != src.shape
        or src.stride(1) != 1
        or dst.stride() != src.stride()
    ):
        raise ValueError("Require equal row-strided plans")
    if counts.ndim != 1 or counts.numel() < src.shape[0] or real.numel() != 1:
        raise ValueError("Invalid counts/real sizes")
    if (
        records.shape != (blocks * 32, 15)
        or records.dtype != torch.int64
        or not records.is_contiguous()
    ):
        raise ValueError("Invalid records allocation")
    if output.device.type != "cuda" or any(
        t.device != output.device for t in (src, dst, counts, real, records)
    ):
        raise ValueError("All GPU operands must share device")
    if any(
        t.dtype != torch.uint8
        or t.ndim != 2
        or t.shape[1] != 37440
        or not t.is_contiguous()
        for t in (source, output)
    ):
        raise ValueError("Require contiguous C4 source/output")
    if source.device.type == "cpu":
        if not source.is_pinned():
            raise ValueError("Host source must be pinned")
    elif source.device != output.device:
        raise ValueError("HBM source device mismatch")
    if variant == "legacy":
        from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla

        copy_cache_planned_mla(
            miss_src=src,
            miss_dst=dst,
            miss_count=counts,
            num_real_reqs=real,
            host_cache=source,
            device_buffer=output,
            item_size_bytes=584,
            num_blocks=blocks,
            block_size=1024,
            is_dsv4_layout=True,
            load_policy="legacy",
        )
    else:
        module(variant == "trace").copy_cache_segments(
            src, dst, counts, real, source, output, records, blocks
        )


def loaded_artifacts():
    paths = {
        f[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(f := line.split(None, 5)) == 6
    }
    result = []
    for variant, name in LIBRARIES.items():
        found = [p for p in paths if Path(p).name == name]
        if len(found) != 1:
            raise ValueError(f"Expected one loaded {variant}: {found}")
        result.append(dict(variant=variant, path=found[0], sha256=sha256(found[0])))
    return result


def validate_records(rows, blocks, counts, real, trace=True):
    """Input includes one guard row at each end. Do not repair invalid times."""
    warps = blocks * 32
    if len(rows) != warps + 2 or any(len(r) != 15 for r in rows):
        raise ValueError("Record geometry mismatch")
    if rows[0] != [-1] * 15 or rows[-1] != [-1] * 15:
        raise ValueError("Record guard overwritten")
    if not trace:
        if any(row != [-1] * 15 for row in rows):
            raise ValueError("Uninstrumented kernel wrote records")
        return []
    if real < 0 or real > len(counts) or any(c < 0 for c in counts[:real]):
        raise ValueError("Invalid oracle counts")
    # Independent flat-work partition: each warp owns every W-th KV.
    total = sum(counts[:real])
    result = []
    for w, row in enumerate(rows[1:-1]):
        expected = len(range(w, total, warps))
        cta, local, gid, sm, begin, end, copy, kv, visits = row[:9]
        if (cta, local, gid, kv, visits) != (w // 32, w % 32, w, expected, real):
            raise ValueError(f"Record {w}: wrong schedule {row}")
        if sm < 0 or begin < 0 or end < begin or not 0 <= copy <= end - begin:
            raise ValueError(f"Record {w}: invalid timestamps {row}")
        if expected and (copy == 0 or end == begin):
            raise ValueError(f"Record {w}: unresolved nonempty timing")
        if row[14] != 0 or any(x < 0 for x in row[9:13]) or sum(row[9:13]) != copy:
            raise ValueError(f"Record {w}: invalid segment accounting {row}")
        if not expected and (copy != 0 or row[13] != 0):
            raise ValueError("Empty warp has nonempty segment data")
        result.append(dict(zip(FIELDS, row)))
    return result
