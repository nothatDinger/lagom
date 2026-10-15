"""Round23 diagnostic dispatch and CPU-only validation; no serving integration."""

import functools
import math
import statistics
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_grid_perf import LEGACY_LIBRARY

VARIANTS = ("legacy", "control", "trace")
LIBRARIES = dict(
    legacy=LEGACY_LIBRARY,
    control="sgl_kernel_jit_hisparse_copy_latency_1024_False.so",
    trace="sgl_kernel_jit_hisparse_copy_latency_1024_True.so",
)
ORDERS = (
    ("legacy", "control", "trace"),
    ("trace", "control", "legacy"),
    ("control", "legacy", "trace"),
    ("trace", "legacy", "control"),
    ("legacy", "trace", "control"),
    ("control", "trace", "legacy"),
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
)


@functools.cache
def module(trace):
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "hisparse_copy_latency",
        1024,
        trace,
        cuda_files=["hisparse_copy_latency.cuh"],
        cuda_wrappers=[
            ("copy_cache_latency", f"copy_cache_latency<1024, {str(trace).lower()}>")
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
        records.shape != (blocks * 32, 9)
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
        module(variant == "trace").copy_cache_latency(
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
    if len(rows) != warps + 2 or any(len(r) != 9 for r in rows):
        raise ValueError("Record geometry mismatch")
    if rows[0] != [-1] * 9 or rows[-1] != [-1] * 9:
        raise ValueError("Record guard overwritten")
    if not trace:
        if any(row != [-1] * 9 for row in rows):
            raise ValueError("Uninstrumented kernel wrote records")
        return []
    if real < 0 or real > len(counts) or any(c < 0 for c in counts[:real]):
        raise ValueError("Invalid oracle counts")
    # Independent flat-work partition: each warp owns every W-th KV.
    total = sum(counts[:real])
    result = []
    for w, row in enumerate(rows[1:-1]):
        expected = len(range(w, total, warps))
        cta, local, gid, sm, begin, end, copy, kv, visits = row
        if (cta, local, gid, kv, visits) != (w // 32, w % 32, w, expected, real):
            raise ValueError(f"Record {w}: wrong schedule {row}")
        if sm < 0 or begin < 0 or end < begin or not 0 <= copy <= end - begin:
            raise ValueError(f"Record {w}: invalid timestamps {row}")
        if expected and (copy == 0 or end == begin):
            raise ValueError(f"Record {w}: unresolved nonempty timing")
        result.append(dict(zip(FIELDS, row)))
    return result


def stats(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Empty/nonfinite statistics")
    return dict(
        p50=statistics.median(values),
        p95=values[math.ceil(0.95 * len(values)) - 1],
        max=max(values),
    )


def summarize_invocation(rows):
    """Warp distributions inside ONE invocation, never sums across warps."""
    begin = min(r["begin_ns"] for r in rows)
    end = max(r["end_ns"] for r in rows)
    last = max(rows, key=lambda r: r["end_ns"])
    metrics = {}
    for name, fn in dict(
        loop_us=lambda r: (r["end_ns"] - r["begin_ns"]) / 1000,
        copy_us=lambda r: r["copy_ns"] / 1000,
        other_us=lambda r: (r["end_ns"] - r["begin_ns"] - r["copy_ns"]) / 1000,
        copy_per_kv_us=lambda r: r["copy_ns"] / 1000 / r["kv"],
    ).items():
        metrics[name] = stats(
            [fn(r) for r in rows if name != "copy_per_kv_us" or r["kv"]]
        )
    ctas = []
    for cta in sorted({r["cta"] for r in rows}):
        group = [r for r in rows if r["cta"] == cta]
        ctas.append(
            dict(
                cta=cta,
                sms=sorted({r["sm"] for r in group}),
                kv=sum(r["kv"] for r in group),
                begin_us=(min(r["begin_ns"] for r in group) - begin) / 1000,
                end_us=(max(r["end_ns"] for r in group) - begin) / 1000,
            )
        )
    return dict(
        metrics=metrics,
        ctas=ctas,
        sm_count=len({r["sm"] for r in rows}),
        start_spread_us=(max(r["begin_ns"] for r in rows) - begin) / 1000,
        end_spread_us=(end - min(r["end_ns"] for r in rows)) / 1000,
        envelope_us=(end - begin) / 1000,
        last_loop_warp={**last, "relative_end_us": (last["end_ns"] - begin) / 1000},
        by_kv={
            str(k): {
                "warps": sum(r["kv"] == k for r in rows),
                "copy_us": stats([r["copy_ns"] / 1000 for r in rows if r["kv"] == k]),
            }
            for k in sorted({r["kv"] for r in rows})
        },
    )


def perturbation(waves):
    lookup = {
        (r["trial"], r["variant"], r["blocks"]): statistics.fmean(r["copy_us"])
        for r in waves
    }
    if (
        len(waves) != 36
        or len(lookup) != 36
        or any(not math.isfinite(v) or v <= 0 for v in lookup.values())
    ):
        raise ValueError("Require all 36 unique positive waves")
    result = {}

    def gate(name, vals, tolerance):
        hits = sum(1 - tolerance <= v <= 1 + tolerance for v in vals)
        median = statistics.median(vals)
        result[name] = dict(
            ratios=vals,
            median=median,
            within=hits,
            valid=1 - tolerance <= median <= 1 + tolerance and hits >= 5,
        )

    for blocks in (4, 8):
        for a, b, tol in (("control", "legacy", 0.03), ("trace", "control", 0.05)):
            gate(
                f"{a}/{b}_CTA{blocks}",
                [lookup[t, a, blocks] / lookup[t, b, blocks] for t in range(6)],
                tol,
            )
    gate(
        "relative_grid_ratio",
        [
            (lookup[t, "trace", 8] / lookup[t, "trace", 4])
            / (lookup[t, "legacy", 8] / lookup[t, "legacy", 4])
            for t in range(6)
        ],
        0.03,
    )
    return dict(
        valid=all(g["valid"] for g in result.values()),
        gates=result,
        legacy_grid_ratios=[
            lookup[t, "legacy", 8] / lookup[t, "legacy", 4] for t in range(6)
        ],
    )
