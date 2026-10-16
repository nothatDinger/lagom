"""CPU analysis for the two-reader diagnostic; overlap is not proof of no internal lock."""

import math
from collections import Counter, defaultdict

from sglang.benchmark.hisparse_h2d_pair import launch_shape


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Nonpositive activity")
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(b - a for a, b in intervals)


def activity_overlap(trace, banks, parallel):
    kernels = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("cat") == "kernel"
        and "copy_cache_planned_kernel" in e.get("name", "")
        and e.get("ph") == "X"
    ]
    groups = defaultdict(list)
    for e in kernels:
        a = e.get("args", {})
        if (
            launch_shape(a.get("grid")) != (4, 1, 1)
            or launch_shape(a.get("block")) != (1024, 1, 1)
            or a.get("device") != 0
            or "stream" not in a
        ):
            raise ValueError("Missing/wrong activity device/grid/block/stream")
        groups[str(a["stream"])].append((e["ts"], e["ts"] + e["dur"]))
    if len(kernels) != 2 * banks:
        raise ValueError(f"Expected {2*banks} copy activities, found {len(kernels)}")
    if parallel and (len(groups) != 2 or any(len(v) != banks for v in groups.values())):
        raise ValueError(
            f"Parallel capture did not expose two independent streams: { {k:len(v) for k,v in groups.items()} }"
        )
    united = merge([x for v in groups.values() for x in v])
    per_stream = {k: merge(v) for k, v in groups.items()}
    overlap = duration([x for v in per_stream.values() for x in v]) - duration(united)
    if not parallel and overlap > 1e-3:
        raise ValueError(
            "Serial reference unexpectedly has overlapping stream activities"
        )
    return dict(
        valid=True,
        kernels=len(kernels),
        stream_counts=dict(Counter(str(e["args"]["stream"]) for e in kernels)),
        activity_union_us=duration(united),
        envelope_us=united[-1][1] - united[0][0],
        stream_active_us={k: duration(v) for k, v in per_stream.items()},
        intersection_us=overlap,
        overlap_fraction_shorter=(
            overlap / min(duration(v) for v in per_stream.values())
            if parallel
            else None
        ),
    )
