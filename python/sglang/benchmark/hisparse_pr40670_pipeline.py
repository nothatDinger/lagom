"""CPU validation for cold PR-candidate/local whole-pipeline comparisons."""

import math
import statistics

from sglang.benchmark.hisparse_h2d_pair import launch_shape
from sglang.benchmark.hisparse_pr40670_check import launch_config

METHODS = ("pr_retry_fixed", "local_plan_copy")
ORDERS = (METHODS, METHODS[::-1]) * 3


def validate_pipeline_profile(trace, method, batch):
    if method not in METHODS:
        raise ValueError("Unknown pipeline")
    shape = launch_config(batch, 6, 16)
    expected = [
        (f"load_cache_to_device_buffer_spec_{k}_kernel", v) for k, v in shape.items()
    ]
    if method == "local_plan_copy":
        expected.append(
            ("copy_cache_planned_kernel", dict(grid=[4, 1, 1], block=[1024, 1, 1]))
        )
    events = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]
    if len(events) != len(expected):
        raise ValueError(f"Wrong pipeline kernel count: {len(events)}")
    rows = []
    for name, config in expected:
        matches = [e for e in events if name in e.get("name", "")]
        if len(matches) != 1:
            raise ValueError(f"Missing/duplicate {name}")
        e = matches[0]
        a = e.get("args", {})
        if (
            list(launch_shape(a.get("grid")) or ()) != config["grid"]
            or list(launch_shape(a.get("block")) or ()) != config["block"]
            or a.get("device") != 0
            or not math.isfinite(e.get("dur", 0))
            or e["dur"] <= 0
        ):
            raise ValueError(f"Invalid {name} launch")
        rows.append(dict(kind=name, duration_us=e["dur"], **config))
    ordered = sorted(events, key=lambda e: e["ts"])
    if any(
        name not in event.get("name", "") for (name, _), event in zip(expected, ordered)
    ):
        raise ValueError("Wrong pipeline order")
    return dict(valid=True, kernels=rows)


def summarize(waves, samples):
    expected = {(t, m) for t in range(6) for m in METHODS}
    keys = [(w["trial"], w["method"]) for w in waves]
    if len(keys) != 12 or set(keys) != expected:
        raise ValueError("Incomplete/duplicate balanced waves")
    values = {}
    for w in waves:
        x = w["us"]
        if len(x) != samples or not all(math.isfinite(v) and v > 0 for v in x):
            raise ValueError("Invalid samples")
        values[w["trial"], w["method"]] = statistics.mean(x)
    means = {m: [values[t, m] for t in range(6)] for m in METHODS}
    ratios = [means[METHODS[1]][t] / means[METHODS[0]][t] for t in range(6)]
    return dict(
        wave_means_us=means,
        mean_us={m: statistics.mean(v) for m, v in means.items()},
        local_over_pr_wave_ratios=ratios,
        local_over_pr=statistics.mean(means[METHODS[1]])
        / statistics.mean(means[METHODS[0]]),
        samples_per_method=6 * samples,
    )


def graph_sample_us(start, end):
    """Read actual external event timestamps, rejecting unusable measurements."""
    end.synchronize()
    value = start.elapsed_time(end) * 1000
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid captured event duration: {value}")
    return value
