"""Standard-library analysis of real service pipeline activities; one TP rank."""

import math
import statistics
from collections import defaultdict

GATHER = "load_cache_to_device_buffer_spec_gather_kernel"
COMMIT = "load_cache_to_device_buffer_spec_commit_kernel"
COPY = "copy_cache_planned_kernel"
MARKER = "hisparse_h2d_record_kernel"


def analyze_service(probe, events, backend):
    if (
        backend not in ("split", "pr_fused")
        or probe.get("verify_backend", "split") != backend
    ):
        raise ValueError("Wrong service backend")
    config = probe.get("verify_config", {})
    if config.get("top_k") != 512 or config.get("force_h2d_each_step") is not True:
        raise ValueError("Require effective top_k=512 and force_h2d_each_step=true")
    rows = probe.get("samples", [])
    if probe.get("overflow") or not rows or probe.get("total_records") != len(rows):
        raise ValueError("Missing or overflowing payload records")
    for i, row in enumerate(rows):
        counts = row["misses_per_request"]
        if (
            row["index"] != i
            or row["real_batch"] <= 0
            or len(counts) != row["real_batch"]
            or row["captured_batch"] < row["real_batch"]
            or any(v < 0 for v in counts)
            or sum(counts) != row["unique_misses"]
            or row["skip_io"]
            or row["item_bytes"] != 584
            or row["h2d_bytes"] != sum(counts) * 584
        ):
            raise ValueError("Invalid payload record")
    kernels = sorted(
        [
            e
            for e in events
            if e.get("ph") == "X"
            and "kernel" in e.get("cat", "").lower()
            and e.get("args", {}).get("device") == probe["device"]
        ],
        key=lambda e: e["ts"],
    )
    expected = [GATHER, COMMIT] + ([COPY] if backend == "split" else [])
    pending, pairs = defaultdict(list), []
    for e in kernels:
        name = e.get("name", "")
        kind = next((k for k in (GATHER, COMMIT, COPY) if k in name), None)
        marker = MARKER in name
        if not kind and not marker:
            continue  # PR profile-only count extraction is outside the pipeline.
        stream = e.get("args", {}).get("stream")
        if (
            stream is None
            or not math.isfinite(e.get("dur", 0))
            or e["dur"] <= 0
            or not math.isfinite(e["ts"])
        ):
            raise ValueError("Invalid pipeline activity")
        if kind:
            pending[stream].append((kind, e))
        else:
            nodes = pending.pop(stream, [])
            if [k for k, _ in nodes] != expected:
                raise ValueError(
                    f"Pipeline/recorder mismatch on stream {stream}: {[k for k, _ in nodes]}"
                )
            if max(n["ts"] + n["dur"] for _, n in nodes) > e["ts"] + 0.01:
                raise ValueError("Pipeline extends beyond its recorder")
            pairs.append((e, nodes))
    if any(pending.values()) or len(pairs) != len(rows):
        raise ValueError("Unpaired pipeline activity or record")
    pairs.sort(key=lambda pair: pair[0]["ts"])
    if any(b[0]["ts"] < a[0]["ts"] + a[0]["dur"] for a, b in zip(pairs, pairs[1:])):
        raise ValueError("Overlapping recorders: cannot assign per-layer payload")
    joined = []
    for row, (marker, nodes) in zip(rows, pairs):
        by_kind = {kind: e for kind, e in nodes}
        batch, width = row["captured_batch"], row["verify_width"]
        expected_shapes = {
            GATHER: ([batch, (width * 512 + 63) // 64, 1], [64, 1, 1]),
            COMMIT: ([batch, 1, 1], [512, 1, 1]),
            COPY: ([4, 1, 1], [1024, 1, 1]),
        }
        policy = (
            "Dsv4PagedKVTransferPolicy"
            if backend == "pr_fused"
            else "MetadataOnlyKVTransferPolicy"
        )
        for kind, node in nodes:
            grid, block = expected_shapes[kind]
            a = node.get("args", {})
            if list(a.get("grid", [])) != grid or list(a.get("block", [])) != block:
                raise ValueError("Unexpected service launch shape")
            if kind != COPY and policy not in node["name"]:
                raise ValueError("Kernel policy does not match backend")
        start = min(e["ts"] for _, e in nodes)
        end = max(e["ts"] + e["dur"] for _, e in nodes)
        joined.append(
            dict(
                **row,
                pipeline_start_us=start,
                pipeline_us=end - start,
                gather_us=by_kind[GATHER]["dur"],
                commit_us=by_kind[COMMIT]["dur"],
                copy_us=by_kind[COPY]["dur"] if COPY in by_kind else None,
                recorder_us=marker["dur"],
                stream=marker["args"]["stream"],
            )
        )
    groups = defaultdict(list)
    for row in joined:
        groups[row["real_batch"], row["verify_width"]].append(row)
    summaries = []
    for (batch, width), group in sorted(groups.items()):
        entry = dict(
            batch=batch,
            verify_width=width,
            samples=len(group),
            unique_misses_per_request_mean=statistics.fmean(
                m for s in group for m in s["misses_per_request"]
            ),
            unique_misses_per_request_min=min(
                m for s in group for m in s["misses_per_request"]
            ),
            unique_misses_per_request_max=max(
                m for s in group for m in s["misses_per_request"]
            ),
        )
        for field in ("gather_us", "commit_us", "copy_us", "pipeline_us"):
            values = [s[field] for s in group if s[field] is not None]
            entry[field] = (
                dict(mean=statistics.fmean(values), median=statistics.median(values))
                if values
                else None
            )
        layers = sorted({s["layer"] for s in group})
        entry["layers"] = []
        for layer in layers:
            subset = [s for s in group if s["layer"] == layer]
            entry["layers"].append(
                dict(
                    layer=layer,
                    samples=len(subset),
                    **{
                        field: statistics.fmean(s[field] for s in subset)
                        if subset[0][field] is not None
                        else None
                        for field in (
                            "gather_us",
                            "commit_us",
                            "copy_us",
                            "pipeline_us",
                        )
                    },
                )
            )
        # Only exact contiguous repeated layer cycles support this cadence proxy.
        n = len(layers)
        blocks = [group[i : i + n] for i in range(0, len(group), n)]
        template = [s["layer"] for s in blocks[0]]
        entry["verify_cadence"] = None
        if (
            len(blocks) >= 2
            and len(set(template)) == n
            and all([s["layer"] for s in b] == template for b in blocks)
            and all(b["index"] == a["index"] + 1 for a, b in zip(group, group[1:]))
            and len({s["stream"] for s in group}) == 1
        ):
            periods = []
            costs = []
            for current, following in zip(blocks, blocks[1:]):
                period = (
                    following[0]["pipeline_start_us"] - current[0]["pipeline_start_us"]
                )
                cost = sum(s["pipeline_us"] for s in current)
                if period <= 0 or cost > period + 0.01:
                    raise ValueError("Invalid synchronous cadence")
                periods.append(period)
                costs.append(cost)
            entry["verify_cadence"] = dict(
                intervals=len(periods),
                mean_period_ms=statistics.fmean(periods) / 1000,
                mean_sync_pipeline_ms=statistics.fmean(costs) / 1000,
                sync_pipeline_fraction=sum(costs) / sum(periods),
                scope="profiled verify-to-verify cadence, not a TPOT attribution",
            )
        summaries.append(entry)
    return dict(
        valid=True,
        backend=backend,
        pid=probe["pid"],
        device=probe["device"],
        gpu_uuid=probe.get("gpu_uuid"),
        groups=summaries,
        samples=joined,
    )
