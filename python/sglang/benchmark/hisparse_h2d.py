"""Join real verify miss bytes to GPU copy durations, one rank at a time.

Accepts a torch Chrome trace (.json/.json.gz) or an Nsight Systems SQLite export.
Uses only the standard library. Multiple streams are paired locally; serialized
recorder intervals establish sample order. Overlapping recorders permit capture
totals only, without guessing per-sample bytes/time associations.
"""

import argparse
import gzip
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def read_events(path, *, capture_id=None, pid=None):
    path = Path(path)
    if path.suffix == ".sqlite":
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            names = dict(db.execute("SELECT id, value FROM StringIds"))
            db.row_factory = sqlite3.Row
            events = []
            for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
                name_id = row["demangledName"]
                if name_id not in names:
                    name_id = row["shortName"]
                events.append(
                    dict(
                        ph="X",
                        cat="kernel",
                        name=names[name_id],
                        ts=row["start"] / 1000,
                        dur=(row["end"] - row["start"]) / 1000,
                        args=dict(device=row["deviceId"], stream=row["streamId"]),
                    )
                )
            return events
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        trace = json.load(stream)
    if capture_id is not None:
        identity = trace.get("hisparse_h2d_identity", {})
        if identity.get("capture_id") != capture_id or identity.get("pid") != pid:
            raise ValueError("Trace and probe have different capture/rank identities")
    return trace["traceEvents"]


def distribution(values):
    values = sorted(values)
    return dict(
        min=values[0],
        median=statistics.median(values),
        mean=statistics.fmean(values),
        p95=values[math.ceil(0.95 * len(values)) - 1],
        max=values[-1],
    )


def _validate_probe(probe):
    if probe.get("schema_version") != 1:
        raise ValueError("Unsupported H2D probe schema")
    if probe.get("overflow"):
        raise ValueError("Probe overflow; increase capacity and recollect")
    rows = probe["samples"]
    if not rows or probe["total_records"] != len(rows):
        raise ValueError("Probe has no complete samples")
    for index, sample in enumerate(rows):
        if sample["index"] != index:
            raise ValueError("Non-contiguous probe record indices")
        counts = sample["misses_per_request"]
        if (
            sample["item_bytes"] != 584
            or sample["captured_batch"] < sample["real_batch"]
        ):
            raise ValueError(
                "Probe must use DSv4 C4 payload size and valid padding geometry"
            )
        if sample["real_batch"] <= 0 or len(counts) != sample["real_batch"]:
            raise ValueError("Invalid real batch or per-request miss counters")
        if any(m < 0 for m in counts) or sum(counts) != sample["unique_misses"]:
            raise ValueError("Invalid miss counters")
        expected = 0 if sample["skip_io"] else sum(counts) * sample["item_bytes"]
        if expected != sample["h2d_bytes"]:
            raise ValueError("Bytes do not match miss counts and item size")
        if sample["skip_io"]:
            raise ValueError("Collect normal IO; skip-IO cannot measure H2D bandwidth")
    return rows


def _payload_stats(samples):
    return dict(
        samples=len(samples),
        h2d_bytes=sum(s["h2d_bytes"] for s in samples),
        bytes_per_copy=distribution([s["h2d_bytes"] for s in samples]),
        unique_misses_per_request=distribution(
            [m for s in samples for m in s["misses_per_request"]]
        ),
    )


def _union_us(intervals):
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def _time_stats(samples, byte_count):
    copy_us = sum(s["copy_us"] for s in samples)
    union_us = _union_us(
        [(s["copy_start_us"], s["copy_start_us"] + s["copy_us"]) for s in samples]
    )
    marker_us = sum(s["recorder_us"] for s in samples)
    return dict(
        copy_gpu_us=copy_us,
        copy_gpu_union_us=union_us,
        copy_gpu_overlap_us=max(0, copy_us - union_us),
        copy_GBps=byte_count / copy_us / 1000,
        copy_active_wall_GBps=byte_count / union_us / 1000,
        recorder_gpu_us=marker_us,
        recorder_over_copy_pct=100 * marker_us / copy_us,
        copy_us=distribution([s["copy_us"] for s in samples]),
    )


def _summarize(samples):
    payload = _payload_stats(samples)
    return dict(**payload, **_time_stats(samples, payload["h2d_bytes"]))


def _group_stats(samples, statistics_fn):
    geometry, captured, layers = defaultdict(list), defaultdict(list), defaultdict(list)
    for sample in samples:
        geometry[(sample["real_batch"], sample["verify_width"])].append(sample)
        captured[
            (sample["real_batch"], sample["captured_batch"], sample["verify_width"])
        ].append(sample)
        layers[sample["layer"]].append(sample)
    return dict(
        geometry=[
            dict(batch=b, verify_width=w, **statistics_fn(s))
            for (b, w), s in sorted(geometry.items())
        ],
        captured_geometry=[
            dict(batch=b, captured_batch=c, verify_width=w, **statistics_fn(s))
            for (b, c, w), s in sorted(captured.items())
        ],
        layers={str(layer): statistics_fn(s) for layer, s in sorted(layers.items())},
    )


def summarize_payload(probe, *, batch_size=None, verify_width=None):
    rows = _validate_probe(probe)
    rows = [
        s
        for s in rows
        if (batch_size is None or s["real_batch"] == batch_size)
        and (verify_width is None or s["verify_width"] == verify_width)
    ]
    if not rows:
        raise ValueError("No samples match the selected geometry")
    return dict(
        device=probe.get("device"),
        pid=probe["pid"],
        gpu=probe["gpu"],
        scope="Probe payload only; no timing or bandwidth inferred",
        summary=_payload_stats(rows),
        **_group_stats(rows, _payload_stats),
        samples=rows,
    )


def _copy_recorder_pairs(kernels):
    streams = defaultdict(list)
    for event in kernels:
        streams[event.get("args", {}).get("stream")].append(event)
    pairs = []
    for stream, timeline in streams.items():
        previous = None
        for event in timeline:
            if "hisparse_h2d_record_kernel" in event["name"]:
                if stream is None:
                    raise ValueError("Recorder event has no stream ID")
                if (
                    previous is None
                    or "copy_cache_planned_kernel" not in previous["name"]
                ):
                    raise ValueError(
                        "Recorder is not immediately after a copy on its stream"
                    )
                for node in (previous, event):
                    if (
                        not math.isfinite(node["ts"])
                        or not math.isfinite(node["dur"])
                        or node["dur"] <= 0
                    ):
                        raise ValueError("Invalid GPU kernel timestamps or durations")
                if previous["ts"] + previous["dur"] > event["ts"] + 0.001:
                    raise ValueError("Copy and its recorder overlap on the same stream")
                pairs.append((previous, event))
            previous = event
    return sorted(pairs, key=lambda pair: pair[1]["ts"])


def _pair_times(copy, marker):
    return dict(
        copy_start_us=copy["ts"],
        copy_us=copy["dur"],
        recorder_us=marker["dur"],
        copy_stream=copy["args"]["stream"],
    )


def summarize(
    probe,
    events,
    *,
    device=None,
    batch_size=None,
    verify_width=None,
    start_us=None,
    end_us=None,
):
    rows = _validate_probe(probe)
    if start_us is not None and end_us is not None and start_us >= end_us:
        raise ValueError("Window start must precede window end")
    kernels = [
        e for e in events if e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()
    ]
    markers = [e for e in kernels if "hisparse_h2d_record_kernel" in e["name"]]
    devices = {int(e["args"]["device"]) for e in markers}
    if device is None:
        if len(devices) != 1:
            raise ValueError("Select one GPU with --device; do not sum TP ranks")
        device = devices.pop()
    kernels = sorted(
        [e for e in kernels if int(e.get("args", {}).get("device", -1)) == device],
        key=lambda e: e["ts"],
    )
    markers = [e for e in kernels if "hisparse_h2d_record_kernel" in e["name"]]
    if len(markers) != len(rows):
        raise ValueError(
            f"Trace has {len(markers)} recorder nodes, probe has {len(rows)} records. "
            "Use the matching profile/rank and export individual CUDA graph nodes."
        )
    pairs = _copy_recorder_pairs(kernels)
    marker_end = -math.inf
    overlapping = False
    for _, marker in pairs:
        overlapping |= marker["ts"] < marker_end
        marker_end = max(marker_end, marker["ts"] + marker["dur"])
    pairing = dict(
        mode="capture_totals_only" if overlapping else "serialized_recorders",
        recorder_streams=sorted({m["args"]["stream"] for _, m in pairs}, key=str),
        recorder_intervals_overlap=overlapping,
        paired_copies=len(pairs),
    )
    if overlapping:
        selected = summarize_payload(
            probe, batch_size=batch_size, verify_width=verify_width
        )
        if (
            len(selected["samples"]) != len(rows)
            or start_us is not None
            or end_us is not None
        ):
            raise ValueError(
                "Overlapping recorders support whole-capture totals only; remove subset filters or use --probe-only"
            )
        times = [_pair_times(c, m) for c, m in pairs]
        selected.update(
            device=device,
            units="GB/s = 10^9 bytes/s; GPU trace timestamps are microseconds",
            pairing=pairing,
            scope="Whole-capture payload and GPU copy times; per-sample/layer timing is unassigned because recorder intervals overlap",
            copy_intervals=times,
        )
        selected["summary"].update(_time_stats(times, selected["summary"]["h2d_bytes"]))
        return selected
    joined = []
    stamps = []
    for index, (sample, (copy, marker)) in enumerate(zip(rows, pairs)):
        stamps.append(sample["gpu_stamp_ns"])
        # Include whole copies only. Clipping time while retaining full bytes
        # would silently overstate bandwidth at the boundaries.
        if start_us is not None and copy["ts"] < start_us:
            continue
        if end_us is not None and copy["ts"] + copy["dur"] > end_us:
            continue
        if batch_size is not None and sample["real_batch"] != batch_size:
            continue
        if verify_width is not None and sample["verify_width"] != verify_width:
            continue
        joined.append(
            dict(
                **sample,
                **_pair_times(copy, marker),
            )
        )
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Probe timestamps are not in stream order")
    if not joined:
        raise ValueError("No complete copies match the selected window/geometry")
    report = dict(
        device=device,
        pid=probe["pid"],
        gpu=probe["gpu"],
        units="GB/s = 10^9 bytes/s; GPU trace timestamps are microseconds",
        scope="Actual miss payload / copy-kernel GPU duration, one rank; excludes planner, launch gaps and recorder. This is not PCIe wire bandwidth or marginal decode latency.",
        summary=_summarize(joined),
        pairing=pairing,
        **_group_stats(joined, _summarize),
        samples=joined,
    )
    if start_us is not None and end_us is not None:
        if batch_size is not None or verify_width is not None:
            report["window_note"] = (
                "Geometry filters applied; no whole-decode bandwidth reported"
            )
        else:
            report["selected_window"] = dict(
                start_us=start_us,
                end_us=end_us,
                wall_us=end_us - start_us,
                payload_GBps=report["summary"]["h2d_bytes"]
                / (end_us - start_us)
                / 1000,
                copy_time_share_pct=100
                * report["summary"]["copy_gpu_union_us"]
                / (end_us - start_us),
                note="Window must be chosen as complete steady decode iterations; partial boundary copies were excluded",
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Report payload distributions without inferring bandwidth",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--verify-width", type=int)
    parser.add_argument("--window-start-us", type=float)
    parser.add_argument("--window-end-us", type=float)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    if args.probe_only:
        if (
            args.trace is not None
            or args.window_start_us is not None
            or args.window_end_us is not None
            or args.device is not None
        ):
            parser.error(
                "--probe-only accepts geometry filters but no trace/time/device selectors"
            )
    elif args.trace is None:
        parser.error("Provide --trace or --probe-only")
    try:
        probe = json.loads(args.probe.read_text())
        if args.probe_only:
            report = summarize_payload(
                probe, batch_size=args.batch_size, verify_width=args.verify_width
            )
        else:
            report = summarize(
                probe,
                read_events(
                    args.trace, capture_id=probe.get("capture_id"), pid=probe["pid"]
                ),
                device=args.device,
                batch_size=args.batch_size,
                verify_width=args.verify_width,
                start_us=args.window_start_us,
                end_us=args.window_end_us,
            )
        args.output_file.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"samples", "copy_intervals"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
