"""Join real verify miss bytes to GPU copy durations, one rank at a time.

Accepts a torch Chrome trace (.json/.json.gz) or an Nsight Systems SQLite export.
Uses only the standard library. A strict join rejects missing graph-node traces,
overflow, or concurrent recorder streams rather than reporting guessed GB/s.
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


def _summarize(samples):
    byte_count = sum(s["h2d_bytes"] for s in samples)
    copy_us = sum(s["copy_us"] for s in samples)
    marker_us = sum(s["recorder_us"] for s in samples)
    return dict(
        samples=len(samples),
        h2d_bytes=byte_count,
        copy_gpu_us=copy_us,
        copy_GBps=byte_count / copy_us / 1000,
        recorder_gpu_us=marker_us,
        recorder_over_copy_pct=100 * marker_us / copy_us,
        copy_us=distribution([s["copy_us"] for s in samples]),
        bytes_per_copy=distribution([s["h2d_bytes"] for s in samples]),
        unique_misses_per_request=distribution(
            [m for s in samples for m in s["misses_per_request"]]
        ),
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
    if probe.get("schema_version") != 1:
        raise ValueError("Unsupported H2D probe schema")
    if probe.get("overflow"):
        raise ValueError("Probe overflow; increase capacity and recollect")
    rows = probe["samples"]
    if not rows or probe["total_records"] != len(rows):
        raise ValueError("Probe has no complete samples")
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
    streams = {e.get("args", {}).get("stream") for e in markers}
    if None in streams or len(streams) != 1:
        raise ValueError(
            "This first-round join requires one synchronous verify recorder stream; "
            "retain the raw trace for analysis of multiple streams"
        )
    stream_id = next(iter(streams))
    timeline = [e for e in kernels if e.get("args", {}).get("stream") == stream_id]
    pairs = []
    previous = None
    for event in timeline:
        if "hisparse_h2d_record_kernel" in event["name"]:
            if previous is None or "copy_cache_planned_kernel" not in previous["name"]:
                raise ValueError(
                    "Recorder is not immediately after a copy on its stream"
                )
            pairs.append((previous, event))
        previous = event
    joined = []
    stamps = []
    for index, (sample, (copy, marker)) in enumerate(zip(rows, pairs)):
        if sample["index"] != index:
            raise ValueError("Non-contiguous probe record indices")
        stamps.append(sample["gpu_stamp_ns"])
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
        for event in (copy, marker):
            if (
                not math.isfinite(event["ts"])
                or not math.isfinite(event["dur"])
                or event["dur"] <= 0
            ):
                raise ValueError("Invalid GPU kernel timestamps or durations")
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
                copy_start_us=copy["ts"],
                copy_us=copy["dur"],
                recorder_us=marker["dur"],
            )
        )
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Probe timestamps are not in stream order")
    if not joined:
        raise ValueError("No complete copies match the selected window/geometry")
    groups = defaultdict(list)
    layers = defaultdict(list)
    for sample in joined:
        groups[(sample["real_batch"], sample["verify_width"])].append(sample)
        layers[sample["layer"]].append(sample)
    report = dict(
        device=device,
        pid=probe["pid"],
        gpu=probe["gpu"],
        units="GB/s = 10^9 bytes/s; GPU trace timestamps are microseconds",
        scope="Actual miss payload / copy-kernel GPU duration, one rank; excludes planner, launch gaps and recorder. This is not PCIe wire bandwidth or marginal decode latency.",
        summary=_summarize(joined),
        geometry=[
            dict(batch=b, verify_width=w, **_summarize(samples))
            for (b, w), samples in sorted(groups.items())
        ],
        layers={
            str(layer): _summarize(samples) for layer, samples in sorted(layers.items())
        },
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
                * report["summary"]["copy_gpu_us"]
                / (end_us - start_us),
                note="Window must be chosen as complete steady decode iterations; partial boundary copies were excluded",
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--verify-width", type=int)
    parser.add_argument("--window-start-us", type=float)
    parser.add_argument("--window-end-us", type=float)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        probe = json.loads(args.probe.read_text())
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
            {key: value for key, value in report.items() if key != "samples"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
