import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.benchmark.hisparse_h2d import read_events, summarize, summarize_payload
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def fixtures():
    rows = []
    events = []
    for i, (counts, duration) in enumerate([([20, 40], 20), ([10, 30], 10)]):
        rows.append(
            dict(
                index=i,
                layer=i,
                verify_width=6,
                real_batch=2,
                captured_batch=4,
                item_bytes=584,
                copy_blocks=4,
                copy_threads=1024,
                skip_io=False,
                gpu_stamp_ns=1000 + i,
                misses_per_request=counts,
                unique_misses=sum(counts),
                h2d_bytes=sum(counts) * 584,
            )
        )
        for name, ts, dur in [
            ("copy_cache_planned_kernel<1024>", i * 100, duration),
            ("hisparse_h2d_record_kernel", i * 100 + duration, 2),
        ]:
            events.append(
                dict(
                    ph="X",
                    cat="kernel",
                    name=name,
                    ts=ts,
                    dur=dur,
                    args=dict(device=0, stream=7),
                )
            )
    return (
        dict(
            schema_version=1,
            pid=123,
            gpu="H100",
            total_records=2,
            overflow=False,
            samples=rows,
        ),
        events,
    )


def test_weighted_bandwidth_uses_real_payload_and_gpu_time():
    probe, events = fixtures()
    result = summarize(probe, events)
    assert result["summary"]["h2d_bytes"] == 58400
    assert result["summary"]["copy_GBps"] == pytest.approx(58400 / 30 / 1000)
    assert result["summary"]["recorder_gpu_us"] == 4
    assert result["geometry"][0]["batch"] == 2
    assert len(result["samples"]) == 2


@pytest.mark.parametrize(
    "corruption",
    [
        "overflow",
        "missing_node",
        "wrong_bytes",
        "wrong_counts",
        "wrong_order",
        "two_streams",
        "skip_io",
        "no_copy",
        "zero_duration",
    ],
)
def test_rejects_ambiguous_or_invalid_measurement(corruption):
    probe, events = fixtures()
    if corruption == "overflow":
        probe["overflow"] = True
    elif corruption == "missing_node":
        events.pop()
    elif corruption == "wrong_bytes":
        probe["samples"][0]["h2d_bytes"] += 1
    elif corruption == "wrong_counts":
        probe["samples"][0]["unique_misses"] += 1
    elif corruption == "wrong_order":
        probe["samples"][1]["gpu_stamp_ns"] = 1
    elif corruption == "two_streams":
        events[-1]["args"]["stream"] = 8
    elif corruption == "skip_io":
        probe["samples"][0]["skip_io"] = True
        probe["samples"][0]["h2d_bytes"] = 0
    elif corruption == "no_copy":
        events[0]["name"] = "attention"
    elif corruption == "zero_duration":
        events[0]["dur"] = 0
    with pytest.raises(ValueError):
        summarize(probe, events)


def test_window_does_not_clip_duration_and_keep_full_bytes():
    probe, events = fixtures()
    result = summarize(probe, events, start_us=5, end_us=120)
    assert len(result["samples"]) == 1
    assert result["selected_window"]["payload_GBps"] == pytest.approx(
        23360 / 115 / 1000
    )
    filtered = summarize(probe, events, batch_size=2, start_us=0, end_us=120)
    assert "selected_window" not in filtered


def test_multiple_gpus_are_not_summed():
    probe, events = fixtures()
    other = copy.deepcopy(events)
    for event in other:
        event["args"]["device"] = 1
    with pytest.raises(ValueError, match="one GPU"):
        summarize(probe, events + other)
    assert summarize(probe, events + other, device=0)["summary"]["h2d_bytes"] == 58400


def test_sqlite_nanoseconds_are_converted_to_microseconds(tmp_path):
    path = tmp_path / "nsys.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
        db.execute("INSERT INTO StringIds VALUES(1, 'copy_cache_planned_kernel')")
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, "
            "deviceId INTEGER, streamId INTEGER, demangledName INTEGER, shortName INTEGER)"
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100000, 120000, 0, 7, 1, 1)"
        )
    event = read_events(path)[0]
    assert event["ts"] == 100
    assert event["dur"] == 20


def test_trace_identity_rejects_another_capture_or_rank(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            dict(
                traceEvents=[], hisparse_h2d_identity=dict(capture_id="run-a", pid=123)
            )
        )
    )
    assert read_events(path, capture_id="run-a", pid=123) == []
    with pytest.raises(ValueError, match="identities"):
        read_events(path, capture_id="run-b", pid=123)
    with pytest.raises(ValueError, match="identities"):
        read_events(path, capture_id="run-a", pid=124)


def test_serialized_recorders_on_different_streams_are_paired():
    probe, events = fixtures()
    events[2]["args"]["stream"] = 312
    events[3]["args"]["stream"] = 312
    result = summarize(probe, events)
    assert result["pairing"]["mode"] == "serialized_recorders"
    assert result["pairing"]["recorder_streams"] == [312, 7]
    assert result["samples"][1]["copy_stream"] == 312
    assert result["layers"]["1"]["copy_gpu_us"] == 10
    assert result["summary"]["copy_gpu_overlap_us"] == 0


def test_pairing_follows_recorder_order_rather_than_copy_start_order():
    probe, events = fixtures()
    events[2].update(ts=-5, dur=30)
    events[3].update(ts=25, dur=2)
    events[2]["args"]["stream"] = 312
    events[3]["args"]["stream"] = 312
    result = summarize(probe, events)
    assert result["samples"][0]["copy_us"] == 20
    assert result["samples"][1]["copy_us"] == 30
    assert result["summary"]["copy_gpu_us"] == 50
    assert result["summary"]["copy_gpu_union_us"] == 30
    assert result["summary"]["copy_gpu_overlap_us"] == 20


def overlapping_fixtures():
    probe, events = fixtures()
    events[1]["dur"] = 10
    events[2].update(ts=5, dur=20)
    events[3].update(ts=25, dur=10)
    events[2]["args"]["stream"] = 312
    events[3]["args"]["stream"] = 312
    # Counter increments can precede a stamp on another concurrently executing
    # recorder; no relation between row index and kernel start is assumed.
    probe["samples"][1]["gpu_stamp_ns"] = 500
    return probe, events


def test_overlapping_recorders_report_totals_without_layer_timing():
    probe, events = overlapping_fixtures()
    result = summarize(probe, events)
    assert result["pairing"]["mode"] == "capture_totals_only"
    assert result["summary"]["h2d_bytes"] == 58400
    assert result["summary"]["copy_gpu_us"] == 40
    assert result["summary"]["copy_gpu_union_us"] == 25
    assert result["summary"]["copy_GBps"] == pytest.approx(58400 / 40 / 1000)
    assert result["summary"]["copy_active_wall_GBps"] == pytest.approx(
        58400 / 25 / 1000
    )
    assert "copy_gpu_us" not in result["layers"]["0"]
    assert "copy_us" not in result["samples"][0]


def test_overlapping_recorders_do_not_assign_filtered_bytes_to_times():
    probe, events = overlapping_fixtures()
    with pytest.raises(ValueError, match="whole-capture"):
        summarize(probe, events, start_us=1)
    probe["samples"][1]["real_batch"] = 3
    probe["samples"][1]["misses_per_request"] = [10, 15, 15]
    with pytest.raises(ValueError, match="whole-capture"):
        summarize(probe, events, batch_size=2)


def test_probe_only_retains_payload_without_inventing_timing():
    probe, _ = fixtures()
    result = summarize_payload(probe)
    assert result["summary"]["h2d_bytes"] == 58400
    assert result["captured_geometry"][0]["captured_batch"] == 4
    assert "copy_GBps" not in result["summary"]


@pytest.mark.parametrize("probe_only", [False, True])
def test_offline_cli_analyzes_existing_multistream_files(tmp_path, probe_only):
    probe, events = overlapping_fixtures()
    probe_path, trace_path = tmp_path / "probe.json", tmp_path / "trace.json"
    output = tmp_path / "report.json"
    probe_path.write_text(json.dumps(probe))
    trace_path.write_text(json.dumps(dict(traceEvents=events)))
    script = (
        Path(__file__).resolve().parents[3] / "python/sglang/benchmark/hisparse_h2d.py"
    )
    args = [
        sys.executable,
        str(script),
        "--probe",
        str(probe_path),
        "--output-file",
        str(output),
    ]
    args += ["--probe-only"] if probe_only else ["--trace", str(trace_path)]
    run = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    assert result["summary"]["h2d_bytes"] == 58400
    assert ("copy_GBps" in result["summary"]) is (not probe_only)
