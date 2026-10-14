import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.utils import hisparse_h2d_probe as hp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

path = Path(__file__).resolve().parents[3] / "scripts/pd_test/collect_h2d_profile.py"
spec = importlib.util.spec_from_file_location("collect_h2d_profile", path)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


@pytest.mark.parametrize(
    "required,reserve,expected",
    [(256, None, 256), (256, "64", 256), (256, "1024", 1024), (257, None, 257)],
)
def test_probe_reserve_covers_request_pool_independently_of_graph_batch(
    required, reserve, expected
):
    assert hp.resolve_probe_max_batch(required, reserve) == expected


def test_probe_factory_reserves_pool_size_before_capture(monkeypatch):
    monkeypatch.setattr(hp, "_probes", {})
    monkeypatch.setenv("SGLANG_HISPARSE_H2D_PROBE_CAPACITY", "16384")
    monkeypatch.setenv("SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH", "64")
    instances = []

    class FakeProbe:
        def __init__(self, device, capacity, max_batch):
            self.max_batch = max_batch
            instances.append((device, capacity, max_batch))

    monkeypatch.setattr(hp, "HiSparseH2DProbe", FakeProbe)
    probe = hp.create_h2d_probe("cuda:0", max_batch=256)
    assert probe.max_batch == 256
    assert instances == [("cuda:0", 16384, 256)]
    with pytest.raises(ValueError, match="cannot be resized"):
        hp.create_h2d_probe("cuda:0", max_batch=512)


@pytest.mark.parametrize(
    "status,kind,body,accepted",
    [
        (200, "text/plain", b"Start profiling.", True),
        (200, "text/plain", b"Start profiling.\n", True),
        (200, "application/json", b'{"success":true}', True),
        (200, "application/json", b'{"success":false,"message":"busy"}', False),
        (200, "text/plain", b"Error: profiler already active", False),
        (503, "text/plain", b"Start profiling.", False),
    ],
)
def test_profile_trigger_handles_text_and_json_without_hiding_errors(
    status, kind, body, accepted
):
    result = collector.decode_profile_response(status, kind, body)
    assert result["accepted"] is accepted
    assert result["body"] == body.decode()


@pytest.mark.parametrize("wait", [False, True])
def test_trigger_saves_text_acknowledgment_and_does_not_retry(
    tmp_path, monkeypatch, wait
):
    calls = []

    class Response:
        status = 200
        headers = {"Content-Type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b"Start profiling.\n"

    def open_response(request, timeout):
        calls.append(request)
        assert timeout == 180
        if wait:
            for rank in range(4):
                write_rank(tmp_path, json.loads(request.data), rank)
        return Response()

    monkeypatch.setattr(collector.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--output-dir", str(tmp_path), "--steps", "7"]
        + (["--wait", "--expected-ranks", "4"] if wait else []),
    )
    collector.main()
    saved = json.loads((tmp_path / "capture-request.json").read_text())
    assert saved["response"]["accepted"]
    assert saved["response"]["body"] == "Start profiling.\n"
    assert saved["request"]["num_steps"] == 7
    assert len(calls) == 1
    assert (tmp_path / "capture-complete.json").exists() == wait


def capture_request(capture_id="h2d-current"):
    return dict(
        profile_id=capture_id,
        profile_prefix="round1",
        activities=["CPU", "GPU"],
        merge_profiles=False,
        profile_by_stage=False,
    )


def write_rank(directory, request, rank, **overrides):
    base = f"round1-{request['profile_id']}-TP-{rank}"
    report = dict(
        schema_version=1,
        capture_id=request["profile_id"],
        pid=4000 + rank,
        device=rank,
        total_records=1,
        overflow=False,
        samples=[{"index": 0}],
    )
    report.update(overrides)
    probe = directory / f"{base}-PID-{4000+rank}-GPU-{rank}.h2d.json"
    probe.write_text(json.dumps(report))
    trace = directory / f"{base}.trace.json.gz"
    with gzip.open(trace, "wt") as stream:
        json.dump({"traceEvents": []}, stream)
    return probe, trace


def fake_clock(monkeypatch, on_sleep=lambda _: None):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        assert 0 < seconds <= 1
        clock.now += seconds
        on_sleep(clock.now)

    monkeypatch.setattr(
        collector, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    return clock


def test_wait_ignores_old_capture_and_partial_probes_until_all_ranks(
    tmp_path, monkeypatch
):
    request = capture_request()
    for rank in range(4):
        write_rank(tmp_path, capture_request("h2d-old"), rank)
    for rank in range(3):
        write_rank(tmp_path, request, rank)
    (tmp_path / "old-decode.log").write_text("Profiling done\n" * 4)
    status = collector.capture_artifacts(tmp_path, request, 4)
    assert len(status["ranks"]) == 3 and not status["artifacts_complete"]

    def finish_later(now):
        assert not (tmp_path / "capture-complete.json").exists()
        probe, _ = write_rank(tmp_path, request, 3)
        if now == 1:
            probe.write_text('{"schema_version":')  # Writer has not finished.

    clock = fake_clock(monkeypatch, finish_later)
    result = collector.wait_for_capture(tmp_path, request, 4, 5)
    assert clock.now == 2 and result["artifacts_complete"]
    saved = json.loads((tmp_path / "capture-complete.json").read_text())
    assert saved["capture_id"] == "h2d-current"
    assert [r["rank"] for r in saved["ranks"]] == [0, 1, 2, 3]
    assert not saved["timing_analysis_validated"]  # Completion is not analysis.


def test_timeout_keeps_diagnostic_and_never_creates_completion_marker(
    tmp_path, monkeypatch
):
    fake_clock(monkeypatch)
    with pytest.raises(TimeoutError, match="wait-only"):
        collector.wait_for_capture(tmp_path, capture_request(), 4, 2.5)
    assert not (tmp_path / "capture-complete.json").exists()
    saved = json.loads((tmp_path / "capture-incomplete.json").read_text())
    assert not saved["artifacts_complete"]
    assert len(saved["pending"]) == 4
    assert saved["wait_elapsed_seconds"] == 2.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"capture_id": "wrong"},
        {"pid": 99},
        {"device": 3},
        {"overflow": True},
        {"samples": [], "total_records": 0},
        {"total_records": 2},
    ],
)
def test_invalid_finished_probe_is_rejected(tmp_path, overrides):
    request = capture_request()
    write_rank(tmp_path, request, 0, **overrides)
    with pytest.raises(ValueError):
        collector.capture_artifacts(tmp_path, request, 1)


def test_probe_requires_nonempty_trace_and_unambiguous_rank(tmp_path):
    request = capture_request()
    _, trace = write_rank(tmp_path, request, 0)
    trace.write_bytes(b"")
    assert not collector.capture_artifacts(tmp_path, request, 1)["artifacts_complete"]
    trace.write_bytes(b"trace")
    trace.with_name(trace.name.replace(".trace", "-EP-0.trace")).write_bytes(b"trace")
    with pytest.raises(ValueError, match="ambiguous"):
        collector.capture_artifacts(tmp_path, request, 1)


def test_wait_only_reads_saved_request_without_posting_again(tmp_path, monkeypatch):
    request = capture_request()
    (tmp_path / "capture-request.json").write_text(
        json.dumps(dict(request=request, response={"accepted": True}))
    )
    for rank in range(4):
        write_rank(tmp_path, request, rank)

    def never_post(*args, **kwargs):
        pytest.fail("Resuming a wait must not send another /start_profile request")

    monkeypatch.setattr(collector.urllib.request, "urlopen", never_post)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--output-dir",
            str(tmp_path),
            "--wait-only",
            "--expected-ranks",
            "4",
        ],
    )
    collector.main()
    assert (tmp_path / "capture-complete.json").exists()
    monkeypatch.setattr(sys, "argv", ["collector", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code != 0


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--expected-ranks", "0"],
        ["--expected-ranks", "4", "--activity", "nsys"],
        ["--expected-ranks", "4", "--timeout-seconds", "nan"],
    ],
)
def test_wait_requires_explicit_valid_scope(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(
        sys, "argv", ["collector", "--output-dir", str(tmp_path), "--wait", *extra]
    )
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code != 0
    assert not (tmp_path / "capture-request.json").exists()
