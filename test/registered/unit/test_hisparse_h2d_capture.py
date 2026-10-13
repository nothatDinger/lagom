import importlib.util
import json
import sys
from pathlib import Path

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


def test_trigger_saves_text_acknowledgment_and_does_not_retry(tmp_path, monkeypatch):
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
        return Response()

    monkeypatch.setattr(collector.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(
        sys, "argv", ["collector", "--output-dir", str(tmp_path), "--steps", "7"]
    )
    collector.main()
    saved = json.loads((tmp_path / "capture-request.json").read_text())
    assert saved["response"]["accepted"]
    assert saved["response"]["body"] == "Start profiling.\n"
    assert saved["request"]["num_steps"] == 7
    assert len(calls) == 1
