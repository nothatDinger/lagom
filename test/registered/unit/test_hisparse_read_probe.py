import copy
import importlib.util
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.benchmark.hisparse_read_probe import (
    check_result,
    check_wave_result,
    compact_report,
    make_chains,
    time_stats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def capture_runtime(monkeypatch):
    """Exercise the real torch.cuda.graph Python context, without CUDA calls."""
    scripts = Path(__file__).resolve().parents[3] / "scripts/pd_test"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "read_bench", scripts / "bench_h2d_reads.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    state = SimpleNamespace(device=0, streams={})

    def stream(device=None):
        return SimpleNamespace(
            device=torch.device("cuda", state.device if device is None else device)
        )

    state.streams = {device: stream(device) for device in (0, 1)}

    @contextmanager
    def select_device(device):
        previous = state.device
        state.device = device
        try:
            yield
        finally:
            state.device = previous

    @contextmanager
    def select_stream(selected):
        device = selected.device.index
        previous = state.streams[device]
        with select_device(device):
            state.streams[device] = selected
            try:
                yield
            finally:
                state.streams[device] = previous

    class Event:
        def __init__(self, **kwargs):
            self.device = None

        def record(self):
            self.device = torch.device("cuda", state.device)

    monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", None)
    monkeypatch.setattr(torch.cuda, "Stream", stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "device", select_device)
    monkeypatch.setattr(torch.cuda, "stream", select_stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state.device)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: state.streams[state.device]
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: None, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "CUDAGraph",
        lambda: SimpleNamespace(
            capture_begin=lambda **kwargs: None,
            capture_end=lambda: None,
        ),
    )
    return bench, state, select_device


def test_second_device_capture_does_not_reuse_first_device_default(capture_runtime):
    bench, state, select_device = capture_runtime
    launches = []
    # Reproduce the old helper call with the installed PyTorch graph context:
    # a device-1 caller still enters the default device-0 capture stream.
    for device in (0, 1):
        with select_device(device):
            bench.capture_wave(torch, [lambda: launches.append(state.device)])
    assert launches == [0, 0]
    launches.clear()
    for device in (0, 1):
        with select_device(device):
            _, events, stream, check = bench.capture_on_device(
                torch, [lambda: launches.append(state.device)], device
            )
        assert stream.device == torch.device("cuda", device)
        assert all(e.device == stream.device for pair in events for e in pair)
        assert check["device_checked"]
    assert launches == [0, 1]


def test_wrong_capture_context_stops_before_ffi_launch(capture_runtime, monkeypatch):
    bench, state, select_device = capture_runtime
    monkeypatch.setattr(torch.cuda, "stream", lambda _: select_device(0))
    launches = []
    with pytest.raises(RuntimeError, match="Capture launch context"):
        bench.capture_on_device(torch, [lambda: launches.append(state.device)], 1)
    assert launches == []


def test_wrong_event_device_rejected_before_replay(capture_runtime, monkeypatch):
    bench, _, _ = capture_runtime
    monkeypatch.setattr(
        torch.cuda.Event,
        "record",
        lambda event: setattr(event, "device", torch.device("cuda", 0)),
    )
    with pytest.raises(RuntimeError, match="Captured events"):
        bench.capture_on_device(torch, [lambda: None], 1)


@pytest.mark.parametrize("chains", [1, 4, 16, 32, 64, 128])
def test_rings_visit_same_unique_set_and_match_independent_oracle(chains):
    order = random.Random(58).sample(range(1024), 256)
    plan = make_chains(order, chains)
    successor = dict(zip(plan["nodes"], plan["successors"]))
    seen, rows = [], [[-1] * 4]
    for start in plan["starts"]:
        index, checksum = start, 0
        for _ in range(plan["steps"]):
            index = successor[index]
            checksum += index
            seen.append(index)
        rows.append([index, checksum, 141, 160])  # IDs need not be dense.
    rows.append([-1] * 4)
    assert sorted(seen) == sorted(order)
    assert len(set(seen)) == len(order)
    assert check_result(rows, plan["expected"])["sm_ids"] == {141: chains}
    assert plan["node_ids_sha256"] == make_chains(order, 1)["node_ids_sha256"]


@pytest.mark.parametrize(
    "order,chains", [([], 1), ([1, 1], 1), ([-1, 2], 1), ([1, 2], 0), ([1, 2, 3], 2)]
)
def test_invalid_chain_layout_rejected(order, chains):
    with pytest.raises(ValueError):
        make_chains(order, chains)


@pytest.mark.parametrize(
    "row,column,value",
    [(0, 1, 0), (3, 0, 0), (1, 0, 9), (1, 1, 99), (2, 2, 200), (2, 3, 0)],
)
def test_wrong_result_or_guard_rejected(row, column, value):
    rows = [[-1] * 4, [3, 8, 10, 144], [7, 18, 30, 144], [-1] * 4]
    expected = [[3, 8], [7, 18]]
    check_result(rows, expected)
    rows[row][column] = value
    with pytest.raises(ValueError):
        check_result(rows, expected)


def test_read_units_and_wave_means_do_not_claim_bandwidth():
    # 128 total loads, four chains, 32 dependent hops per chain.
    stats = time_stats([10, 30], 128, 4)
    assert stats["mean_event_us"] == 20
    assert stats["ns_per_chain_hop"] == 625
    assert stats["aggregate_Mloads_s"] == 6.4
    report = dict(
        success=True,
        correctness=[],
        expected_correctness=0,
        config=dict(devices=[0, 1], total_reads=128),
        code={},
        devices={},
        waves=[],
    )
    for device, times in [(0, [10, 30]), (1, [20, 60])]:
        report["waves"].append(
            dict(
                stride=128,
                blocks=1,
                warps=4,
                source="host",
                device=device,
                event_us=times,
                before={},
                after={},
            )
        )
    result = compact_report(report)
    assert "Mloads/s=6.400" in result
    assert "d1/d0=2.0000" in result
    assert "no PCIe GB/s estimate" in result
    failed = copy.deepcopy(report)
    failed.update(success=False, error="fixture failure", waves=[])
    assert "ERROR fixture failure" in compact_report(failed)
    failed.update(
        expected_correctness=16,
        correctness=[dict(stride=128, blocks=1, warps=1, source="host", device=0)],
        failure_context=dict(device=1, stage="graph_check", eager_passed=True),
    )
    text = compact_report(failed)
    assert "correctness=1/16" in text
    assert text.count("COMPLETED ") == 1
    assert "FAILURE_CONTEXT" in text and "graph_check" in text
    assert "CAPTURE_DEVICE_CHECK 0/16" in text  # Old results are not re-certified.


def test_wave_checks_every_kernel_output_not_only_the_last_one():
    expected = [[3, 8]]
    good = [[-1] * 4, [3, 8, 141, 160], [-1] * 4]
    wave = [copy.deepcopy(good) for _ in range(3)]
    assert check_wave_result(wave, expected)["samples_checked"] == 3
    wave[1][1][1] = -1
    with pytest.raises(ValueError):
        check_wave_result(wave, expected)


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_bad_cuda_event_times_rejected(samples):
    with pytest.raises(ValueError):
        time_stats(samples, 128, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
