"""Concurrency caps, variable decode batches, TP accounting, and load generation."""

import copy
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.observability import lagom_decode_perf, lagom_perf
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmark/lagom"))
    spec = importlib.util.spec_from_file_location(
        "lagom_concurrent_cli", ROOT / "benchmark/lagom/run_concurrent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(bench, *extra):
    args = bench.parser().parse_args(
        [
            "--model",
            "target",
            "--draft",
            "draft",
            "--sharegpt",
            "data.json",
            "--output",
            "result",
            *extra,
        ]
    )
    bench.validate(args)
    return args


@pytest.mark.parametrize(
    "size,graph_sizes",
    [
        (8, [1, 2, 4, 8]),
        (16, [1, 2, 4, 8, 16]),
        (32, [1, 2, 4, 8, 16, 32]),
        (64, [1, 2, 4, 8, 16, 32, 64]),
    ],
)
def test_concurrency_cap_commands_keep_all_baselines_comparable(
    bench, size, graph_sizes
):
    args = settings(bench, "--max-concurrency", str(size))
    assert (
        args.phase == "throughput"
        and args.input_len == 32000
        and args.output_len == 1024
        and args.requests == 16
    )
    assert args.trace_modes == list(bench.common.MODES)
    for mode in bench.common.MODES:
        cmd = bench.common.server_command(args, mode, 30000, max_running_requests=size)
        assert cmd[cmd.index("--max-running-requests") + 1] == str(size)
        assert json.loads(cmd[cmd.index("--cuda-graph-config") + 1]) == {
            "decode": {"bs": graph_sizes}
        }
        assert cmd[cmd.index("--speculative-num-draft-tokens") + 1] == "8"
        assert ("--enable-hisparse" in cmd) == (mode != "dspark")


@pytest.mark.parametrize("hot", [7680, 8192])
def test_hisparse_and_lagom_share_cache_capacity_and_other_cache_controls(bench, hot):
    args = settings(bench, "--hot", str(hot))
    configs = {}
    for mode in ("lagom", "hisparse"):
        command = bench.common.server_command(args, mode, 30000, max_running_requests=8)
        configs[mode] = json.loads(command[command.index("--hisparse-config") + 1])
    lagom = configs["lagom"]
    assert lagom.pop("prefetcher") == "lagom"
    assert lagom.pop("prefetcher_config") == {"prediction_steps": args.p}
    assert args.p == 4
    assert (
        lagom
        == configs["hisparse"]
        == {
            "device_buffer_size": hot,
            "host_to_device_ratio": 2,
            "force_h2d_each_step": False,
        }
    )


def test_decode_warmup_retains_old_cli_spelling_as_alias(bench):
    assert settings(bench, "--decode-warmup", "2").decode_warmup == 2
    assert settings(bench, "--full-batch-warmup", "2").decode_warmup == 2


def test_window_allows_batch_and_cohort_changes_but_restarts_on_prefill():
    window = lagom_decode_perf.DecodeWindow(8, steps=3, warmup=1)
    ids = list(map(str, range(8)))
    assert window.select(ids[:7], True) is None  # Warm up below the cap.
    assert window.select(ids[:6], True) == 0
    assert window.select(ids, False) is None  # Interleaved prefill discards step 0.
    assert window.discarded == 1
    assert window.select(ids[:1], True) is None
    assert window.select(ids[:7], True) == 0
    changed = ids[:-1] + ["replacement"]
    assert window.select(changed, True) == 1
    assert window.select(ids[:3], True) == 2
    assert window.discarded == 1
    window.complete = True
    assert window.select(ids[:2], True) is None


def test_diagnostics_allow_below_cap_and_reject_only_budget_or_invalid_batches():
    diagnostics = lagom_decode_perf.DecodeDiagnostics(8, 8, "r")
    ids = [f"r{i}" for i in range(8)]
    assert diagnostics.observe(ids[:7], [900] * 7, True, 1)
    assert not diagnostics.observe(ids, [16] + [900] * 7, True, 2)
    assert diagnostics.observe(ids, [17] + [900] * 7, True, 3)
    assert diagnostics.observe(ids[:6], [17] + [900] * 5, True, 4)
    assert not diagnostics.observe(ids, [900] * 8, False, 5)
    assert diagnostics.observe(ids, [900] * 8, True, 6)
    assert not diagnostics.observe(ids + ["r8"], [900] * 9, True, 7)
    data = diagnostics.snapshot()
    assert data["benchmark_decode_batch_histogram"] == {7: 1, 8: 3, 6: 1, 9: 1}
    assert data["at_cap_decode_forwards_before_budget_filter"] == 3
    assert data["eligible_decode_forwards"] == 4
    assert data["longest_eligible_decode_streak"] == 2
    assert data["first_decode_at_cap"]["remaining_output_budget_cpu"][0] == 16
    assert data["rejected_forward_counts"] == {
        "batch_size_out_of_range": 1,
        "output_headroom": 1,
        "not_pure_decode": 1,
    }


def test_final_status_is_not_throttled_and_counts_batch_before_budget_filter(
    bench, tmp_path
):
    meter = object.__new__(lagom_decode_perf.DecodeMeter)
    meter.window = lagom_decode_perf.DecodeWindow(8, 32, 4)
    meter.width, meter.request_prefix, meter.flush_request_id = 8, "r", "flush"
    meter.diagnostics = lagom_decode_perf.DecodeDiagnostics(8, 8, "r")
    meter.poll = Mock()
    meter.written, meter.active = False, None
    meter.forwards, meter.observed_max_batch = 0, 0
    meter.last_progress = time.monotonic() + 60  # Force the ordinary path to throttle.
    meter.stem = tmp_path / "rank-test"

    def batch(ids, remaining):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            is_extend_in_batch=False,
            forward_iter=10,
            reqs=[
                SimpleNamespace(
                    rid=rid,
                    sampling_params=SimpleNamespace(max_new_tokens=remaining),
                    output_ids=[],
                )
                for rid in ids
            ],
        )

    meter.begin(batch([f"r{i}" for i in range(8)], 16))
    assert not meter.stem.with_suffix(".status.json").exists()
    meter.begin(batch(["flush"], 8))
    data = json.loads(meter.stem.with_suffix(".status.json").read_text())
    assert data["cohort_finished"]
    assert data["observed_max_decode_batch_size"] == 8
    assert data["at_cap_decode_forwards_before_budget_filter"] == 1
    assert data["eligible_decode_forwards"] == 0
    message = bench.window_diagnostics(tmp_path)
    assert "max_decode_batch=8" in message and "eligible_decode_forwards=0" in message


def reports():
    records = []
    for i, (start, end) in enumerate(((0, 4), (6, 12))):
        ids = (
            [f"r{j}" for j in range(7)]
            if i == 0
            else [f"r{j}" for j in (0, 1, 2, 3, 4, 7)]
        )
        size = len(ids)
        records.append(
            {
                "step": i,
                "forward_iter": 10 + i,
                "batch_size": size,
                "request_ids": ids,
                "accepted_tokens": [2] * size,
                "remaining_output_budget_cpu": [900] * size,
                "cuda_graph": True,
                "draft_cuda_graph": True,
                "start_ms": start,
                "end_ms": end,
                "step_ms": end - start,
                "cpu_submit_ms": 1,
            }
        )
    first = {
        "schema_version": 2,
        "records": records,
        "max_batch_size": 8,
        "window_ms": 12,
        "verify_width": 8,
    }
    return [first, copy.deepcopy(first)]


def test_tp_tokens_count_once_and_window_includes_scheduler_gaps(bench):
    data = reports()
    summary, steps = bench.aggregate_decode(
        data, mode="lagom", max_batch_size=8, steps=2
    )
    assert summary["output_tokens"] == 26  # Actual batch 7 + 6, never reaches cap 8.
    assert summary["decode_tokens_per_s"] == pytest.approx(26000 / 12)
    assert summary["gpu_forward_tokens_per_s"] == 2600
    assert summary["mean_running_batch_size"] == 6.5
    assert summary["max_running_batch_size"] == 7
    assert summary["min_running_batch_size"] == 6
    assert json.loads(summary["running_batch_histogram"]) == {"6": 1, "7": 1}
    assert summary["mean_accepted_tokens_per_request_per_step"] == 2
    assert "mean_per_request_tokens_per_s" not in summary
    assert [s["running_batch_size"] for s in steps] == [7, 6]
    assert steps[1]["inter_step_gap_ms_max_rank"] == 2
    by_batch = bench.group_steps_by_batch_size(steps)
    assert [r["running_batch_size"] for r in by_batch] == [6, 7]
    assert [r["output_tokens"] for r in by_batch] == [12, 14]
    assert [r["gpu_forward_tokens_per_s"] for r in by_batch] == [2000, 3500]


def test_full_verify_acceptance_is_valid(bench):
    data = reports()
    for report in data:
        for record in report["records"]:
            record["accepted_tokens"] = [report["verify_width"]] * record["batch_size"]
    summary, _ = bench.aggregate_decode(data, mode="dspark", max_batch_size=8, steps=2)
    assert summary["output_tokens"] == 104


@pytest.mark.parametrize(
    "case",
    [
        "size_mismatch",
        "above_cap",
        "empty",
        "different_cohort",
        "eager",
        "different_step",
        "step_gap",
        "different_accept",
        "output_cap",
    ],
)
def test_invalid_measurements_fail(bench, case):
    data = reports()
    record = data[1]["records"][0]
    if case == "size_mismatch":
        record["batch_size"] = 6
    elif case == "above_cap":
        record["batch_size"] = 9
    elif case == "empty":
        record["batch_size"] = 0
    elif case == "different_cohort":
        record["request_ids"][0] = "another-request"
    elif case == "eager":
        record["draft_cuda_graph"] = False
    elif case == "different_step":
        record["forward_iter"] = 2
    elif case == "step_gap":
        for report in data:
            report["records"][1]["forward_iter"] += 1
    elif case == "different_accept":
        record["accepted_tokens"][0] = 3
    else:
        record["remaining_output_budget_cpu"][0] = 16
    with pytest.raises(ValueError):
        bench.aggregate_decode(data, mode="dspark", max_batch_size=8, steps=2)


@pytest.mark.parametrize("requests", [3, 8, 19])
def test_request_count_is_independent_and_clients_replenish_without_wave_barriers(
    bench, monkeypatch, tmp_path, requests
):
    args = settings(bench, "--requests", str(requests), "--max-concurrency", "8")
    clients = min(requests, 8)
    lock = threading.Lock()
    arrived, active, peak = threading.Event(), 0, 0
    replenished = threading.Event()
    seen = set()

    def generate(url, tokens, output_len, timeout, *, rid):
        nonlocal active, peak
        index = int(rid.removeprefix(bench.REQUEST_PREFIX))
        with lock:
            seen.add(rid)
            active += 1
            peak = max(peak, active)
            assert active <= clients
            if active == clients:
                arrived.set()
        assert arrived.wait(5), "requests were serialized instead of submitted together"
        if index >= clients:
            replenished.set()
        if index == 0 and requests > clients:
            assert replenished.wait(
                5
            ), "next request waited for the entire initial wave to finish"
        with lock:
            active -= 1
        return {"output_tokens": output_len, "ttft_s": 0.001}

    monkeypatch.setattr(bench.common, "generate", generate)
    rows = [{"input_ids": [i], "token_sha256": str(i)} for i in range(requests + 1)]
    bench.run_cohort(args, "http://unused", rows, tmp_path, time.monotonic() + 10)
    assert peak == clients and len(seen) == requests
    results = json.loads((tmp_path / "requests.json").read_text())
    assert len(results) == requests
    assert sorted(r["request_index"] for r in results) == list(range(requests))
    assert all(r["first_token_s"] == r["submitted_s"] + r["ttft_s"] for r in results)
    cohort = json.loads((tmp_path / "cohort.json").read_text())
    assert cohort["requests"] == cohort["requested_requests"] == requests
    assert cohort["max_concurrency"] == 8
    assert cohort["client_concurrency_limit"] == clients
    assert cohort["load_pattern"] == "replenish_on_completion"
    assert cohort["first_token_spread_s"] == max(
        r["first_token_s"] for r in results
    ) - min(r["first_token_s"] for r in results)
    initial = [r for r in results if r["request_index"] < clients]
    assert cohort["initial_wave_first_token_spread_s"] == max(
        r["first_token_s"] for r in initial
    ) - min(r["first_token_s"] for r in initial)


@pytest.mark.parametrize("requests", [0, -1])
def test_request_count_must_be_positive(bench, requests):
    with pytest.raises(ValueError, match="request count must be positive"):
        settings(bench, "--requests", str(requests))


def test_meter_defers_device_reads_and_discards_prefill_interruption(
    monkeypatch, tmp_path
):
    clock, ready = [0.0], [False]

    class Event:
        def record(self):
            self.stamp = clock[0]

        def query(self):
            return ready[0]

        def elapsed_time(self, other):
            assert ready[0]
            return other.stamp - self.stamp

    class DeviceValues:
        def __init__(self, values):
            self.values = list(values)

        def detach(self):
            return self

        def clone(self):
            return DeviceValues(self.values)

        def cpu(self):
            assert ready[0], "D2H before the bounded window completed"
            return self

        def tolist(self):
            assert ready[0]
            return self.values

    meter = object.__new__(lagom_decode_perf.DecodeMeter)
    meter.window = lagom_decode_perf.DecodeWindow(8, 2, 1)
    meter.width, meter.request_prefix = 8, "r"
    meter.flush_request_id = "flush"
    meter.diagnostics = lagom_decode_perf.DecodeDiagnostics(8, 8, "r")
    meter.starts, meter.ends, meter.done = (
        [Event(), Event()],
        [Event(), Event()],
        Event(),
    )
    meter.records, meter.accepted, meter.prefixes = [], [], []
    meter.active, meter.written = None, False
    meter.forwards, meter.observed_max_batch = 0, 0
    meter.device, meter.stem = 0, tmp_path / "rank-test"
    meter.progress = Mock()
    monkeypatch.setattr(lagom_decode_perf, "get_meter", lambda: meter)
    monkeypatch.setattr(lagom_perf, "get_meter", lambda: meter)
    monkeypatch.setattr(lagom_perf, "get_trace", lambda: None)
    result = SimpleNamespace(can_run_cuda_graph=True, accept_lens=DeviceValues([2] * 8))

    @lagom_decode_perf.measure_decode
    def forward(self, batch):
        lagom_perf.note_draft_graph(True)
        clock[0] += batch.work_ms
        result.accept_lens = DeviceValues([2] * len(batch.reqs))
        batch.seq_lens.values[:] = [99999] * len(batch.reqs)
        return result

    for i, (decode, size, work) in enumerate(
        [
            (False, 7, 1),
            (True, 7, 1),
            (True, 7, 4),
            (False, 8, 100),
            (True, 6, 1),
            (True, 7, 4),
            (True, 6, 6),
        ]
    ):
        if i == 6:
            clock[0] += 2
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda decode=decode: decode),
            is_extend_in_batch=False,
            reqs=[
                SimpleNamespace(
                    rid=f"r{j}",
                    sampling_params=SimpleNamespace(max_new_tokens=1024),
                    output_ids=[],
                )
                for j in range(size)
            ],
            forward_iter=i,
            seq_lens_cpu=None,
            seq_lens=DeviceValues([32000 + i] * size),
            work_ms=work,
        )
        assert forward(None, batch) is result
    meter.poll()
    assert not meter.written
    ready[0] = True
    meter.poll()
    data = json.loads(meter.stem.with_suffix(".json").read_text())
    assert data["window_ms"] == 12 and data["discarded_partial_steps"] == 1
    assert [r["step_ms"] for r in data["records"]] == [4, 6]
    assert [r["prefix_lens"][0] for r in data["records"]] == [32005, 32006]
    assert all(r["draft_cuda_graph"] for r in data["records"])
    assert [r["batch_size"] for r in data["records"]] == [7, 6]
    assert [len(r["accepted_tokens"]) for r in data["records"]] == [7, 6]


def test_trace_allows_under_cap_batches_excludes_warmup_and_pauses_prefill():
    trace = object.__new__(lagom_perf.StepTrace)
    trace.required_batch_size = None
    trace.max_batch_size, trace.request_prefix = 8, "r"
    trace.seen, trace.skip, trace.steps = 0, 1, 2
    trace.poll, trace.slot, trace.nsys, trace.records = Mock(), Mock(), False, []
    for i, size in enumerate([1, 1, 9, 7, 6]):
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=f"r{j}" if i else "warmup") for j in range(size)],
            seq_lens_cpu=SimpleNamespace(tolist=lambda size=size: [32000] * size),
            forward_iter=i,
        )
        trace.begin(batch)
        if i == 3:
            trace.pause()
            assert trace.active == -1
            trace.slot.fill_.assert_called_with(-1)
    assert trace.seen == 3
    assert [r["forward_iter"] for r in trace.records] == [3, 4]
    assert [r["batch_size"] for r in trace.records] == [7, 6]


@pytest.mark.parametrize("mode", ["lagom", "hisparse", "dspark"])
def test_trace_report_validation_accepts_variable_batches_below_cap(
    bench, tmp_path, mode
):
    records = reports()[0]["records"]
    for record in records:
        record["request_ids"] = [
            bench.REQUEST_PREFIX + rid for rid in record["request_ids"]
        ]
    names = [
        "step",
        "draft",
        "draft.model",
        "target_verify",
        "target.attention_core",
        "target.attention_block",
        "target.mlp",
    ]
    if mode == "lagom":
        names += [
            "target.kv_plan",
            "target.kv_refill",
            "target.kv_prefetch",
            "target.prefetch_wait",
        ]
    elif mode == "hisparse":
        names += ["target.kv_refill_fused"]
    data = {
        "records": records,
        "regions": [
            {"step": step, "name": name, "start_ns": 1, "end_ns": 2}
            for step in range(2)
            for name in names
        ],
    }
    path = tmp_path / "rank-test.json"
    path.write_text(json.dumps(data))
    bench.common.validate_traces(
        tmp_path,
        1,
        2,
        mode,
        max_batch_size=8,
        request_prefix=bench.REQUEST_PREFIX,
    )
    records[0]["batch_size"] = 9
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="outside configured cap"):
        bench.common.validate_traces(tmp_path, 1, 2, mode, max_batch_size=8)
    records[0]["batch_size"] = 7
    records[0]["request_ids"][0] = "lagom-warmup"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="outside benchmark cohort"):
        bench.common.validate_traces(
            tmp_path,
            1,
            2,
            mode,
            max_batch_size=8,
            request_prefix=bench.REQUEST_PREFIX,
        )


def test_decode_event_meter_with_cuda_graph(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA event timing and graph replay")
    meter = lagom_decode_perf.DecodeMeter(
        {
            "output_dir": str(tmp_path),
            "max_batch_size": 8,
            "steps": 2,
            "warmup": 0,
            "verify_width": 8,
            "request_prefix": "r",
        }
    )
    x = torch.ones(256, 256, device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        x.mul_(1.01)
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid=f"r{i}",
                sampling_params=SimpleNamespace(max_new_tokens=1024),
                output_ids=[],
            )
            for i in range(8)
        ],
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        is_extend_in_batch=False,
        seq_lens_cpu=None,
        seq_lens=torch.full((8,), 32000, device="cuda"),
        forward_iter=0,
    )
    for i in range(2):
        batch.forward_iter = i
        meter.begin(batch)
        meter.records[-1]["draft_cuda_graph"] = True
        graph.replay()
        batch.seq_lens.add_(1)
        meter.finish(
            SimpleNamespace(
                can_run_cuda_graph=True,
                accept_lens=torch.ones(8, dtype=torch.int32, device="cuda"),
            )
        )
    torch.cuda.synchronize()
    meter.poll()
    data = json.loads(meter.stem.with_suffix(".json").read_text())
    assert [r["prefix_lens"][0] for r in data["records"]] == [32000, 32001]
    assert data["window_ms"] > 0 and all(r["step_ms"] > 0 for r in data["records"])
