"""Workload identity, fair baseline commands, trace accounting and graph markers."""

import importlib.util
import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.observability import lagom_perf
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/lagom" / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


run = module("run")
nsys = module("analyze_nsys")


def args(*extra):
    result = run.parser().parse_args(
        [
            "--model",
            "target",
            "--draft",
            "draft",
            "--sharegpt",
            "sharegpt.json",
            "--output",
            "result",
            *extra,
        ]
    )
    run.validate(result)
    return result


def test_three_modes_change_only_prefetch_and_offload():
    settings = args()
    commands = {mode: run.server_command(settings, mode, 30000) for mode in run.MODES}
    for command in commands.values():
        assert command[command.index("--speculative-num-draft-tokens") + 1] == "8"
        assert command[command.index("--speculative-dspark-block-size") + 1] == "7"
        assert command[command.index("--max-running-requests") + 1] == "1"
        assert (
            command[command.index("--cuda-graph-config") + 1] == '{"decode":{"bs":[1]}}'
        )
    configurations = {
        m: json.loads(c[c.index("--hisparse-config") + 1])
        for m, c in commands.items()
        if m != "dspark"
    }
    assert configurations["lagom"]["prefetcher"] == "lagom"
    assert configurations["lagom"]["prefetcher_config"]["prediction_steps"] == 7
    assert "prefetcher" not in configurations["hisparse"]
    assert all(
        c["force_h2d_each_step"] is False and c["device_buffer_size"] == 8192
        for c in configurations.values()
    )
    assert "--enable-hisparse" not in commands["dspark"]
    assert "--disable-cuda-graph" not in commands["lagom"]


def test_capacity_and_trace_length_follow_physical_window():
    with pytest.raises(ValueError, match="W\\+P\\+1"):
        args("--hot", "4096")
    value = args("--w", "7", "--p", "7", "--hot", "8192", "--trace-output", "16")
    assert value.trace_output > (value.trace_skip + value.trace_steps) * (value.w + 1)
    assert value.input_len == 32000 and value.output_len == 1024


@pytest.mark.parametrize("draft_backend", ["marlin", "auto"])
def test_moe_backends_are_explicit_and_identical_across_modes(draft_backend, tmp_path):
    settings = args(
        "--moe-runner-backend",
        "marlin",
        "--speculative-moe-runner-backend",
        draft_backend,
    )
    for mode in run.MODES:
        command = run.server_command(settings, mode, 30000)
        assert command[command.index("--moe-runner-backend") + 1] == "marlin"
        assert (
            command[command.index("--speculative-moe-runner-backend") + 1]
            == draft_backend
        )
        assert "--disable-cuda-graph" not in command
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(["--moe-runner-backend=triton"]))
    settings.server_args_json = str(extra)
    with pytest.raises(ValueError, match="cannot override"):
        run.server_command(settings, "lagom", 30000)


def test_sharegpt_packing_exact_lengths_stable_hashes_no_random_padding():
    dataset = [
        {"conversations": [{"from": "human", "value": str(i) * 20}]} for i in range(12)
    ]
    encode = lambda text: list(text.encode())
    first = run.pack_sharegpt(dataset, encode, count=3, length=53, seed=42)
    second = run.pack_sharegpt(dataset, encode, count=3, length=53, seed=42)
    assert first == second
    assert all(len(row["input_ids"]) == 53 for row in first)
    assert len({r["token_sha256"] for r in first}) == 3
    with pytest.raises(ValueError, match="insufficient"):
        run.pack_sharegpt([], encode, count=1, length=53, seed=42)


def test_streaming_counts_speculative_first_chunk_correctly(monkeypatch):
    class Response:
        def __enter__(self):
            return iter(
                [
                    b'data: {"text":"ab","meta_info":{"prompt_tokens":3,"completion_tokens":2}}\n',
                    b'data: {"text":"abcde","meta_info":{"prompt_tokens":3,"completion_tokens":5}}\n',
                    b"data: [DONE]\n",
                ]
            )

        def __exit__(self, *_):
            pass

    def opening(request, **kwargs):
        body = json.loads(request.data)
        assert body["input_ids"] == [10, 11, 12]
        assert body["sampling_params"]["ignore_eos"] is True
        assert body["sampling_params"]["temperature"] == 0
        return Response()

    monkeypatch.setattr(run.urllib.request, "urlopen", opening)
    result = run.generate("http://localhost", [10, 11, 12], 5, 10)
    assert result["output_tokens"] == 5 and result["decode_tokens"] == 3
    with pytest.raises(RuntimeError, match="wrong token counts"):
        run.generate("http://localhost", [10, 11, 12], 6, 10)


def test_summary_keeps_inclusive_spans_and_unknown_cpu_wait_separate():
    regions = [
        {"step": 0, "name": "step", "start_ns": 100, "end_ns": 1100},
        {"step": 0, "name": "target.attention_core", "start_ns": 200, "end_ns": 600},
        {"step": 0, "name": "target.kv_prefetch", "start_ns": 300, "end_ns": 800},
        {"step": 0, "name": "target.kv_prefetch", "start_ns": 700, "end_ns": 1000},
        {"step": 1, "name": "step", "start_ns": 1300, "end_ns": 2000},
    ]
    summary = lagom_perf.summarize(regions, 2)
    assert summary[0]["step_ms"] == 0.001
    assert summary[0]["target.kv_prefetch_ms"] == 0.0007
    assert summary[0]["gpu_wait_cpu_ms"] is None
    assert summary[1]["inter_step_gap_ms"] == 0.0002


def test_disabled_trace_does_not_access_cuda_or_change_result(monkeypatch):
    monkeypatch.delenv("SGLANG_LAGOM_PERF_TRACE", raising=False)
    lagom_perf.get_trace.cache_clear()

    @lagom_perf.verify_step
    def forward(self, batch):
        return 17

    assert forward(object(), object()) == 17
    with lagom_perf.phase("kv_refill"):
        pass
    assert lagom_perf.prefetch_slot(1) is None


@pytest.mark.parametrize("cpu_lengths", [False, True])
def test_verify_step_snapshots_optional_cpu_lengths_without_host_wait(
    monkeypatch, tmp_path, cpu_lengths
):
    import torch

    ready = False

    class DeviceValues:
        def __init__(self, values):
            self.values = list(values)
            self.clones = 0

        def detach(self):
            return self

        def clone(self):
            self.clones += 1
            return DeviceValues(self.values)

        def cpu(self):
            assert ready, "trace must not read device values before completion"
            return self

        def tolist(self):
            assert ready, "trace must not implicitly synchronize for metadata"
            return list(self.values)

    trace = object.__new__(lagom_perf.StepTrace)
    trace.written, trace.nsys, trace.nsys_running = False, False, False
    trace.seen, trace.skip, trace.steps, trace.active = 0, 1, 2, -1
    trace.output, trace.device = tmp_path, 0
    trace.records, trace.cpu, trace.events = [], [], []
    trace.accepted, trace.prefix_lens_gpu = {}, {}
    trace.slot = torch.tensor([-1])
    trace.stamps = torch.tensor([[[100, 200], [300, 400]]])
    trace.names = {("step", None): 0}
    event = Mock(query=lambda: ready)
    trace.torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda: event))
    # Exercise the real decorator/begin/finish/export; only CUDA timers are mocked.
    trace.region = lambda *_: nullcontext()
    monkeypatch.setattr(lagom_perf, "get_trace", lambda: trace)
    batch = SimpleNamespace(
        seq_lens=DeviceValues([32000]),
        seq_lens_cpu=torch.tensor([32000]) if cpu_lengths else None,
        reqs=[SimpleNamespace(rid="baseline")],
        forward_iter=0,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    result = SimpleNamespace(can_run_cuda_graph=True, accept_lens=DeviceValues([1]))

    @lagom_perf.verify_step
    def decode(self, batch):
        lagom_perf.note_draft_graph(True)
        batch.seq_lens.values[0] += 1
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu.add_(1)
        return result

    # One skipped step, two recorded steps, then an unrecorded step while the
    # completion event is still pending. Batch storage is reused every time.
    for iteration in range(4):
        batch.forward_iter = iteration
        assert decode(None, batch) is result
    assert not trace.written and not list(tmp_path.iterdir())
    assert len(trace.records) == 2 and trace.active == -1
    assert batch.seq_lens.clones == (0 if cpu_lengths else 2)
    assert (batch.seq_lens_cpu is None) == (not cpu_lengths)
    ready = True
    trace.poll()
    report = next(
        p for p in tmp_path.glob("rank-*.json") if not p.name.endswith(".trace.json")
    )
    records = json.loads(report.read_text())["records"]
    assert [r["prefix_lens"] for r in records] == [[32001], [32002]]
    assert [r["forward_iter"] for r in records] == [1, 2]
    assert all(r["cuda_graph"] and r["draft_cuda_graph"] for r in records)
    assert all(r["accepted_tokens"] == [1] for r in records)


def test_nsys_idle_is_conservative_union_across_devices(tmp_path):
    path = tmp_path / "nsys.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER)"
        )
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES(0,10000000,'lagom/cpu_dependency_commit/step=2',99)"
        )
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER)"
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)",
            [(0, 3000000, 0), (2000000, 5000000, 1)],
        )
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(7000000,8000000)")
    rows = nsys.analyze(path)["rows"]
    assert rows == [
        {
            "step": 2,
            "global_tid": 99,
            "dependency_wall_ms": 10,
            "all_gpu_idle_in_dependency_ms": 4,
        }
    ]


def test_export_waits_for_transfer_completion_and_writes_valid_tracks(tmp_path):
    import torch

    trace = object.__new__(lagom_perf.StepTrace)
    trace.written, trace.nsys_running = False, False
    trace.events = [Mock(query=Mock(side_effect=[False, True]))]
    trace.stamps = torch.tensor([[[100, 200]], [[120, 220]]])
    trace.names = {("step", None): 0, ("target.kv_prefetch", 2): 1}
    trace.records = [
        {
            "step": 0,
            "cuda_graph": True,
            "draft_cuda_graph": True,
            "prefix_lens": [32000],
        }
    ]
    trace.accepted = {0: torch.tensor([2])}
    trace.output, trace.device, trace.steps = tmp_path, 0, 1
    trace.cpu = [
        {"step": 0, "name": "step", "layer": None, "start_ns": 1000, "end_ns": 1500}
    ]
    trace.poll()
    assert not trace.written and not list(tmp_path.iterdir())
    trace.poll()
    assert trace.written
    chrome = json.loads(next(tmp_path.glob("*.trace.json")).read_text())
    assert all(
        isinstance(e["pid"], int) and isinstance(e["tid"], int)
        for e in chrome["traceEvents"]
    )
    spans = [e for e in chrome["traceEvents"] if e["ph"] == "X"]
    assert len(spans) == 3
    assert {e["cat"] for e in spans} == {"GPU", "CPU (separate clock)"}
    assert next(tmp_path.glob("*.steps.csv")).read_text().count("\n") == 2


def test_timestamp_ring_survives_independent_graph_replays(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Graph and globaltimer")
    trace = lagom_perf.StepTrace({"output_dir": str(tmp_path), "steps": 2, "skip": 0})
    side = torch.cuda.Stream()
    trace.streams = [side]
    batch = SimpleNamespace(
        seq_lens=torch.tensor([32000], device="cuda"),
        seq_lens_cpu=None,
        reqs=[SimpleNamespace(rid="baseline")],
        forward_iter=0,
    )
    result = SimpleNamespace(
        can_run_cuda_graph=True,
        accept_lens=torch.ones(1, dtype=torch.int32, device="cuda"),
    )
    side_slot = torch.full_like(trace.slot, -1)
    ready = torch.cuda.Event(external=True)
    done = torch.cuda.Event(external=True)
    ready.record()
    done.record()

    def target():
        done.wait()
        with trace.region("step"):
            side_slot.copy_(trace.slot)
            ready.record()

    def prefetch():
        ready.wait()
        with trace.region("kv_prefetch", layer=0, slot=side_slot):
            torch.empty(8, device="cuda").fill_(2)
        done.record()

    target()
    with torch.cuda.stream(side):
        prefetch()
    torch.cuda.synchronize()
    main_graph, side_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(main_graph):
        target()
    with torch.cuda.graph(side_graph, stream=side):
        prefetch()
    for step in (0, 1):
        batch.forward_iter = step
        trace.begin(batch)
        main_graph.replay()
        with torch.cuda.stream(side):
            side_graph.replay()
        batch.seq_lens.add_(1)
        trace.finish(result)
    torch.cuda.synchronize()
    values = trace.stamps.cpu()
    for index in trace.names.values():
        assert (values[index, :, 0] > 0).all()
        assert (values[index, :, 1] >= values[index, :, 0]).all()
        assert values[index, 1, 0] >= values[index, 0, 1]
    trace.poll()
    report = next(
        p for p in tmp_path.glob("rank-*.json") if not p.name.endswith(".trace.json")
    )
    assert [r["prefix_lens"] for r in json.loads(report.read_text())["records"]] == [
        [32000],
        [32001],
    ]
