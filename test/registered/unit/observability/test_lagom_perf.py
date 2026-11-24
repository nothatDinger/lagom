"""Workload identity, fair baseline commands, trace accounting and graph markers."""

import csv
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
    assert configurations["lagom"]["prefetcher_config"]["prediction_steps"] == settings.p == 4
    assert configurations["lagom"]["prefetcher_config"]["buffer_slots"] == settings.prefetch_slots == 21
    assert "prefetcher" not in configurations["hisparse"]
    assert all(
        c["force_h2d_each_step"] is False and c["device_buffer_size"] == 8192
        for c in configurations.values()
    )
    assert "--enable-hisparse" not in commands["dspark"]
    assert "--disable-cuda-graph" not in commands["lagom"]


def test_capacity_and_trace_length_follow_physical_window():
    with pytest.raises(ValueError, match="hot must hold K"):
        args("--hot", "256")
    assert args("--hot", "512").hot == 512
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
        {
            "step": 0, "name": "target.kv_prefetch", "start_ns": 300,
            "end_ns": 800, "kv_entries": 7,
        },
        {
            "step": 0, "name": "target.kv_prefetch", "start_ns": 700,
            "end_ns": 1000, "kv_entries": 3,
        },
        {"step": 1, "name": "step", "start_ns": 1300, "end_ns": 2000},
    ]
    summary = lagom_perf.summarize(regions, 2)
    assert summary[0]["step_ms"] == 0.001
    assert summary[0]["target.kv_prefetch_ms"] == 0.0007
    assert summary[0]["target.kv_prefetch_entries"] == 10
    assert summary[1]["target.kv_prefetch_entries"] is None
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
    trace.written, trace.nsys = False, False
    trace.seen, trace.skip, trace.steps, trace.active = 0, 1, 2, -1
    trace.required_batch_size = None
    trace.max_batch_size, trace.request_prefix = None, None
    trace.output, trace.device = tmp_path, 0
    trace.records, trace.cpu, trace.events = [], [], []
    trace.accepted, trace.prefix_lens_gpu = {}, {}
    trace.slot = torch.tensor([-1])
    trace.stamps = torch.tensor([[[100, 200], [300, 400]]])
    trace.names = {("step", None): 0}
    trace.kv_entry_regions = set()
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
    trace.written = False
    trace.events = [Mock(query=Mock(side_effect=[False, True]))]
    trace.stamps = torch.tensor([[[100, 200]], [[120, 220]]])
    trace.names = {("step", None): 0, ("target.kv_prefetch", 2): 1}
    trace.kv_entry_regions = {1}
    trace.kv_entries = Mock()
    trace.kv_entries.cpu.return_value = torch.tensor([[0], [0]])
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
    trace.kv_entries.cpu.assert_not_called()
    trace.poll()
    trace.kv_entries.cpu.assert_called_once()
    assert trace.written
    chrome = json.loads(next(tmp_path.glob("*.trace.json")).read_text())
    assert all(
        isinstance(e["pid"], int) and isinstance(e["tid"], int)
        for e in chrome["traceEvents"]
    )
    spans = [e for e in chrome["traceEvents"] if e["ph"] == "X"]
    assert len(spans) == 3
    assert {e["cat"] for e in spans} == {"GPU", "CPU (separate clock)"}
    event = next(e for e in spans if e["name"] == "target.kv_prefetch")
    assert event["args"] == {"step": 0, "kv_entries": 0}
    csv_path = next(tmp_path.glob("*.steps.csv"))
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["target.kv_prefetch_entries"] == "0"


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

    # Graph batch 2 contains one live request. Its plan includes a host-only
    # prediction, an invalid source, and a stale tail past the actual count.
    plan = SimpleNamespace(
        kinds=torch.tensor(
            [[1, 2, 3, 1, 0, 1], [1, 1, 1, 1, 1, 1]],
            dtype=torch.int32,
            device="cuda",
        ),
        demand=torch.tensor(
            [[1, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]],
            dtype=torch.int32,
            device="cuda",
        ),
        counts=torch.tensor([5, 6], dtype=torch.int32, device="cuda"),
    )
    live = torch.tensor([1], dtype=torch.int32, device="cuda")
    reqs = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
    forecast_counts = torch.tensor([19, 0, 7], dtype=torch.int32, device="cuda")

    def target():
        done.wait()
        with trace.region("step"):
            with trace.region("kv_stage_copy", layer=0):
                pass
            with trace.region("kv_refill", layer=0):
                pass
            trace.record_verify_copies(0, plan, live, 2)
            forecast_counts[2:3].copy_(trace.slot + 7)
            side_slot.copy_(trace.slot)
            ready.record()

    def prefetch():
        ready.wait()
        with trace.region("kv_prefetch", layer=0, slot=side_slot):
            torch.empty(8, device="cuda").fill_(2)
        trace.record_forecast_copies(0, side_slot, forecast_counts, reqs, live)
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
        plan.demand[0, 3] = step
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
    data = json.loads(report.read_text())
    assert [r["prefix_lens"] for r in data["records"]] == [[32000], [32001]]
    assert [s["target.kv_refill_entries"] for s in data["summary"]] == [1, 2]
    assert [s["target.kv_stage_copy_entries"] for s in data["summary"]] == [2, 2]
    assert [s["target.kv_prefetch_entries"] for s in data["summary"]] == [7, 8]


def test_compare_matched_fused_steps_counts_payload_and_committed_tokens(tmp_path):
    analysis = module("h2d_trace")
    gather, reread, commit = analysis.PHASES
    reports = []
    for force in (False, True):
        folder = tmp_path / str(force)
        (folder / "steps").mkdir(parents=True)
        cell = dict(
            input_len=32768,
            trace_batch_size=2,
            trace_steps=2,
            force_h2d_each_step=force,
        )
        for rank in range(4):
            records = [
                dict(
                    step=i,
                    batch_size=2,
                    cuda_graph=True,
                    draft_cuda_graph=True,
                    accepted_tokens=[2, 3],
                    prefix_lens=[32768, 32770],
                )
                for i in range(2)
            ]
            rows = [
                {
                    "step": i,
                    "step_ms": 24 if force else 20,
                    gather + "_ms": 2,
                    reread + "_ms": 5 if force else 1,
                    commit + "_ms": 3,
                    gather + "_entries": 4,
                    reread + "_entries": 7 if force else 0,
                    commit + "_entries": 2,
                    reread + "_resident_hits": 7,
                    commit + "_admissions": 6,
                }
                for i in range(2)
            ]
            (folder / "steps" / f"rank-{rank}.json").write_text(
                json.dumps(dict(rank=rank, records=records, summary=rows))
            )
        reports.append(analysis.read_cell(folder, cell))
    assert reports[0]["ranks"]["0"]["decode_tpot_ms"] == 8
    result = analysis.compare_cells(*reports)
    assert result["passed"]
    assert result["ranks"]["0"]["delta"]["h2d_payload_bytes_per_step"] == 7 * 584
    reports[1]["ranks"]["1"].update(step_ms=19, admissions=1)
    assert not analysis.compare_cells(*reports)["passed"]
    for rank in range(4):
        path = folder / "steps" / f"rank-{rank}.json"
        trace = json.loads(path.read_text())
        path.write_text(json.dumps(trace | {"nsys": True, "pid": rank + 100}))
    profiled = analysis.read_cell(folder, cell | {"nsys": True})
    assert profiled["ranks"]["3"]["pid"] == 103
    assert profiled["ranks"]["0"]["h2d_payload_bytes_per_step"] == 13 * 584
    assert "decode_tpot_ms" not in profiled["ranks"]["0"]
    with pytest.raises(ValueError, match="profiler-free"):
        analysis.compare_cells(reports[0], profiled)
    with pytest.raises(ValueError, match="Profiler setting"):
        analysis.read_cell(folder, cell)
    (folder / "trace_config.json").write_text(json.dumps({"nsys": True}))
    current = folder / "steps/rank-3.json"
    previous = current.rename(folder / "steps/rank-device7-pid103.json")
    data = json.loads(previous.read_text())
    previous.write_text(
        json.dumps({k: v for k, v in data.items() if k not in ("nsys", "pid")})
    )
    assert analysis.read_cell(folder, cell | {"nsys": True}) == profiled
    previous.unlink()
    with pytest.raises(ValueError, match="four complete"):
        analysis.read_cell(folder, cell | {"nsys": True})


def test_nsys_step_uses_gpu_edges_and_keeps_api_and_overlaps_separate(tmp_path):
    path = tmp_path / "report.sqlite"
    pid, tid = 17 << 24, (17 << 24) + 5
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE NVTX_EVENTS(start,end,globalTid,text)")
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)",
            [
                (0, 30, tid, "lagom/step/step=0/layer=None"),
                (1, 5, tid, "lagom/step_begin/step=0"),
                (25, 29, tid, "lagom/step_end/step=0"),
            ],
        )
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId)"
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
            [(2, 4, tid, 1), (7, 12, tid, 2), (26, 28, tid, 3)],
        )
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,correlationId,deviceId)"
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,4)",
            [
                (98, 100, pid, 1),
                (110, 150, pid, 2),
                (200, 202, pid, 3),
                (100, 200, pid + 1, 2),
            ],
        )
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,globalPid,deviceId,copyKind,bytes)"
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(130,160,?,4,1,584)", (pid,)
        )
    (row,) = nsys.step_breakdown(path)
    for name, ns in dict(
        gpu_step_ms=100,
        kernel_only_ms=20,
        explicit_h2d_only_ms=10,
        gpu_overlap_ms=20,
        gpu_idle_ms=50,
        cpu_cuda_api_union_ms=9,
    ).items():
        assert row[name] == pytest.approx(ns / 1e6)
    assert row["explicit_h2d_bytes"] == 584


@pytest.mark.parametrize("rank", [0, 1])
def test_profile_window_brackets_all_tp_stamps_and_final_streams(rank):
    trace = object.__new__(lagom_perf.StepTrace)
    trace.written, trace.nsys, trace.rank = False, True, rank
    trace.events, trace.records, trace.accepted = [], [], {}
    trace.profile_group, trace.torch, trace.slot = Mock(), Mock(), Mock()
    trace.seen, trace.skip, trace.steps = 0, 1, 2
    trace.max_batch_size = trace.required_batch_size = trace.request_prefix = None
    trace.streams = [Mock()]
    calls = []
    trace.profile_group.barrier.side_effect = lambda: calls.append("barrier")
    cuda = trace.torch.cuda
    cuda.cudart().cudaProfilerStart.side_effect = lambda: calls.append("start")
    cuda.cudart().cudaProfilerStop.side_effect = lambda: calls.append("stop")
    cuda.Event.return_value.synchronize.side_effect = lambda: calls.append(
        "stream_complete"
    )
    batch = SimpleNamespace(
        seq_lens_cpu=Mock(), forward_iter=1, reqs=[SimpleNamespace(rid="a")]
    )
    result = SimpleNamespace(can_run_cuda_graph=True, accept_lens=Mock())
    for _ in range(3):
        trace.begin(batch)
        if trace.active >= 0:
            calls.append("step")
        trace.finish(result)
    expected = [
        "barrier",
        "step",
        "step",
        "stream_complete",
        "stream_complete",
        "barrier",
    ]
    assert calls == (["start", *expected, "stop"] if rank == 0 else expected)
