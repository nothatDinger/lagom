"""Offline probe isolation, real miss accounting and next-window matching."""

import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
from sglang.srt.speculative.oasiskv_trace import (
    ForwardObservation,
    OasisKVTrace,
    TensorSnapshot,
    check_live_locations,
    replace_method,
    resident_lookup,
    validate_trace_hisparse,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"oasiskv_trace_{name}", ROOT / "benchmark/oasiskv/trace" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = load_script("analyze")
runner = load_script("run")
auditor = load_script("audit")


def records():
    common = {
        "schema_version": 1,
        "run_id": "test",
        "request_id": "r",
        "generation": 1,
        "window_id": 0,
        "layer": 2,
        "head": 0,
        "snapshot_id": "snapshot-0",
        "verify_candidates": 1,
        "unit": "c4",
        "unit_tokens": 4,
        "bytes_per_unit": 584,
        "target_forward_ms": None,
    }
    source = dict(
        common,
        kind="window",
        prefix_len=40,
        prompt_len=40,
        accepted_drafts=0,
        commit_len=1,
        history_units=10,
        resident_after=[1, 2],
        input_ids=[20, 21],
        topk=[[1, 2], [1, 2]],
        miss_events=[{"row": 0, "ids": []}, {"row": 1, "ids": []}],
    )
    following = dict(
        source,
        window_id=1,
        snapshot_id="snapshot-1",
        prefix_len=41,
        input_ids=[22, 23],
        topk=[[1, 3, 10], [3, 4]],
        miss_events=[{"row": 0, "ids": [3, 10]}, {"row": 1, "ids": [3, 4]}],
    )
    probe = dict(
        common,
        kind="probe",
        p=1,
        mode="resident",
        input_ids=[20, 21, 25],
        topk=[[1, 3, 3, -1, 4, 10, 11]],
        probe_demand_h2d_bytes=0,
    )
    return source, following, probe


def test_next_window_uses_actual_commit_and_deduplicates_resident_and_future_kv():
    source, following, probe = records()
    report = analyzer.analyze([source, following, probe], 10)
    sample = report["samples"][0]
    assert sample["prediction_units"] == 3  # 1,3,4; no tentative 10/11
    assert sample["prefetch_candidate_units"] == 2  # subtract resident 1
    assert sample["next_miss_events"] == 4
    assert sample["next_miss_unique_units"] == 3
    assert sample["next_miss_bytes"] == 4 * 584
    assert sample["potential_first_fetch_bytes"] == 2 * 584  # repeated 3 counted once
    assert sample["potential_unique_miss_coverage"] == pytest.approx(2 / 3)
    assert sample["timely_miss_coverage"] is None
    assert sample["saved_critical_path_ms"] is None
    assert sample["delta_target_ms"] is None


def test_no_implicit_prediction_from_p0():
    source, following, _ = records()
    assert analyzer.analyze([source, following], 256)["matched_records"] == 0


def test_admission_cap_and_unsorted_topk_frequency():
    source, following, probe = records()
    probe.update(p=2, input_ids=[20, 21, 25, 26], topk=[[3, 4], [4, 4]])
    sample = analyzer.analyze([source, following, probe], 1)["samples"][0]
    assert sample["prefetch_candidate_units"] == 1
    assert sample["potential_first_fetch_bytes"] == 584
    zero = analyzer.analyze([source, following, probe], 0)["samples"][0]
    assert zero["prefetch_candidate_units"] == zero["potential_miss_units"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefix_len", 42),
        ("prompt_len", 39),
        ("bytes_per_unit", 1),
    ],
)
def test_rejects_mismatched_next_window(field, value):
    source, following, probe = records()
    following[field] = value
    with pytest.raises(ValueError):
        analyzer.analyze([source, following, probe], 256)


@pytest.mark.parametrize(
    "change",
    [
        {"snapshot_id": "other"},
        {"input_ids": [20, 99, 25]},
        {"probe_demand_h2d_bytes": 584},
        {"p": 0},
    ],
)
def test_rejects_invalid_probe(change):
    source, following, probe = records()
    probe.update(change)
    with pytest.raises(ValueError):
        analyzer.analyze([source, following, probe], 256)


def test_generation_reuse_missing_steps_and_duplicate_windows():
    source, following, probe = records()
    following["generation"] = 2
    report = analyzer.analyze([source, following, probe], 256)
    assert report["matched_records"] == 0
    assert report["skipped"]["no_immediate_next_window"] == 1
    following.update(generation=1, window_id=2)
    assert analyzer.analyze([source, following, probe], 256)["matched_records"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        analyzer.analyze([source, source], 256)


def test_unknown_misses_are_not_filled_in_from_topk():
    source, following, probe = records()
    following["miss_events"] = None
    sample = analyzer.analyze([source, following, probe], 256)["samples"][0]
    assert sample["next_miss_bytes"] is None
    assert sample["potential_unique_miss_coverage"] is None
    assert sample["history_coverage"] == 1


def test_snapshot_restores_aliases_in_place_and_enforces_host_budget():
    data = torch.arange(16).reshape(4, 4)
    view = data[:, :2]
    pointer = data.data_ptr()
    snap = TensorSnapshot([("whole", data), ("duplicate", data), ("view", view)], 1024)
    data.fill_(42)
    view.fill_(-5)
    snap.restore()
    assert torch.equal(data, torch.arange(16).reshape(4, 4))
    assert data.data_ptr() == pointer
    assert len(snap.entries) == 2
    with pytest.raises(RuntimeError, match="snapshot needs"):
        TensorSnapshot([("large", data)], 1)


def test_replace_method_restores_binding_after_exception():
    class Example:
        def read(self):
            return 3

    obj = Example()
    with pytest.raises(RuntimeError), replace_method(obj, "read", lambda: 4):
        assert obj.read() == 4
        raise RuntimeError("shadow failed")
    assert obj.read() == 3
    assert "read" not in vars(obj)


def fake_coordinator():
    """Use the production step-major dispatcher with a tiny CPU IO boundary."""
    c = NS(
        device="cpu",
        top_k=2,
        item_size_bytes=584,
        req_device_buffer_tokens=torch.tensor([[[0, 1, -1, -1]]], dtype=torch.int32),
        req_device_buffer_token_locs=torch.tensor(
            [[[10, 11, 12, 13]]], dtype=torch.int32
        ),
        req_to_host_pool=torch.arange(100, 120).view(1, -1),
        calls=[],
    )

    def kernel(
        self, reqs, lengths, topk, layer, *, output_buffer, record_plan, miss_plan
    ):
        src, _, count = miss_plan
        count.zero_()
        self.calls.append(topk.tolist())
        for col, token in enumerate(topk[0].tolist()):
            if token < 0 or token >= 10:
                output_buffer[0, col] = -1
                continue
            tokens = self.req_device_buffer_tokens[0, 0]
            matches = (tokens == token).nonzero()
            if matches.numel():
                slot = int(matches[0, 0])
            else:
                empty = (tokens == -1).nonzero()
                slot = int(empty[0, 0]) if empty.numel() else 0
                tokens[slot] = token
                src[0, int(count[0])] = self.req_to_host_pool[0, token]
                count[0] += 1
            output_buffer[0, col] = self.req_device_buffer_token_locs[0, 0, slot]
        return output_buffer

    c._run_swap_in_kernel = MethodType(kernel, c)
    c.swap_in_selected_pages_spec = MethodType(
        HiSparseCoordinator.swap_in_selected_pages_spec, c
    )
    return c


@pytest.mark.parametrize("padding", [0, 2])
def test_resident_probe_never_calls_refill_and_preserves_raw_prediction(padding):
    c = fake_coordinator()
    raw = torch.tensor(
        [[0, 2], [2, 3], [1, 4]] + [[-1, -1]] * padding, dtype=torch.int32
    )
    obs = ForwardObservation(
        c, real_width=2, mode="resident", history_units=10, req_slot=0
    )
    with obs.capture():
        result = c.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([0]),
            compressed_seq_lens=torch.full((3 + padding,), 10),
            top_k_result=raw,
            layer_id=0,
            verify_lens_cpu=[3],
            output_buffer=None,
        )
    assert len(c.calls) == 2  # No refill launch for the third (probe) row.
    assert result[2].tolist() == [11, -1]
    assert obs.layers[0]["topk"][2] == [1, 4]  # Missing 4 remains a prediction.
    assert obs.layers[0]["miss_events"] == [
        {"row": 0, "ids": [2]},
        {"row": 1, "ids": [3]},
        {"row": 2, "ids": []},
    ]
    if padding:
        assert (result[3:] == -1).all()


def test_exact_probe_refills_and_detects_eviction_of_still_live_rows():
    c = fake_coordinator()
    obs = ForwardObservation(
        c, real_width=2, mode="exact", history_units=10, req_slot=0
    )
    with pytest.raises(RuntimeError, match="evicted C4"), obs.capture():
        c.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([0]),
            compressed_seq_lens=torch.full((3,), 10),
            top_k_result=torch.tensor([[0, 2], [2, 3], [1, 4]], dtype=torch.int32),
            layer_id=0,
            verify_lens_cpu=[3],
            output_buffer=None,
        )
    assert len(c.calls) == 3
    # Both nested method replacements must have been unwound.
    assert c._run_swap_in_kernel.__self__ is c
    assert c.swap_in_selected_pages_spec.__self__ is c


def test_missing_probes_and_current_scratch_are_not_mistaken_for_stale_history():
    assert resident_lookup([[2, -1, 9]], [2, 3], [10, 11]) == [[10, -1, -1]]
    check_live_locations([[2, 10, 4]], [[10, -1, -1]], [2], [10], 10)


def test_trace_disabled_has_no_runtime_dependencies(monkeypatch):
    monkeypatch.delenv("SGLANG_OASISKV_TRACE_CONFIG", raising=False)
    assert OasisKVTrace.from_environment(object()) is None


def trace_startup_runner():
    allocator = object()
    return NS(
        enable_hisparse=True,
        is_draft_worker=False,
        token_to_kv_pool_allocator=allocator,
        hisparse_coordinator=NS(
            is_dsv4_hisparse=True,
            prefetcher=None,
            prefetcher_name=None,
            enable_prefetch=False,
            skip_io=False,
            token_to_kv_pool_allocator=allocator,
        ),
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        (None, None, "hisparse_coordinator is None"),
        ("is_dsv4_hisparse", False, "is_dsv4_hisparse=False"),
        ("prefetcher", object(), "prefetcher is builtins.object"),
        ("enable_prefetch", True, "enable_prefetch=True"),
        ("skip_io", True, "skip_io=True"),
    ],
)
def test_startup_guard_identifies_each_actual_failure(field, value, reason):
    model_runner = trace_startup_runner()
    if field is None:
        model_runner.hisparse_coordinator = None
    else:
        setattr(model_runner.hisparse_coordinator, field, value)
    with pytest.raises(ValueError) as error:
        validate_trace_hisparse(model_runner)
    message = str(error.value)
    assert reason in message
    details = json.loads(message.split("; runtime: ", 1)[1])
    assert details["runner_enable_hisparse"] is True
    assert details["runner_allocator_type"] == "builtins.object"
    if field == "enable_prefetch":
        # A missing named prefetcher does not disable the legacy shared-index mode.
        assert details["prefetcher_type"] is None
        assert details["enable_prefetch"] is True


def test_startup_guard_accepts_real_io_zero_environment(monkeypatch):
    monkeypatch.setenv("SGLANG_DEBUG_HISPARSE_SKIP_IO", "0")
    model_runner = trace_startup_runner()
    model_runner.hisparse_coordinator.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
    details = validate_trace_hisparse(model_runner)
    assert details["skip_io"] is False
    assert details["is_dsv4_hisparse"] is True


@pytest.mark.parametrize("active_phase", [None, "decode", "prefill"])
def test_trace_startup_uses_resolved_graph_backends(monkeypatch, active_phase):
    from sglang.srt import runtime_context
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    from sglang.srt.speculative import oasiskv_trace

    graph = CudaGraphConfig.from_dict(
        {"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}}
    )
    if active_phase:
        getattr(graph, active_phase).backend = "full"
    # Deliberately contradict the legacy flag: resolved phases are authoritative.
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: NS(
            graph=NS(cuda_graph_config=graph, disable_cuda_graph=bool(active_phase)),
            moe=NS(enable_eplb=False),
        ),
    )
    monkeypatch.setattr(
        runtime_context, "get_schedule", lambda: NS(disable_overlap_schedule=True)
    )
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: NS(enable_dp_attention=False, pp_size=1),
    )
    monkeypatch.setattr(oasiskv_trace, "target_state_tensors", lambda runner: [])
    worker = NS(
        gamma=7,
        _oasiskv_lookahead=False,
        _simulate_acc_len=0,
        model_runner=trace_startup_runner(),
        ps=NS(tp_rank=1),
    )
    config = {"verify_candidates": 1, "probes": [1, 2], "max_windows": 2}
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        if active_phase:
            with pytest.raises(ValueError, match="ordinary DSPARK, eager"):
                OasisKVTrace(worker, config)
        else:
            trace = OasisKVTrace(worker, config)
            assert trace.coordinator is worker.model_runner.hisparse_coordinator


def test_fixed_trace_does_not_need_confidence_head_but_backend_keeps_compact_mode():
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        kwargs = {
            "draft_model": NS(),
            "gamma": 7,
            "model_runner": object(),
            "device": "cpu",
            "tp_rank": 0,
            "server_args": ServerArgs(model_path="dummy"),
            "verify_num_draft_tokens": 8,
        }
        trace_planner = DSparkVerifyPlanner(**kwargs, offline_trace=True)
        assert not trace_planner.carries_confidence
        assert read_ragged_verify_mode() is RaggedVerifyMode.COMPACT
        with pytest.raises(ValueError, match="no confidence head"):
            DSparkVerifyPlanner(**kwargs)


def test_delivered_server_command_is_accepted_by_real_cli_parser():
    args = NS(
        target=runner.TARGET,
        draft=runner.DRAFT,
        port=30000,
        top_k=512,
        resident_units=4096,
        context_length=40960,
        mem_fraction=0.9,
        seed=42,
        request_timeout=21600,
    )
    command = runner.server_command(args, 7)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(command[3:])
    assert parsed.speculative_num_draft_tokens == 8
    assert parsed.tp_size == 4
    assert parsed.disable_overlap_schedule
    assert parsed.attention_backend == "dsv4"
    assert parsed.moe_runner_backend == "flashinfer_mxfp4"
    assert json.loads(parsed.hisparse_config)["host_to_device_ratio"] == 1


def test_untraced_arm_removes_inherited_hook_but_keeps_real_io(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_OASISKV_TRACE_CONFIG", "/old/config.json")
    monkeypatch.setenv("SGLANG_DEBUG_HISPARSE_SKIP_IO", "1")
    monkeypatch.setenv("SGLANG_DSV4_FP4_DEQUANT", "1")
    args = NS(gpus="0,1,2,3", diagnostics=True)
    plain = runner.run_environment(args, None)
    assert "SGLANG_OASISKV_TRACE_CONFIG" not in plain
    assert plain["SGLANG_OASISKV_TRACE_DIAGNOSTICS"] == "1"
    assert plain["SGLANG_DEBUG_HISPARSE_SKIP_IO"] == "0"
    assert plain["SGLANG_DSV4_FP4_DEQUANT"] == "0"
    traced = runner.run_environment(args, tmp_path / "config.json")
    assert traced.pop("SGLANG_OASISKV_TRACE_CONFIG") == str(tmp_path / "config.json")
    assert traced == plain


@pytest.mark.parametrize(
    "length,ratio,valid",
    [(4096, 2, True), (16384, 2, True), (32768, 2, False), (32768, 1, True)],
)
def test_prefill_c4_capacity_is_distinct_from_decode_residency(length, ratio, valid):
    args = NS(
        verify_candidates=2,
        probes=[1, 2, 3],
        input_lengths=[length],
        context_length=40960,
        output_length=32,
        requests=1,
        snapshot_gib=32,
        max_windows=3,
        resident_units=3072,
        top_k=512,
        mem_fraction=0.9,
        prefetch_units=256,
        gpus="0,1,2,3",
        host_to_device_ratio=ratio,
    )
    if valid:
        assert runner.validate_args(args, 5) == 2
    else:
        with pytest.raises(ValueError, match="global prefill C4 pool"):
            runner.validate_args(args, 5)


def test_server_warmup_is_not_observed_or_shadowed():
    trace = OasisKVTrace.__new__(OasisKVTrace)
    trace.config = {"request_ids": ["trace-i4096-r0"]}
    trace.v, trace.probes, trace.shadows = 2, [1, 2, 3], True
    trace.runner = NS(req_to_token_pool=NS(req_generation=torch.tensor([1])))
    trace.windows = {}
    trace.layout = lambda width: width
    sentinel = object()
    calls = []

    def forward(**kwargs):
        calls.append(kwargs["layout"])
        return sentinel

    trace.worker = NS(device="cpu", _verify_executor=NS(run_compact=forward))
    batch = NS(
        reqs=[NS(rid="server-warmup")],
        has_grammar=False,
        return_logprob=False,
        req_pool_indices_cpu=torch.tensor([0]),
    )
    # No snapshot/observer/coordinator exists on this fixture: touching one
    # fails, proving startup traffic bypasses them even in the probes arm.
    assert (
        trace.run(
            batch=batch, draft_block_ids=None, draft_tokens=None, sampling_info=None
        )
        is sentinel
    )
    assert calls == [3]
    assert trace.windows == {}
    assert trace.pending is None


def test_admission_diagnostics_are_opt_in_cpu_only_and_throttled(monkeypatch, caplog):
    from sglang.srt.speculative import oasiskv_trace_diagnostics as diagnostic

    req = NS(
        rid="trace-i4096-r0",
        origin_input_ids=list(range(4096)),
        full_untruncated_fill_ids=list(range(4096)),
        prefix_indices=[],
    )
    adder = NS(
        can_run_list=[],
        rem_total_tokens=20224,
        rem_chunk_tokens=4096,
        is_hybrid_swa=True,
        rem_swa_tokens=4096,
        token_to_kv_pool_allocator=NS(size_swa=4096),
        _swa_budget_for_req=lambda *args: 4352,
        _swa_new_tokens=lambda req: 32,
    )
    result = NS(name="NO_TOKEN")
    monkeypatch.setattr(diagnostic, "ENABLED", False)
    diagnostic.log_admission(adder, req, result)
    assert not caplog.records
    monkeypatch.setattr(diagnostic, "ENABLED", True)
    monkeypatch.setattr(diagnostic, "_last_log", {})
    with caplog.at_level("INFO"):
        diagnostic.log_admission(adder, req, result)
        diagnostic.log_admission(adder, req, result)
    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].getMessage().split("OasisKV progress ")[1])
    assert payload["swa_needed"] == 4352
    assert payload["swa_free"] == 4096
    assert not payload["admitted"]


def test_finish_rejects_probe_acceptance_and_writes_complete_gzip(tmp_path):
    trace = OasisKVTrace.__new__(OasisKVTrace)
    trace.v = 1
    trace.pending = [records()[0]]
    trace.writer = tmp_path / "trace.jsonl.gz"
    with pytest.raises(RuntimeError, match="acceptance boundary"):
        trace.finish(NS(commit_lens=torch.tensor([3])))
    trace.finish(NS(commit_lens=torch.tensor([2])))
    with gzip.open(trace.writer, "rt") as stream:
        saved = json.loads(stream.read())
    assert saved["accepted_drafts"] == 1
    assert trace.pending is None


@pytest.mark.parametrize("v", [1, 2, 3])
def test_native_acceptance_cannot_accept_the_reserved_suffix(v):
    # Even when *every* candidate matches, only V may be accepted. The bonus
    # must come from target row V, not a later probe row or a zero-padded row.
    candidates = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.full((5, 10), -100.0)
    logits[torch.arange(5), torch.tensor([2, 3, 4, 5, 6])] = 1
    accepted, bonus, _ = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=logits,
        verify_num_draft_tokens=5,
        cutoff_verify_lens=torch.tensor([v + 1]),
    )
    assert accepted.tolist() == [v]
    assert bonus.tolist() == [v + 2]


@pytest.mark.parametrize("fail_shadow", [False, True])
def test_whole_shadow_loop_restores_target_state_and_never_commits(
    monkeypatch, fail_shadow
):
    c = fake_coordinator()
    c.req_device_buffer_tokens = torch.tensor(
        [[[0, 1, -1, -1, -1, -1]]], dtype=torch.int32
    )
    c.req_device_buffer_token_locs = torch.arange(10, 16, dtype=torch.int32).view(
        1, 1, -1
    )
    state = torch.zeros(1)
    pool = NS(
        _unified_kv=False,
        swa_kv_pool=NS(kv_buffer=[state]),
        c4_kv_pool=NS(kv_buffer=[]),
        c128_kv_pool=NS(kv_buffer=[]),
        c4_indexer_kv_pool=NS(index_k_with_scale_buffer=[]),
        compress_state_pools=[],
        indexer_compress_state_pools=[],
    )
    trace = OasisKVTrace.__new__(OasisKVTrace)
    trace.runner = NS(
        token_to_kv_pool=pool,
        req_to_token_pool=NS(req_generation=torch.tensor([1])),
        hisparse_coordinator=c,
    )
    trace.v, trace.probes, trace.shadows, trace.max_windows = 1, [1, 2], True, 3
    trace.config = {"snapshot_gib": 0.01, "run_id": "test"}
    trace.coordinator, trace.writer, trace.windows = c, None, {}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(1))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _: None)
    trace.layout = lambda width: NS(verify_lens_cpu=[width])
    calls = []

    def forward(**kw):
        width = kw["layout"].verify_lens_cpu[0]
        calls.append(width)
        state.add_(1)
        if fail_shadow and width > 2:
            raise RuntimeError("shadow failed inside target")
        c.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([0]),
            compressed_seq_lens=torch.full((width,), 10),
            top_k_result=torch.tensor(
                [[0, 2], [2, 3], [1, 4], [4, 5]][:width], dtype=torch.int32
            ),
            layer_id=0,
            verify_lens_cpu=[width],
            output_buffer=None,
        )
        logits = torch.zeros(4, 4)
        logits[:, int(state[0])] = 1
        return NS(logits_output=NS(next_token_logits=logits)), None

    trace.worker = NS(device="cpu", _verify_executor=NS(run_compact=forward))
    batch = NS(
        reqs=[NS(rid="r", origin_input_ids=list(range(40)))],
        has_grammar=False,
        return_logprob=False,
        req_pool_indices_cpu=torch.tensor([0]),
        seq_lens_cpu=torch.tensor([40]),
    )
    kwargs = {
        "batch": batch,
        "draft_block_ids": torch.tensor([[20, 21, 22]]),
        "draft_tokens": torch.tensor([[21, 22, 23]]),
        "sampling_info": None,
    }
    if fail_shadow:
        with pytest.raises(RuntimeError, match="shadow failed inside target"):
            trace.run(**kwargs)
        assert state.item() == 0
        assert c.req_device_buffer_tokens.tolist() == [[[0, 1, -1, -1, -1, -1]]]
    else:
        trace.run(**kwargs)
        assert state.item() == 1  # Only the final baseline forward survives.
        assert c.req_device_buffer_tokens.tolist() == [[[0, 1, 2, 3, -1, -1]]]
        assert calls == [2, 3, 3, 4, 4, 2]
        assert len(trace.pending) == 5
        assert all(
            r["probe_demand_h2d_bytes"] == 0
            for r in trace.pending
            if r.get("mode") == "resident"
        )


@pytest.mark.parametrize(
    "config",
    [
        {"dspark_block_size": 7},
        {"block_size": 7},
        {"dflash_config": {"block_size": 7}, "block_size": 4},
    ],
)
def test_plan_reads_checkpoint_window_without_expanding_it(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert runner.checkpoint_gamma(tmp_path) == 7


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, 512),
        ({"index_topk": 1024}, 1024),
        ({"text_config": {"index_topk": 256}}, 256),
    ],
)
def test_plan_uses_native_v4_topk(tmp_path, config, expected):
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert runner.checkpoint_topk(tmp_path) == expected


def write_run(directory, trace, output_ids):
    directory.mkdir()
    (directory / "outputs.jsonl").write_text(
        json.dumps({"request_id": "r", "output_ids": output_ids}) + "\n"
    )
    with gzip.open(directory / "trace.jsonl.gz", "wt") as stream:
        for record in trace:
            stream.write(json.dumps(record) + "\n")


def test_cross_process_audit_rejects_changed_tokens_or_baseline_topk(tmp_path):
    source, following, probe = records()
    control, experiment = tmp_path / "control", tmp_path / "probes"
    write_run(control, [source, following], [1, 2, 3])
    write_run(experiment, [source, following, probe], [1, 2, 3])
    assert auditor.audit(control, experiment)["passed"]
    changed = copy.deepcopy(source)
    changed["topk"][0] = [4, 5]
    with gzip.open(experiment / "trace.jsonl.gz", "wt") as stream:
        for record in [changed, following, probe]:
            stream.write(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="trajectory/residency"):
        auditor.audit(control, experiment)
    (experiment / "outputs.jsonl").write_text(
        json.dumps({"request_id": "r", "output_ids": [9]}) + "\n"
    )
    with pytest.raises(ValueError, match="token IDs differ"):
        auditor.audit(control, experiment)
