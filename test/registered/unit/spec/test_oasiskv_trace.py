"""Offline probe isolation, real miss accounting and next-window matching."""

import argparse
import copy
import gzip
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
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
    compare_observations,
    compare_topk,
    replace_method,
    replay_diagnostics,
    resident_lookup,
    validate_trace_hisparse,
)
from sglang.srt.speculative.oasiskv_trace_topk import canonicalize_topk_
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
    report = analyzer.analyze([source, following, probe])
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
    assert analyzer.analyze([source, following])["matched_records"] == 0


def test_full_union_does_not_rank_or_truncate_unsorted_topk():
    source, following, probe = records()
    probe.update(p=2, input_ids=[20, 21, 25, 26], topk=[[3, 4], [4, 4]])
    sample = analyzer.analyze([source, following, probe])["samples"][0]
    assert sample["prefetch_candidate_units"] == 2
    assert sample["potential_first_fetch_bytes"] == 2 * 584
    assert sample["potential_unique_miss_coverage"] == pytest.approx(2 / 3)
    probe["topk"] = [[4, 3], [3, 3]]
    assert analyzer.analyze([source, following, probe])["samples"][0] == sample


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
        analyzer.analyze([source, following, probe])


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
        analyzer.analyze([source, following, probe])


def test_generation_reuse_missing_steps_and_duplicate_windows():
    source, following, probe = records()
    following["generation"] = 2
    report = analyzer.analyze([source, following, probe])
    assert report["matched_records"] == 0
    assert report["skipped"]["no_immediate_next_window"] == 1
    following.update(generation=1, window_id=2)
    assert analyzer.analyze([source, following, probe])["matched_records"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        analyzer.analyze([source, source])


def test_unknown_misses_are_not_filled_in_from_topk():
    source, following, probe = records()
    following["miss_events"] = None
    sample = analyzer.analyze([source, following, probe])["samples"][0]
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
    with replace_method(obj, "read", lambda: 5):
        with pytest.raises(RuntimeError), replace_method(obj, "read", lambda: 6):
            assert obj.read() == 6
            raise RuntimeError("nested observation failed")
        assert obj.read() == 5
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


def test_observations_do_not_leak_rows_or_overrides_across_widths():
    c = fake_coordinator()
    original_swap, original_kernel = (
        c.swap_in_selected_pages_spec,
        c._run_swap_in_kernel,
    )
    saved = []
    for width, mode in [(3, "exact"), (5, "resident"), (4, "exact"), (3, "exact")]:
        c.req_device_buffer_tokens[0, 0] = torch.tensor([0, 1, -1, -1])
        raw = torch.tensor([[0, 2], [2, 3]] + [[1, 3]] * (width - 2), dtype=torch.int32)
        expected = raw.tolist()
        observer = ForwardObservation(c, 3, mode, 10, 0)
        with observer.capture():
            c.swap_in_selected_pages_spec(
                req_pool_indices=torch.tensor([0]),
                compressed_seq_lens=torch.full((width,), 10),
                top_k_result=raw,
                layer_id=0,
                verify_lens_cpu=[width],
                output_buffer=None,
            )
        raw.fill_(-99)  # Later forwards may overwrite their temporary GPU buffers.
        assert observer.layers[0]["topk"] == expected
        assert len(observer.layers[0]["miss_events"]) == width
        assert c.swap_in_selected_pages_spec is original_swap
        assert c._run_swap_in_kernel is original_kernel
        saved.append((observer, copy.deepcopy(observer.layers)))
    for observer, expected in saved:
        assert observer.layers == expected


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


def test_live_reference_accepts_both_hot_and_reserved_copies_but_not_stale_slots():
    # A newer query may load the formerly-newest C4 into a hot slot, while the
    # earlier query still references its valid reserved-slot copy.
    check_live_locations([[7], [7]], [[10], [14]], [7, 7], [10, 14], 8, buffer_size=4)
    with pytest.raises(RuntimeError, match='"history_union_units": 1') as error:
        check_live_locations([[7]], [[10]], [6, 7], [10, 14], 8, buffer_size=4)
    assert '"tokens_at_returned_location": [6]' in str(error.value)
    assert '"hot_buffer_size": 4' in str(error.value)


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
@pytest.mark.parametrize("topk_order", ["native", "logical", "logical-all"])
def test_trace_startup_uses_resolved_graph_backends(
    monkeypatch, active_phase, topk_order
):
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
    config = {
        "verify_candidates": 1,
        "probes": [1, 2],
        "max_windows": 2,
        "topk_order": topk_order,
    }
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        if active_phase:
            with pytest.raises(ValueError, match="ordinary DSPARK, eager"):
                OasisKVTrace(worker, config)
        else:
            trace = OasisKVTrace(worker, config)
            assert trace.coordinator is worker.model_runner.hisparse_coordinator
            assert trace.coordinator.oasiskv_trace_topk_order == topk_order


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


@pytest.mark.parametrize("wide", [False, True])
def test_delivered_server_command_is_accepted_by_real_cli_parser(wide):
    from sglang.srt.arg_groups.speculative_hook import _handle_dspark

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
    if wide:
        args.verify_width, args.probes = 6, [1, 2, 3, 4, 5, 6]
    command = runner.server_command(args, 5 if wide else 7)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(command[3:])
    assert parsed.speculative_num_draft_tokens == (13 if wide else 8)
    assert parsed.speculative_dspark_block_size == (12 if wide else 7)
    assert parsed.tp_size == 4
    assert parsed.disable_overlap_schedule
    assert parsed.attention_backend == "dsv4"
    assert parsed.moe_runner_backend == "flashinfer_mxfp4"
    assert json.loads(parsed.hisparse_config)["host_to_device_ratio"] == 1
    # Exercise the semantic gamma cross-check too, not just argparse. Passing
    # num-draft-tokens alone would be rejected for the trained G=5 checkpoint.
    server_args = ServerArgs(model_path="dummy")
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = args.draft
    server_args.speculative_dspark_block_size = parsed.speculative_dspark_block_size
    server_args.speculative_num_draft_tokens = parsed.speculative_num_draft_tokens
    _handle_dspark(server_args)
    assert server_args.speculative_num_draft_tokens == (13 if wide else 8)


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
        prefetch_units="all",
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
    trace.finish(NS(commit_lens=torch.tensor([2]), out_tokens=torch.tensor([[7, 8]])))
    with gzip.open(trace.writer, "rt") as stream:
        saved = json.loads(stream.read())
    assert saved["accepted_drafts"] == 1
    assert trace.pending is None


def test_finish_rejects_acceptance_that_disagrees_with_pre_shadow_control(tmp_path):
    trace = OasisKVTrace.__new__(OasisKVTrace)
    trace.v = 1
    trace.pending = [records()[0]]
    trace.pending_baseline = [{"committed_output_ids": [22]}]
    trace.writer = tmp_path / "trace.jsonl.gz"
    with pytest.raises(RuntimeError, match="pre-shadow greedy baseline"):
        trace.finish(
            NS(commit_lens=torch.tensor([1]), out_tokens=torch.tensor([[99, -1]]))
        )
    assert (tmp_path / "rejection.json").exists()
    assert not trace.writer.exists()
    assert not (tmp_path / "paired-baseline.jsonl.gz").exists()


@pytest.mark.parametrize("v", [1, 2, 3, 5, 6])
def test_native_acceptance_cannot_accept_the_reserved_suffix(v):
    # Even when *every* candidate matches, only V may be accepted. The bonus
    # must come from target row V, not a later probe row or a zero-padded row.
    width = 12
    candidates = torch.arange(1, width + 1).view(1, -1)
    logits = torch.full((width, width + 2), -100.0)
    logits[torch.arange(width), torch.arange(2, width + 2)] = 1
    accepted, bonus, _ = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=logits,
        verify_num_draft_tokens=width,
        cutoff_verify_lens=torch.tensor([v + 1]),
    )
    assert accepted.tolist() == [v]
    assert bonus.tolist() == [v + 2]


@pytest.mark.parametrize(
    "variant",
    [
        "normal",
        "permutation",
        "changed_ids",
        "changed_logits",
        "same_argmax_logits",
        "changed_hidden",
        "replay_check",
        "p0",
        "p0_only",
    ],
)
@pytest.mark.parametrize("fail_shadow", [False, True])
def test_whole_shadow_loop_restores_target_state_and_never_commits(
    monkeypatch, tmp_path, fail_shadow, variant
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
        layer_mapping=[None] * 9 + [NS(compress_ratio=4, compress_layer_id=0)],
    )
    trace = OasisKVTrace.__new__(OasisKVTrace)
    trace.runner = NS(
        token_to_kv_pool=pool,
        req_to_token_pool=NS(req_generation=torch.tensor([1])),
        hisparse_coordinator=c,
    )
    trace.v, trace.probes, trace.shadows, trace.max_windows = 1, [1, 2], True, 3
    if variant == "replay_check":
        trace.probes = []
    elif variant == "p0":
        trace.probes = [0, 1, 2]
    elif variant == "p0_only":
        trace.probes = [0]
    trace.config = {"snapshot_gib": 0.01, "run_id": "test"}
    trace.coordinator, trace.writer, trace.windows = c, tmp_path / "trace.jsonl.gz", {}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(1))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _: None)
    trace.layout = lambda width: NS(verify_lens_cpu=[width])
    calls = []

    def forward(**kw):
        width = kw["layout"].verify_lens_cpu[0]
        assert kw["draft_tokens"] is kwargs["draft_tokens"]
        assert kw["draft_block_ids"] is kwargs["draft_block_ids"]
        calls.append(width)
        state.add_(1)
        if fail_shadow and width > 2:
            raise RuntimeError("shadow failed inside target")
        raw = torch.tensor([[0, 2], [2, 3], [1, 4], [4, 5]][:width], dtype=torch.int32)
        if variant == "permutation" and len(calls) % 2 == 0:
            raw = raw.flip(-1)
        if variant == "changed_ids" and len(calls) == 6:
            raw[0, 1] = 4
        c.swap_in_selected_pages_spec(
            req_pool_indices=torch.tensor([0]),
            compressed_seq_lens=torch.full((width,), 10),
            top_k_result=raw,
            layer_id=0,
            verify_lens_cpu=[width],
            output_buffer=None,
        )
        logits = torch.zeros(4, 4)
        logits[:, int(state[0])] = 1
        if variant == "changed_logits" and len(calls) == 6:
            logits[:, 2] = 2
        if variant == "same_argmax_logits" and len(calls) == 6:
            logits[:, 1] = 2
        hidden = torch.ones(width, 4)
        if variant == "changed_hidden" and len(calls) == 6:
            hidden.add_(1)
        return NS(logits_output=NS(next_token_logits=logits)), hidden

    trace.worker = NS(gamma=3, device="cpu", _verify_executor=NS(run_compact=forward))
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
    if fail_shadow and variant not in ("replay_check", "p0_only"):
        with pytest.raises(RuntimeError, match="shadow failed inside target"):
            trace.run(**kwargs)
        assert state.item() == 0
        assert c.req_device_buffer_tokens.tolist() == [[[0, 1, -1, -1, -1, -1]]]
    elif variant in (
        "changed_ids",
        "changed_logits",
        "same_argmax_logits",
        "changed_hidden",
    ):
        with pytest.raises(RuntimeError, match="trace rejected"):
            trace.run(**kwargs)
        assert trace.pending is None
        assert not trace.writer.exists()
        diagnostic = json.loads((tmp_path / "rejection.json").read_text())
        with gzip.open(tmp_path / "rejected-trace.jsonl.gz", "rt") as stream:
            rejected = [json.loads(line) for line in stream]
        assert len(rejected) == 6  # Four shadows plus trial/replay baseline.
        assert {r["kind"] for r in rejected} == {"rejected_window", "rejected_probe"}
        if variant == "changed_ids":
            assert diagnostic["trial_predictions"] == diagnostic["replayed_predictions"]
            assert diagnostic["layer_differences"] == [
                {"layer": 0, "fields": ["topk", "resident_after", "miss_events"]}
            ]
        elif variant == "changed_logits":
            assert diagnostic["differing_layers"] == []
            assert diagnostic["trial_predictions"] != diagnostic["replayed_predictions"]
        else:
            assert diagnostic["differing_layers"] == []
            assert diagnostic["trial_predictions"] == diagnostic["replayed_predictions"]
            assert not diagnostic[
                (
                    "hidden_states_exact_equal"
                    if variant == "changed_hidden"
                    else "logits_exact_equal"
                )
            ]
    else:
        trace.run(**kwargs)
        assert state.item() == 1  # Only the final baseline forward survives.
        assert c.req_device_buffer_tokens.tolist() == [[[0, 1, 2, 3, -1, -1]]]
        assert calls == (
            [2, 2] if variant in ("replay_check", "p0_only") else [2, 3, 3, 4, 4, 2]
        )
        assert len(trace.pending) == 1 + 2 * len(trace.probes)
        for record in trace.pending:
            assert record["model_layer"] == 9
            assert record["proposal_input_ids"] == [20, 21, 22, 23]
            assert (
                record["input_ids"]
                == record["proposal_input_ids"][: len(record["input_ids"])]
            )
            if record.get("p") == 0:
                assert record["topk"] == []
                assert record["input_ids"] == [20, 21]
                assert (
                    record["probe_demand_h2d_bytes"]
                    == record["probe_workspace_d2d_bytes"]
                    == 0
                )
        assert all(
            r["probe_demand_h2d_bytes"] == 0
            for r in trace.pending
            if r.get("mode") == "resident"
        )
        diagnostic = json.loads((tmp_path / "replay-audit.jsonl").read_text())
        assert diagnostic["passed"]
        assert diagnostic["topk_order_only_layers"] == (
            [0] if variant == "permutation" else []
        )
        if variant != "replay_check":
            trace.finish(
                NS(
                    commit_lens=torch.tensor([1]),
                    out_tokens=torch.tensor([[1, -1, -1]]),
                )
            )
            (tmp_path / "config.json").write_text(
                json.dumps(
                    {
                        "shadows": True,
                        "probes": trace.probes,
                        "verify_candidates": 1,
                    }
                )
            )
            (tmp_path / "runtime.json").write_text(
                json.dumps({"c4_layer_to_model_layer": {"0": 0}})
            )
            (tmp_path / "outputs.jsonl").write_text(
                json.dumps({"request_id": "r", "output_ids": [20, 1]}) + "\n"
            )
            assert auditor.audit_within_server(tmp_path)["passed"]


def test_comparison_preserves_query_boundaries_duplicates_and_miss_counts():
    left = {
        0: {
            "topk": [[1, 2, -1], [3, 4, -1]],
            "resident_after": [1, 2, 3, 4],
            "miss_events": [{"row": 0, "ids": [2, 1]}, {"row": 1, "ids": [4, 3]}],
        }
    }
    right = copy.deepcopy(left)
    right[0]["topk"][0].reverse()
    right[0]["miss_events"][0]["ids"].reverse()
    assert compare_observations(left, right) == []
    # Same argmax need not mean identical logits; record both facts.
    diagnostic = replay_diagnostics(
        left, right, torch.tensor([[1.0, 0]]), torch.tensor([[2.0, 0]])
    )
    assert diagnostic["trial_predictions"] == diagnostic["replayed_predictions"]
    assert diagnostic["topk_order_only_layers"] == [0]
    assert not diagnostic["logits_exact_equal"]
    assert diagnostic["logits_max_abs_diff_finite"] == 1
    for field, value in (
        ("topk", [[1, 1, 2], [3, 4, -1]]),
        ("topk", [[3, 4, -1], [1, 2, -1]]),
        ("topk", [[1, 2, -1]]),
        ("resident_after", [1, 2, 3]),
        ("miss_events", [{"row": 0, "ids": [1]}, {"row": 1, "ids": [3, 4]}]),
        ("miss_events", [{"row": 0, "ids": [1, 2, 2]}, {"row": 1, "ids": [3, 4]}]),
        ("miss_events", [{"row": 1, "ids": [1, 2]}, {"row": 0, "ids": [3, 4]}]),
    ):
        changed = copy.deepcopy(left)
        changed[0][field] = value
        assert compare_observations(left, changed) == [{"layer": 0, "fields": [field]}]
    assert compare_observations(left, {}) == [
        {"layer": 0, "fields": ["layer_presence"]}
    ]


def test_row_diagnostics_distinguish_order_membership_and_multiplicity():
    report = compare_topk(
        [[0, 2, 1], [94, 10], [5, 5, -1], [9]],
        [[2, 1, 0], [232, 10], [5, -1, -1]],
    )
    assert report["rows_reordered_only"] == 1
    assert report["rows_different_ids"] == 3
    assert report["rows"][0]["trial_only_ids"] == []
    assert report["rows"][1]["trial_only_ids"] == [94]
    assert report["rows"][1]["replayed_only_ids"] == [232]
    assert report["rows"][2]["trial_only_ids"] == [5]
    assert report["rows"][2]["replayed_only_ids"] == [-1]
    assert not report["rows"][3]["replayed_row_present"]


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_canonical_order_keeps_raw_page_scratch_alignment_and_storage(noncontiguous):
    raw = torch.tensor([[7, 3, -1, 5], [-1, -1, -1, -1]], dtype=torch.int32)
    # Physical pages do NOT have the same order as logical C4 IDs.
    pages = torch.tensor([[20, 500, -1, 90], [-1, -1, -1, -1]], dtype=torch.int32)
    if noncontiguous:
        raw = torch.stack((raw, raw), dim=-1)[..., 0]
        pages = torch.stack((pages, pages), dim=-1)[..., 0]
    pointers = raw.data_ptr(), pages.data_ptr()
    canonicalize_topk_(raw, pages)
    assert (raw.data_ptr(), pages.data_ptr()) == pointers
    assert raw.tolist() == [[3, 5, 7, -1], [-1, -1, -1, -1]]
    assert pages.tolist() == [[500, 90, 20, -1], [-1, -1, -1, -1]]
    # These are the aligned logical page indices later cloned for scratch lookup.
    saved = pages.clone()
    canonicalize_topk_(raw, pages)
    assert torch.equal(saved, pages)
    with pytest.raises(ValueError, match="matching 2D"):
        canonicalize_topk_(raw, pages[:, :2])


@pytest.mark.parametrize("order", ["logical", "logical-all"])
def test_analysis_rejects_mixing_attention_visitation_order(order):
    source, following, probe = records()
    changed = dict(probe, topk_order=order)
    with pytest.raises(ValueError, match="different topk_order"):
        analyzer.analyze([source, following, changed])
    report = analyzer.analyze(
        [dict(r, topk_order=order) for r in (source, following, probe)]
    )
    assert report["topk_order"] == order


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


def test_cross_process_audit_accepts_scatter_order_but_rejects_failed_runs(tmp_path):
    source, following, probe = records()
    control, experiment = tmp_path / "control", tmp_path / "probes"
    write_run(control, [source, following], [1, 2, 3])
    changed = copy.deepcopy(following)
    for row in changed["topk"]:
        row.reverse()
    for event in changed["miss_events"]:
        event["ids"].reverse()
    write_run(experiment, [source, changed, probe], [1, 2, 3])
    assert auditor.audit(control, experiment)["passed"]
    # A previous successfully committed window is not evidence the run passed.
    (experiment / "rejection.json").write_text("{}")
    with pytest.raises(ValueError, match="Rejected shadow replay"):
        auditor.audit(control, experiment)
    with pytest.raises(ValueError, match="Rejected shadow replay"):
        list(analyzer.read_records([experiment / "trace.jsonl.gz"]))


def test_cross_process_audit_rejects_different_visitation_order(tmp_path):
    source, following, probe = records()
    control, experiment = tmp_path / "control", tmp_path / "probes"
    write_run(control, [source, following], [1, 2, 3])
    write_run(
        experiment,
        [dict(record, topk_order="logical") for record in (source, following, probe)],
        [1, 2, 3],
    )
    with pytest.raises(ValueError, match="trajectory/residency"):
        auditor.audit(control, experiment)


@pytest.mark.parametrize("order", ["logical", "logical-all"])
def test_untraced_cannot_claim_fixed_attention_order(order):
    with pytest.raises(ValueError, match="no ordering hook"):
        runner.validate_args(NS(mode="untraced", topk_order=order), 5)


def test_audit_separates_committed_output_from_kv_baseline_and_persists_failure(
    tmp_path,
):
    source, following, probe = records()
    control, experiment = tmp_path / "control", tmp_path / "probes"
    write_run(control, [source, following], [1, 2, 3])
    changed = copy.deepcopy(source)
    changed["input_ids"][1] = 999  # Rejected proposal; committed output stays equal.
    changed["topk"][1] = [3, 4]
    changed["resident_after"] = [3, 4]
    write_run(experiment, [changed, following, probe], [1, 2, 3])
    with pytest.raises(auditor.AuditMismatch) as caught:
        auditor.audit(control, experiment)
    report = caught.value.report
    assert report["output_ids_equal"] and report["window_keys_equal"]
    assert report["differing_layer_windows"] == 1
    assert report["differing_fields"] == {
        "input_ids": 1,
        "topk": 1,
        "resident_after": 1,
    }
    assert report["inventory"]["probes"]["records_by_kind"] == {"window": 2, "probe": 1}
    assert report["inventory"]["probes"]["request_windows_by_kind"] == {
        "window": 2,
        "probe": 1,
    }
    output = tmp_path / "audit.json"
    process = subprocess.run(
        [
            sys.executable,
            auditor.__file__,
            str(control),
            str(experiment),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert not output.exists()
    assert json.loads((tmp_path / "audit-failure.json").read_text()) == report


def test_cross_process_audit_does_not_compare_shadow_topk(tmp_path):
    source, following, probe = records()
    control, experiment = tmp_path / "control", tmp_path / "probes"
    probe["topk"] = [[999, 1000]]
    write_run(control, [source, following], [1, 2, 3])
    write_run(experiment, [source, following, probe], [1, 2, 3])
    assert auditor.audit(control, experiment)["passed"]


def write_paired_run(directory):
    source, following, probe = records()
    source["committed_output_ids"], following["committed_output_ids"] = [22], [24]
    trace = [source, following]
    for window in (source, following):
        for mode in ("exact", "resident"):
            trace.append(
                dict(
                    probe,
                    window_id=window["window_id"],
                    snapshot_id=window["snapshot_id"],
                    input_ids=window["input_ids"] + [25],
                    mode=mode,
                )
            )
    write_run(directory, trace, [20, 22, 24])
    with gzip.open(directory / "paired-baseline.jsonl.gz", "wt") as stream:
        for window in (source, following):
            stream.write(json.dumps(window) + "\n")
    (directory / "config.json").write_text(
        json.dumps({"probes": [1], "shadows": True, "verify_candidates": 1})
    )
    (directory / "runtime.json").write_text(
        json.dumps({"c4_layer_to_model_layer": {"2": 4}})
    )
    diagnostics = []
    for window in (source, following):
        predictions = window["committed_output_ids"] + [99]
        diagnostics.append(
            dict(
                window,
                paired_audit_version=1,
                passed=True,
                logits_exact_equal=True,
                hidden_states_exact_equal=True,
                differing_layers=[],
                layer_ids=[2],
                trial_predictions=predictions,
                replayed_predictions=predictions,
                topk_order="native",
            )
        )
    (directory / "replay-audit.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in diagnostics)
    )
    return trace, diagnostics


def test_within_server_audit_and_analysis_bind_exact_artifacts(tmp_path):
    directory = tmp_path / "probes"
    write_paired_run(directory)
    report = auditor.audit_within_server(directory)
    assert report["passed"] and report["layer_windows"] == 2
    assert report["scope"] == "within-server-snapshot-paired"
    path = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            auditor.__file__,
            str(directory),
            "--scope",
            "within-server",
            "--output",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    trace = directory / "trace.jsonl.gz"
    assert analyzer.validated_audit(path, [trace]) == report
    analysis = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            analyzer.__file__,
            str(trace),
            "--audit",
            str(path),
            "--output",
            str(analysis),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads((analysis / "report.json").read_text())
    assert result["validation"] == report and result["matched_records"] == 2
    assert result["prefetch_cap_units"] is None
    assert result["prefetch_policy"] == "full-probe-union"
    assert result["analysis_version"] == 3
    assert result["samples"][0]["model_layer"] == 4
    with gzip.open(trace, "at") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="different trace bytes"):
        analyzer.validated_audit(path, [trace])


@pytest.mark.parametrize(
    "defect",
    [
        "topk",
        "input_ids",
        "resident_after",
        "miss_events",
        "missing_pair",
        "missing_diagnostic",
        "hidden_drift",
        "logits_drift",
        "old_evidence",
        "missing_probe",
        "wrong_snapshot",
        "wrong_output",
        "fake_acceptance",
        "failure_marker",
    ],
)
def test_within_server_audit_rejects_incomplete_or_changed_evidence(tmp_path, defect):
    directory = tmp_path / "probes"
    trace, diagnostics = write_paired_run(directory)
    if defect in ("topk", "input_ids", "resident_after", "miss_events"):
        trace[0][defect] = []
    elif defect == "missing_pair":
        (directory / "paired-baseline.jsonl.gz").unlink()
    elif defect == "missing_diagnostic":
        diagnostics.pop()
    elif defect in ("hidden_drift", "logits_drift"):
        diagnostics[0][
            (
                "hidden_states_exact_equal"
                if defect == "hidden_drift"
                else "logits_exact_equal"
            )
        ] = False
    elif defect == "old_evidence":
        diagnostics[0].pop("paired_audit_version")
    elif defect == "missing_probe":
        trace.pop()
    elif defect == "wrong_snapshot":
        trace[-1]["snapshot_id"] = "another snapshot"
    elif defect == "wrong_output":
        (directory / "outputs.jsonl").write_text(
            json.dumps({"request_id": "r", "output_ids": [20, 999, 24]})
        )
    elif defect == "fake_acceptance":
        trace[0]["committed_output_ids"] = [99]
        with gzip.open(directory / "paired-baseline.jsonl.gz", "wt") as stream:
            for window in trace[:2]:
                stream.write(json.dumps(window) + "\n")
    elif defect == "failure_marker":
        (directory / "failure.json").write_text("{}")
    with gzip.open(directory / "trace.jsonl.gz", "wt") as stream:
        for record in trace:
            stream.write(json.dumps(record) + "\n")
    (directory / "replay-audit.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in diagnostics)
    )
    with pytest.raises(ValueError):
        auditor.audit_within_server(directory)


def test_independent_control_drift_is_diagnostic_without_weakening_paired_audit(
    tmp_path,
):
    probes = tmp_path / "probes"
    trace, _ = write_paired_run(probes)
    control = tmp_path / "control"
    changed = copy.deepcopy(trace[:2])
    changed[0]["input_ids"][1] = 999
    changed[0]["topk"][1] = [3, 4]
    write_run(control, changed, [20, 22, 24])
    with pytest.raises(auditor.AuditMismatch):
        auditor.audit(control, probes)
    assert auditor.audit_within_server(probes)["passed"]
    output = tmp_path / "cross-server-audit.json"
    process = subprocess.run(
        [
            sys.executable,
            auditor.__file__,
            str(control),
            str(probes),
            "--output",
            str(output),
            "--diagnostic-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert not output.exists()
    diagnostic = json.loads((tmp_path / "cross-server-audit-failure.json").read_text())
    assert not diagnostic["passed"] and diagnostic["output_ids_equal"]


def test_paired_plan_uses_one_server():
    plan = runner.mode_plan(NS(mode="paired", skip_full_control=False, port=32100))
    assert [(mode, args.port) for mode, args in plan] == [("probes", 32100)]


@pytest.mark.parametrize("skip_full", [False, True])
def test_mode_plan_assigns_ports_without_mutating_args(skip_full):
    args = NS(mode="experiment", skip_full_control=skip_full, port=32100)
    plan = runner.mode_plan(args)
    assert [(mode, a.port) for mode, a in plan] == [
        ("control", 32100),
        ("probes", 32101),
    ] + ([] if skip_full else [("full-control", 32102)])
    assert args.port == 32100
    args.mode, args.port = "control", 65535
    assert runner.mode_plan(args)[0][1].port == 65535
    args.mode = "experiment"
    with pytest.raises(ValueError, match="mode ports"):
        runner.mode_plan(args)


def test_port_check_rejects_listener_and_accepts_reusable_closed_connection():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with pytest.raises(OSError):
            runner.check_listen_port(port)
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = listener.accept()
            # Server active-close leaves a TIME_WAIT socket, not a live server.
            connection.close()
            assert client.recv(1) == b""
    runner.check_listen_port(port)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc")
def test_server_teardown_reaps_orphan_that_escaped_the_session(tmp_path):
    # This child is deliberately outside the experiment; cleanup must not kill it.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    marker = tmp_path / "orphan.json"
    script = """
import json, os, signal, socket, sys, time
from pathlib import Path
if os.fork():
    os._exit(0)
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
sock = socket.socket()
sock.bind(('127.0.0.1', 0))
sock.listen()
Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'port': sock.getsockname()[1]}))
while True:
    time.sleep(1)
"""
    try:
        with runner.ServerProcessTree() as tree:
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(marker)], start_new_session=True
            )
            tree.attach(process)
            try:
                process.wait(timeout=5)
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                data = json.loads(marker.read_text())
                assert runner.process_table()[data["pid"]][1] == os.getpid()
                assert process.poll() == 0  # Old stop() would skip all cleanup.
            finally:
                tree.stop(term_timeout=0.1, kill_timeout=5)
        assert data["pid"] not in runner.process_table()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", data["port"]))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
