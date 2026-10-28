"""Run the real worker constructor and scheduler pool lifecycle on CPU.

Only weight loading, pool allocation, and unrelated planner/observer setup are
stubbed. The trace's environment gate, validation, inventory and output creation
are real, so an early constructor call reproduces the reported startup failure.
"""

import json
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt import runtime_context
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.speculative import oasiskv_trace
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as worker_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("trace_mode", ["disabled", "enabled", "invalid-io"])
@pytest.mark.parametrize("rank", [0, 1])
def test_trace_binds_after_both_pools_and_still_fails_closed(
    monkeypatch, tmp_path, trace_mode, rank
):
    events = []
    output = tmp_path / "artifacts"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "verify_candidates": 1,
                "probes": [1, 2],
                "max_windows": 3,
                "output": str(output),
            }
        )
    )
    if trace_mode == "disabled":
        monkeypatch.delenv("SGLANG_OASISKV_TRACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("SGLANG_OASISKV_TRACE_CONFIG", str(config_path))

    graph = CudaGraphConfig.from_dict(
        {
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        }
    )
    execution = NS(
        graph=NS(cuda_graph_config=graph, disable_cuda_graph=True),
        moe=NS(enable_eplb=False),
    )
    parallel = NS(enable_dp_attention=False, pp_size=1, tp_size=4)
    scheduling = NS(page_size=256, disable_overlap_schedule=True)
    for module in (runtime_context, worker_module):
        monkeypatch.setattr(module, "get_exec", lambda: execution)
        monkeypatch.setattr(module, "get_parallel", lambda: parallel)
        monkeypatch.setattr(module, "get_schedule", lambda: scheduling)

    target_runner = NS(
        model_config=NS(hf_text_config=NS(), vocab_size=128),
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool=None,
        token_to_kv_pool_allocator=None,
        hisparse_coordinator=None,
        enable_hisparse=True,
        is_draft_worker=False,
    )
    draft_runner = NS(model_config=NS(hf_config=NS()), token_to_kv_pool=None)

    def allocate_target():
        events.append("target-pools")
        allocator = object()
        target_runner.memory_pool_config = object()
        target_runner.token_to_kv_pool_allocator = allocator
        target_runner.req_to_token_pool = NS(req_to_token=torch.zeros(1, 8))
        target_runner.token_to_kv_pool = NS(
            _unified_kv=False,
            swa_kv_pool=NS(kv_buffer=[torch.zeros(1)]),
            c4_kv_pool=NS(kv_buffer=[torch.zeros(1)]),
            c128_kv_pool=NS(kv_buffer=[torch.zeros(1)]),
            c4_indexer_kv_pool=NS(index_k_with_scale_buffer=[torch.zeros(1)]),
            compress_state_pools=[],
            indexer_compress_state_pools=[],
            layer_mapping=[NS(compress_layer_id=0, compress_ratio=4)],
        )
        target_runner.hisparse_coordinator = NS(
            is_dsv4_hisparse=True,
            prefetcher=None,
            prefetcher_name=None,
            enable_prefetch=False,
            skip_io=trace_mode == "invalid-io",
            token_to_kv_pool_allocator=allocator,
            item_size_bytes=584,
            top_k=2,
            device_buffer_size=16,
        )

    def allocate_draft(**kwargs):
        assert kwargs["req_to_token_pool"] is target_runner.req_to_token_pool
        assert (
            kwargs["token_to_kv_pool_allocator"]
            is target_runner.token_to_kv_pool_allocator
        )
        assert target_runner.hisparse_coordinator is not None
        events.append("draft-pools")
        draft_runner.token_to_kv_pool = object()

    target = NS(
        model_runner=target_runner,
        device="cpu",
        alloc_memory_pool=allocate_target,
        get_memory_pool=lambda: (
            target_runner.req_to_token_pool,
            target_runner.token_to_kv_pool_allocator,
        ),
    )
    bundle = NS(
        draft_worker=NS(alloc_memory_pool=allocate_draft),
        draft_model_runner=draft_runner,
        draft_model=NS(uses_own_vocab_modules=True, markov_head=object()),
        resolved_attention_backend="test",
    )
    monkeypatch.setattr(worker_module, "build_draft_tp_worker", lambda **kw: bundle)
    monkeypatch.setattr(worker_module, "draft_is_deepseek_v4", lambda **kw: True)
    monkeypatch.setattr(worker_module, "mambaish_config", lambda config: None)
    monkeypatch.setattr(
        worker_module,
        "resolve_runtime_config",
        lambda **kw: NS(
            gamma=7,
            verify_num_draft_tokens=8,
            mask_token_id=99,
        ),
    )
    monkeypatch.setattr(
        worker_module, "DSparkVerifyPlanner", lambda **kw: NS(is_compact_mode=True)
    )
    monkeypatch.setattr(worker_module, "DsparkStepObservers", lambda **kw: NS())
    original_validate = oasiskv_trace.validate_trace_hisparse

    def validate(model_runner):
        assert draft_runner.token_to_kv_pool is not None
        events.append("trace-validation")
        return original_validate(model_runner)

    monkeypatch.setattr(oasiskv_trace, "validate_trace_hisparse", validate)
    monkeypatch.setattr(
        scheduler_module.kv_cache_builder,
        "resolve_decode_retraction_backup",
        lambda **kw: events.append("resolve-retraction"),
    )
    with envs.SGLANG_RAGGED_VERIFY_MODE.override(
        "compact"
    ), envs.SGLANG_SIMULATE_ACC_LEN.override(0):
        worker = worker_module.DSparkWorkerV2(
            server_args=NS(
                is_oasiskv_lookahead=False,
                disaggregation_mode="null",
                speculative_num_draft_tokens=8,
            ),
            gpu_id=0,
            ps=ParallelState.trivial(tp_rank=rank, tp_size=4),
            nccl_port=0,
            target_worker=target,
        )
        assert worker._oasiskv_trace is None
        assert target_runner.hisparse_coordinator is None
        assert not events and not output.exists()
        worker.init_hicache_draft_plan = lambda: events.append("hicache-plan")
        scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
        scheduler.tp_worker, scheduler.draft_worker = target, worker
        if trace_mode == "invalid-io":
            with pytest.raises(ValueError, match="skip_io=True"):
                scheduler.init_memory_pools()
            assert events == [
                "target-pools",
                "resolve-retraction",
                "draft-pools",
                "trace-validation",
            ]
            assert worker._oasiskv_trace is None
            assert not output.exists()
            return
        scheduler.init_memory_pools()

    expected = ["target-pools", "resolve-retraction", "draft-pools"]
    if trace_mode == "enabled":
        expected.append("trace-validation")
        assert worker._oasiskv_trace.runner is target_runner
        assert worker._oasiskv_trace.coordinator is target_runner.hisparse_coordinator
        assert list(oasiskv_trace.target_state_tensors(target_runner))
    else:
        assert worker._oasiskv_trace is None
    assert events == expected + ["hicache-plan"]
    if trace_mode == "enabled" and rank == 0:
        runtime = json.loads((output / "runtime.json").read_text())
        assert runtime["tp"] == 4
        assert runtime["hisparse_runtime"]["runner_allocator_type"] == "builtins.object"
        assert runtime["hisparse_runtime"]["skip_io"] is False
        assert (output / "trace.jsonl.gz").exists()
    else:
        assert not output.exists()
