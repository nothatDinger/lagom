"""DSPARK proposal -> paired target -> root-only commit, without checkpoints.

Run the production worker, verify executor and batch preparation on CPU;
replace model execution, CUDA staging and cache-location allocation at their
boundaries. Actual graph replay is covered by test_oasiskv_graph_inputs.py.
"""

from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _handle_oasiskv_lookahead,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    ragged_verify_compact_graphs_enabled,
)
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    VerifyWindow,
)
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.oasiskv_lookahead import OasisKVCommitBuffers
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _args(*, v4=False, **kwargs):
    args = ServerArgs(model_path="dummy")
    args.device = "cuda"
    args.speculative_algorithm = "DSPARK"
    args.speculative_draft_model_path = "dspark-checkpoint"
    args.disable_radix_cache = True
    args.enable_hisparse = v4
    args.attention_backend = None if v4 else "oasiskv"
    args.hisparse_config = '{"prefetcher":"oasiskv"}' if v4 else "{}"
    for name, value in kwargs.items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize("v4", [False, True])
def test_dspark_configuration_keeps_one_candidate_and_decode_graphs(v4):
    args = _args(v4=v4)
    # The checkpoint's training block width must not replace the one-step
    # runtime width. Compact/ragged env settings must not trim the probe.
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        _handle_oasiskv_lookahead(args)
        _handle_dspark(args)
        assert args.speculative_algorithm == "DSPARK"
        assert args.speculative_dspark_block_size == 1
        assert args.speculative_num_draft_tokens == 2
        assert args.is_oasiskv_lookahead
        assert not args.disable_cuda_graph
        planner = DSparkVerifyPlanner(
            draft_model=NS(confidence_head=object()),
            gamma=1,
            model_runner=object(),
            device="cpu",
            tp_rank=0,
            server_args=args,
            verify_num_draft_tokens=2,
        )
        assert planner.mode_value == "static"
        assert not planner.carries_confidence
        assert planner.confidence_budget_prepare() is None
        assert not ragged_verify_compact_graphs_enabled(
            SpeculativeAlgorithm.DSPARK, oasiskv_lookahead=True
        )
        # Ordinary DSPARK still uses compact graphs under the same env.
        assert ragged_verify_compact_graphs_enabled(SpeculativeAlgorithm.DSPARK)


@pytest.mark.parametrize(
    "override",
    [
        {"speculative_dspark_block_size": 4},
        {"speculative_num_draft_tokens": 5},
        {"speculative_eagle_topk": 2},
    ],
)
def test_dspark_rejects_nonpaired_runtime_geometry(override):
    with pytest.raises(ValueError, match="requires"):
        _handle_oasiskv_lookahead(_args(**override))


def test_dspark_bundled_checkpoint_default_still_works():
    args = _args(v4=True, speculative_draft_model_path=None)
    args.model_config = NS(
        hf_config=NS(
            architectures=["DeepseekV4ForCausalLM"],
            dspark_block_size=5,
            dspark_markov_rank=256,
            dspark_target_layer_ids=[40, 41, 42],
            dspark_noise_token_id=128799,
        )
    )
    _handle_oasiskv_lookahead(args)
    _handle_dspark(args)
    assert args.speculative_draft_model_path == args.model_path
    assert args.speculative_dspark_block_size == 1
    assert args.speculative_num_draft_tokens == 2


def test_dspark_single_candidate_uses_anchor_and_trained_markov_head(monkeypatch):
    # A multi-token checkpoint can be used without modifying its training
    # config. The proposer and attention metadata supply the runtime width.
    monkeypatch.setattr(
        "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
        lambda: False,
    )
    embed = torch.nn.Embedding(16, 4)
    head = VanillaMarkov(vocab_size=16, markov_rank=2)
    with torch.no_grad():
        head.markov_w1.weight.zero_()
        head.markov_w1.weight[3, 0] = 1
        head.markov_w1.weight[8, 1] = 1
        head.markov_w2.weight.zero_()
        head.markov_w2.weight[4, 0] = 10
        head.markov_w2.weight[9, 1] = 10
    model = NS(
        gamma=5,
        block_size=5,
        markov_head=head,
        compute_base_logits=lambda hidden: (torch.zeros(hidden.shape[0], 16), None),
    )
    batch, _ = _batch((7, 511))
    draft_input = make_next_draft_input(
        bonus_tokens=torch.tensor([3, 8]), new_seq_lens=batch.seq_lens
    )
    positions = batch.seq_lens[:, None] + torch.arange(2)
    locs = torch.tensor([[100, 101], [200, 201]])
    window = VerifyWindow(positions, locs.flatten(), locs)

    def forward(fb):
        assert fb.input_ids.tolist() == [3, 8]
        assert fb.spec_info.draft_token_num == 1
        assert fb.positions.tolist() == batch.seq_lens.tolist()
        assert fb.out_cache_loc.tolist() == [100, 200]
        assert fb.seq_lens_cpu.tolist() == (batch.seq_lens + 1).tolist()
        assert not fb.is_oasiskv_paired
        return NS(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, hidden_states=fb.input_embeds
            ),
            can_run_graph=False,
        )

    proposer = DraftBlockProposer(
        draft_model=model,
        draft_model_runner=NS(forward=forward, decode_cuda_graph_runner=None),
        gamma=1,
        mask_token_id=15,
        draft_block_spec_info=make_draft_block_spec_info(
            draft_token_num=1, device="cpu"
        ),
    )
    result = proposer.propose(
        batch=batch,
        draft_input=draft_input,
        verify_window=window,
        bs=2,
        device="cpu",
        target_model=NS(get_input_embeddings=lambda: embed),
        sampling_info=None,
    )
    assert result.draft_block.draft_tokens.tolist() == [[4], [9]]
    assert result.confidence is None
    assert model.gamma == model.block_size == 5


def _worker(monkeypatch, *, graph=False, v4=False, candidate_matches=False, fail=None):
    import sglang.srt.speculative.dspark_components.dspark_worker_v2 as module

    events = []
    committed_features = {}
    state = NS(fail=fail)
    monkeypatch.setattr(
        "sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify",
        lambda batch: None,
    )

    def init_new(batch, runner, **kwargs):
        return NS(
            batch_size=batch.seq_lens.numel(),
            input_ids=batch.input_ids,
            positions=batch.spec_info.positions,
            spec_info=batch.spec_info,
            extend_seq_lens_cpu=None,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            seq_lens_sum=batch.seq_lens_sum,
            out_cache_loc=batch.out_cache_loc,
            forward_mode=batch.forward_mode,
            req_pool_indices=batch.req_pool_indices,
            sampling_info=batch.sampling_info,
        )

    monkeypatch.setattr(ForwardBatch, "init_new", init_new)

    def alloc_window(**kw):
        batch = kw["batch"]
        positions = batch.seq_lens[:, None] + torch.arange(2)
        # Physical slots need not match request-major row indices.
        locs = batch.req_pool_indices[:, None] * 1000 + positions
        return VerifyWindow(positions, locs.flatten(), locs)

    monkeypatch.setattr(module, "alloc_verify_window", alloc_window)

    def plan(batch):
        assert batch.is_oasiskv_paired
        assert torch.equal(batch.positions[::2], batch.seq_lens)
        assert torch.equal(batch.seq_lens_cpu, batch.seq_lens)
        assert batch.oasiskv_normal_rows.tolist() == list(
            range(0, batch.input_ids.numel(), 2)
        )
        events.append("stage-graph" if graph else "plan-eager")
        if state.fail == "plan":
            raise RuntimeError("failed plan")

    def forward(*, forward_batch, **kwargs):
        assert events[-1] in ("stage-graph", "plan-eager")
        events.append("target")
        if state.fail == "target":
            raise RuntimeError("failed target")
        ids = forward_batch.input_ids
        # Only roots produce output logits; all rows still relay target features.
        logits = torch.zeros(forward_batch.batch_size, 128)
        logits.scatter_(1, (ids[::2] + 1)[:, None], 10)
        output = LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=ids[:, None].float()
        )
        if v4:
            forward_batch._oasiskv_pending_prefetch = {0: (coordinator, {})}
        return NS(logits_output=output, can_run_cuda_graph=graph)

    def sample(logits, batch):
        events.append("sample")
        if state.fail == "sample":
            raise RuntimeError("failed sample")
        return logits.next_token_logits.argmax(-1)

    def inject(**kw):
        # Exercise the production executor's per-request commit mask contract.
        locs = kw["cache_loc_2d"]
        features = kw["target_hidden"].view(*locs.shape, -1)
        for row, count in enumerate(kw["commit_lens"].tolist()):
            for col in range(count):
                committed_features[int(locs[row, col])] = features[row, col].clone()
        events.append("inject")

    coordinator = None
    if v4:
        coordinator = NS(
            is_dsv4_hisparse=True,
            prepare_dspark_verify_window=Mock(
                side_effect=lambda **kw: events.append("prepare-c4") or object()
            ),
            commit_dspark_verify_window=Mock(
                side_effect=lambda *a, **kw: events.append("commit-c4")
            ),
            rollback_dspark_verify_window=Mock(
                side_effect=lambda *a: events.append("rollback-c4")
            ),
            submit_oasiskv_prefetch=Mock(
                side_effect=lambda **kw: events.append("prefetch")
            ),
        )
    runner = NS(
        device="cpu",
        oasiskv_commit_buffers=OasisKVCommitBuffers(8, "cpu"),
        model=object(),
        hisparse_coordinator=coordinator,
        attn_backend=NS(init_forward_metadata=plan),
        decode_cuda_graph_runner=(
            NS(can_run_graph=lambda batch: True, load_batch=plan) if graph else None
        ),
        token_to_kv_pool=NS(clear_unaccepted_c128_draft_states=Mock()),
        sample=sample,
    )
    target = NS(model_runner=runner, forward_batch_generation=forward)
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.model_runner = runner
    worker._target_worker = target
    worker.device = "cpu"
    worker.verify_num_draft_tokens = 2
    worker._block_pos_offsets = torch.arange(2)
    worker._draft_context = nullcontext
    worker._need_mamba_verify_commit = False
    worker._verify_executor = TargetVerifyExecutor(
        target_worker=target,
        gamma=1,
        verify_num_draft_tokens=2,
        model_runner=runner,
        kv_injector=NS(inject_target_hidden=inject),
    )
    worker._verify_executor.accept_and_finalize = Mock(
        side_effect=AssertionError("OasisKV must not accept draft tokens")
    )

    def propose(**kw):
        events.append("draft")
        roots = kw["draft_input"].bonus_tokens
        return NS(
            draft_block=NS(
                draft_tokens=(roots + (1 if candidate_matches else 7))[:, None]
            )
        )

    worker._proposer = NS(propose=propose)
    return worker, events, committed_features, state


def _batch(slots=(7, 511, 1023)):
    bs = len(slots)
    lengths = torch.arange(bs) + 31
    roots = torch.arange(bs) * 10 + 20
    batch = NS(
        forward_mode=ForwardMode.DECODE,
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        seq_lens_sum=int(lengths.sum()),
        req_pool_indices=torch.tensor(slots),
        req_pool_indices_cpu=torch.tensor(slots),
        sampling_info=object(),
        has_grammar=False,
        return_logprob=False,
        can_run_dp_cuda_graph=True,
    )
    return batch, make_next_draft_input(bonus_tokens=roots, new_seq_lens=lengths)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("v4", [False, True])
@pytest.mark.parametrize("candidate_matches", [False, True])
def test_proposals_only_prefetch_and_never_commit(
    monkeypatch, graph, v4, candidate_matches
):
    worker, events, features, _ = _worker(
        monkeypatch, graph=graph, v4=v4, candidate_matches=candidate_matches
    )
    # Include non-power-of-two batches, changed ordering and recycled slots.
    for slots in [(7, 511, 1023), (1023,), (511, 7)]:
        batch, draft_input = _batch(slots)
        prefix = batch.seq_lens.clone()
        published = []
        result = worker._forward_oasiskv_decode(
            batch, draft_input, published.append, None
        )
        roots = draft_input.bonus_tokens
        assert result.accept_lens.tolist() == [1] * len(slots)
        assert result.speculative_num_draft_tokens == 2
        assert result.next_token_ids.view(-1, 2)[:, 0].tolist() == (roots + 1).tolist()
        assert result.next_token_ids.view(-1, 2)[:, 1].tolist() == [-1] * len(slots)
        assert torch.equal(result.new_seq_lens, prefix + 1)
        assert torch.equal(result.next_draft_input.bonus_tokens, roots + 1)
        assert torch.equal(published[0], prefix + 1)
        assert result.can_run_cuda_graph == graph
        assert result.logits_output.hidden_states is None
        assert result.extra_keep_alive_refs[0].is_oasiskv_paired
        for row, slot in enumerate(slots):
            root_loc = slot * 1000 + int(prefix[row])
            assert features[root_loc].item() == roots[row].item()
            assert root_loc + 1 not in features
        worker._verify_executor.accept_and_finalize.assert_not_called()
        if v4:
            assert events.index("commit-c4") < events.index("prefetch")
            coordinator = worker.model_runner.hisparse_coordinator
            commit_kwargs = coordinator.commit_dspark_verify_window.call_args.kwargs
            assert commit_kwargs["commit_lens_cpu"] == [1] * len(slots)
            coordinator.rollback_dspark_verify_window.assert_not_called()
        events.clear()
        features.clear()


@pytest.mark.parametrize("fail", ["plan", "target", "sample"])
def test_v4_failed_forward_rolls_back_without_publishing(monkeypatch, fail):
    worker, _, features, _ = _worker(monkeypatch, v4=True, fail=fail)
    batch, draft_input = _batch()
    published = Mock()
    original_cpu_lengths = batch.seq_lens_cpu
    with pytest.raises(RuntimeError, match="failed"):
        worker._forward_oasiskv_decode(batch, draft_input, published, None)
    coordinator = worker.model_runner.hisparse_coordinator
    coordinator.rollback_dspark_verify_window.assert_called_once()
    coordinator.commit_dspark_verify_window.assert_not_called()
    coordinator.submit_oasiskv_prefetch.assert_not_called()
    assert features == {}
    assert batch.seq_lens_cpu is original_cpu_lengths
    published.assert_not_called()


def test_root_sampling_waits_for_grammar_and_uses_compact_logprobs(monkeypatch):
    worker, events, _, _ = _worker(monkeypatch)
    batch, draft_input = _batch()
    batch.has_grammar = batch.return_logprob = True
    logprobs = Mock()
    monkeypatch.setattr(
        "sglang.srt.speculative.oasiskv_lookahead.compute_oasiskv_logprobs", logprobs
    )
    worker._forward_oasiskv_decode(
        batch, draft_input, None, lambda: events.append("grammar")
    )
    assert events.index("grammar") < events.index("sample")
    assert logprobs.call_args.args[1].next_token_logits.shape[0] == 3
    assert logprobs.call_args.args[2].tolist() == [21, 31, 41]
