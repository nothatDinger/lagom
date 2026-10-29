"""CPU coverage of the decoding/prediction boundary, without loading a model."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.speculative.lagom import (
    decode_proposal,
    decode_window,
    select_decode_rows,
)


def _proposal(bs=2, width=4):
    return SimpleNamespace(
        draft_block_ids=torch.arange(bs * width).view(bs, width),
        draft_block=SimpleNamespace(
            draft_tokens=torch.arange(bs * width).view(bs, width) + 10,
            corrected_logits=torch.arange(bs * width * 3).view(bs, width, 3),
            greedy_mask=torch.tensor([True, False])[:bs],
            temperatures=torch.ones(bs),
        ),
        draft_hidden=torch.arange(bs * width * 6).view(bs, width, 6),
        confidence=None,
        confidence_tap=torch.arange(bs * width * 3).view(bs * width, 3),
        folded=True,
    )


@pytest.mark.parametrize("prediction_width", [1, 2, 5])
def test_proposal_suffix_cannot_enter_acceptance_or_confidence(prediction_width):
    w = 2
    proposal = _proposal(width=w + prediction_width)
    decoded = decode_proposal(proposal, w)
    assert decoded.draft_block.draft_tokens.shape == (2, w)
    torch.testing.assert_close(
        decoded.draft_block.draft_tokens, proposal.draft_block.draft_tokens[:, :w]
    )
    torch.testing.assert_close(
        decoded.draft_block.corrected_logits,
        proposal.draft_block.corrected_logits[:, :w],
    )
    torch.testing.assert_close(decoded.draft_hidden, proposal.draft_hidden[:, :w])
    torch.testing.assert_close(
        decoded.confidence_tap.view(2, w, 3),
        proposal.confidence_tap.view(2, w + prediction_width, 3)[:, :w],
    )
    assert decoded.folded == proposal.folded
    # Slicing has not shortened the full proposal used by target execution.
    assert proposal.draft_block.draft_tokens.shape[1] == w + prediction_width


def test_commit_window_and_hidden_rows_use_identical_request_major_prefix():
    locations = torch.tensor([[40, 41, 42, 43, 44], [80, 81, 82, 83, 84]])
    full = SimpleNamespace(
        positions_2d=locations + 100,
        verify_cache_loc=locations.reshape(-1),
        verify_cache_loc_2d=locations,
    )
    decoded = decode_window(full, 3)
    hidden = select_decode_rows(
        locations.reshape(-1, 1), batch_size=2, execution_width=5, decode_width=3
    )
    assert decoded.verify_cache_loc.tolist() == [40, 41, 42, 80, 81, 82]
    torch.testing.assert_close(decoded.verify_cache_loc, hidden.flatten())
    torch.testing.assert_close(decoded.positions_2d, locations[:, :3] + 100)


def test_bad_row_geometry_fails_before_acceptance():
    with pytest.raises(ValueError, match="request-major"):
        select_decode_rows(
            torch.zeros(9, 2), batch_size=2, execution_width=5, decode_width=3
        )
    with pytest.raises(ValueError, match="within"):
        decode_proposal(_proposal(), 5)


def _load_method(filename, class_name, method_name, namespace):
    """Load production orchestration without CUDA/model import side effects."""
    root = Path(__file__).resolve().parents[4]
    path = root / "python/sglang/srt/speculative/dspark_components" / filename
    module = ast.parse(path.read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    ast_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    # Only compile the named method from this repository, never external input.
    from sglang.srt.observability.lagom_perf import timed, verify_step

    namespace.update(timed=timed, verify_step=verify_step)
    exec(compile(ast.fix_missing_locations(ast_module), str(path), "exec"), namespace)  # noqa: S102
    return namespace[method_name]


@pytest.mark.parametrize("sampling", [False, True])
def test_target_executes_all_rows_but_returns_only_decode_logits_and_hidden(sampling):
    seen = {}
    adjustment = Mock()
    forward = _load_method(
        "dspark_verify.py",
        "TargetVerifyExecutor",
        "run_non_compact",
        {
            "DFlashVerifyInput": lambda **kwargs: SimpleNamespace(**kwargs),
            "CaptureHiddenMode": SimpleNamespace(FULL=1),
            "select_decode_rows": select_decode_rows,
            "apply_dflash_verify_logits_adjustments": adjustment,
        },
    )
    expected = torch.arange(20).reshape(10, 2)

    def target(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=expected.clone(), hidden_states=expected.clone() + 100
            ),
            can_run_cuda_graph=True,
        )

    executor = SimpleNamespace(
        verify_num_draft_tokens=3,
        _verify_backend_self_adds_seq_lens=lambda: True,
        _forward_prepared_verify=target,
    )
    batch = SimpleNamespace(seq_lens_cpu=torch.tensor([7, 19]), seq_lens_sum=26)
    result = forward(
        executor,
        batch=batch,
        draft_input=None,
        verify_ids_2d=torch.arange(10).view(2, 5),
        verify_window=SimpleNamespace(
            positions_2d=torch.arange(10).view(2, 5), verify_cache_loc=torch.arange(10)
        ),
        sampling_info=object() if sampling else None,
        lagom_decode_width=3,
    )
    assert seen["verify_input"].draft_token_num == 5
    assert seen["verify_input"].draft_token.numel() == 10
    assert seen["lagom_decode_width"] == 3
    rows = torch.tensor([0, 1, 2, 5, 6, 7])
    torch.testing.assert_close(result.logits_output.next_token_logits, expected[rows])
    torch.testing.assert_close(result.logits_output.hidden_states, expected[rows] + 100)
    assert result.can_run_cuda_graph
    if sampling:
        assert adjustment.call_args.kwargs["draft_token_num"] == 3
        assert adjustment.call_args.kwargs["next_token_logits"].shape == (6, 2)
    else:
        adjustment.assert_not_called()


@pytest.mark.parametrize("fail_forward", [False, True])
@pytest.mark.parametrize("folded_confidence", [False, True])
def test_worker_commits_only_decode_prefix_with_prefetch_already_in_verify(
    monkeypatch, fail_forward, folded_confidence
):
    events = []
    proposal = _proposal()
    if folded_confidence:
        proposal.confidence_tap = None
    window = SimpleNamespace(
        positions_2d=torch.arange(10).view(2, 5),
        verify_cache_loc=torch.arange(10),
        verify_cache_loc_2d=torch.arange(10).view(2, 5),
    )
    input_type = type("DraftInput", (), {})
    batch = SimpleNamespace(
        spec_info=input_type(),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([100, 200]),
        seq_lens_cpu=torch.tensor([100, 200]),
        sampling_info=None,
        req_pool_indices=torch.tensor([0, 1]),
        req_pool_indices_cpu=torch.tensor([0, 1]),
        global_num_tokens=None,
        has_grammar=False,
        return_logprob=False,
        reqs=[],
        forward_iter=0,
        spec_verify_tier_num_tokens=0,
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda *_: SimpleNamespace(current_stream=lambda: None),
    )

    def verify(**kwargs):
        assert kwargs["verify_ids_2d"].shape == (2, 5)
        assert kwargs["lagom_decode_width"] == 3
        events.append("forward")
        events.append("prefetch")
        if fail_forward:
            raise RuntimeError("target failed")
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=torch.zeros(6, 3), hidden_states=torch.zeros(6, 3)
            ),
            can_run_cuda_graph=True,
        )

    def accept(**kwargs):
        assert kwargs["verify_ids_2d"].shape == (2, 3)
        assert kwargs["draft_block"].draft_tokens.shape == (2, 2)
        assert kwargs["draft_block"].corrected_logits.shape == (2, 2, 3)
        events.append("accept")
        return SimpleNamespace(
            commit_lens=torch.tensor([1, 3]),
            new_seq_lens=torch.tensor([101, 203]),
            out_tokens=torch.zeros((2, 3), dtype=torch.int64),
            bonus=torch.tensor([2, 3]),
            correct_len=torch.tensor([0, 2]),
            cap_trim_lens=torch.zeros(2, dtype=torch.int64),
        )

    def inject(**kwargs):
        assert kwargs["verify_window"].verify_cache_loc.tolist() == [0, 1, 2, 5, 6, 7]
        assert kwargs["logits_output"].hidden_states.shape[0] == 6
        events.append("inject")

    def clear(*args):
        assert args[-1] == 5
        assert args[-2].tolist() == ([0, 0] if fail_forward else [1, 3])
        events.append("clear")

    coordinator = SimpleNamespace(
        is_dsv4_hisparse=True,
        prepare_dspark_verify_window=Mock(return_value=object()),
        commit_dspark_verify_window=lambda *args: events.append("commit"),
        rollback_dspark_verify_window=lambda *args: events.append("rollback"),
    )
    runner = SimpleNamespace(
        model=object(),
        hisparse_coordinator=coordinator,
        token_to_kv_pool=SimpleNamespace(clear_unaccepted_c128_draft_states=clear),
    )

    def confidence(**kwargs):
        if folded_confidence:
            expected = proposal.draft_hidden[:, :2].reshape(4, 2, 3).sum(dim=1)
            torch.testing.assert_close(kwargs["confidence_tap"], expected)

    worker = SimpleNamespace(
        _lagom=True,
        gamma=2,
        verify_num_draft_tokens=3,
        _execution_width=5,
        _execution_pos_offsets=torch.arange(5),
        device="cpu",
        _draft_is_moe=False,
        target_worker=SimpleNamespace(model_runner=runner),
        model_runner=runner,
        draft_model=SimpleNamespace(hc_mult=2, collapse_hc_head=lambda x: x.sum(dim=1)),
        _observers=SimpleNamespace(
            begin_step=lambda: None,
            segment=lambda *_: nullcontext(),
            observe_verify_step=lambda **_: None,
        ),
        _draft_context=lambda: nullcontext(),
        _proposer=SimpleNamespace(propose=lambda **_: proposal),
        _verify_planner=SimpleNamespace(
            carries_confidence=True,
            compute_confidence_tensor=confidence,
            resolve_verify_token_budget=lambda **_: None,
            schedule_layout=lambda **_: None,
            should_run_compact=lambda **_: False,
        ),
        _dp_verify_tier_num_tokens=lambda *_: None,
        _simulate_acc_len=0,
        _verify_executor=SimpleNamespace(
            verify_epilogue=None,
            run_non_compact=verify,
            accept_and_finalize=accept,
            commit_hidden=inject,
        ),
        _commit_target_mamba_states_after_verify=lambda **_: None,
    )
    forward = _load_method(
        "dspark_worker_v2.py",
        "DSparkWorkerV2",
        "_forward_decode",
        {
            "torch": torch,
            "DFlashDraftInputV2": input_type,
            "alloc_verify_window": lambda **_: window,
            "decode_window": decode_window,
            "decode_proposal": decode_proposal,
            "InfoSegment": SimpleNamespace(DRAFT=0, TARGET_VERIFY=1),
            "prepare_mamba_track_for_verify": lambda *_: None,
            "make_next_draft_input": lambda **kwargs: SimpleNamespace(**kwargs),
            "GenerationBatchResult": SimpleNamespace,
        },
    )
    if fail_forward:
        with pytest.raises(RuntimeError, match="target failed"):
            forward(worker, batch, None)
        assert events == ["forward", "prefetch", "rollback", "clear"]
    else:
        result = forward(worker, batch, None)
        assert result.speculative_num_draft_tokens == 3
        assert result.accept_lens.tolist() == [1, 3]
        assert events == [
            "forward",
            "prefetch",
            "accept",
            "commit",
            "clear",
            "inject",
        ]
    assert (
        coordinator.prepare_dspark_verify_window.call_args.kwargs["verify_width"] == 5
    )
