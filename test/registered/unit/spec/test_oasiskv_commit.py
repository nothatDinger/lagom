"""Root-only commit metadata must not introduce a decode host barrier."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.environ import envs
from sglang.srt.speculative.oasiskv_lookahead import (
    OasisKVCommitBuffers,
    build_oasiskv_commit,
    build_oasiskv_pair_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_commits_reuse_constants_across_batch_changes_and_exclude_probes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    buffers = OasisKVCommitBuffers(8, device)
    retained = []
    for bs in (7, 1, 4, 0, 8, 2, 7):
        normal, _ = build_oasiskv_pair_rows(bs, device)
        lengths, indices = build_oasiskv_commit(normal, bs, device, buffers=buffers)
        # Model outputs are deliberately different on roots and probes. Reused
        # commit metadata must select exactly the old root-only result, even
        # when the next batch has a different size and slot ordering.
        output = torch.arange(2 * bs, device=device) + 100
        torch.testing.assert_close(output[indices].flatten(), output[::2])
        assert lengths.tolist() == [1] * bs
        assert lengths.dtype == torch.int32
        assert indices.dtype == torch.int64
        if bs:
            assert lengths.data_ptr() == buffers.accept_lens.data_ptr()
            assert indices.data_ptr() == buffers.normal_rows.data_ptr()
        retained.append((lengths, indices, bs))
    for lengths, indices, bs in retained:
        assert lengths.tolist() == [1] * bs
        assert indices.flatten().tolist() == list(range(0, 2 * bs, 2))


def test_production_commit_does_not_allocate_or_read_device_values(monkeypatch):
    # Meta exercises the non-CPU production branch without CUDA. GPU tests
    # below additionally cover stream initialization and actual graph capture.
    buffers = OasisKVCommitBuffers(8, "meta")
    rows, _ = build_oasiskv_pair_rows(3, "meta")
    monkeypatch.setenv("SGLANG_DEBUG_OASISKV_COMMIT", "0")
    with monkeypatch.context() as patch:
        for name in ("equal", "arange", "ones", "_assert_async"):
            patch.setattr(torch, name, Mock(side_effect=AssertionError(name)))
        lengths, indices = build_oasiskv_commit(rows, 3, "meta", buffers=buffers)
    assert lengths.shape == (3,)
    assert indices.shape == (3, 1)


def test_debug_check_uses_async_tensor_condition(monkeypatch):
    buffers = OasisKVCommitBuffers(2, "meta")
    rows, _ = build_oasiskv_pair_rows(2, "meta")
    check = Mock()
    monkeypatch.setenv("SGLANG_DEBUG_OASISKV_COMMIT", "1")
    monkeypatch.setattr(torch, "equal", Mock(side_effect=AssertionError("host bool")))
    monkeypatch.setattr(torch, "_assert_async", check)
    build_oasiskv_commit(rows, 2, "meta", buffers=buffers)
    condition, message = check.call_args.args
    assert condition.shape == () and condition.dtype == torch.bool
    assert "pair roots" in message


@pytest.mark.parametrize(
    "rows,bs,device,message",
    [
        (torch.tensor([0, 1]), 2, "cpu", "pair roots"),
        (torch.tensor([2, 0]), 2, "cpu", "pair roots"),
        (torch.tensor([[0, 2]]), 2, "cpu", "one normal row"),
        (torch.tensor([0, 2]), 1, "cpu", "one normal row"),
        (torch.tensor([0, 2], dtype=torch.int32), 2, "cpu", "int64"),
        (torch.tensor([0, 2]), 2, "meta", "same device"),
    ],
)
def test_commit_rejects_invalid_cpu_geometry(rows, bs, device, message):
    with pytest.raises(ValueError, match=message):
        build_oasiskv_commit(rows, bs, device)


def test_commit_rejects_capacity_and_buffer_device_mismatch():
    rows = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="capacity"):
        build_oasiskv_commit(rows, 2, "cpu", buffers=OasisKVCommitBuffers(1, "cpu"))
    with pytest.raises(ValueError, match="same device"):
        build_oasiskv_commit(rows, 2, "cpu", buffers=OasisKVCommitBuffers(2, "meta"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("debug", [False, True])
def test_cuda_commit_has_no_host_sync_and_replays_with_padding(debug):
    with envs.SGLANG_DEBUG_OASISKV_COMMIT.override(debug):
        # Initialize on a different stream. The consumer must wait for constants
        # through an event, without making the CPU wait for either stream.
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            buffers = OasisKVCommitBuffers(8, "cuda")
        with torch.cuda.stream(consumer):
            rows, _ = build_oasiskv_pair_rows(3, "cuda")
            old_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                lengths, indices = build_oasiskv_commit(
                    rows, 3, "cuda", buffers=buffers
                )
            finally:
                torch.cuda.set_sync_debug_mode(old_mode)
            actual = lengths.clone(), indices.clone()
        consumer.synchronize()
        assert actual[0].tolist() == [1] * 3
        assert actual[1].flatten().tolist() == [0, 2, 4]

        # Padded graph input has 8 request pairs; commit contains only the 3
        # real requests. Capture must not contain an equal()/scalar host read.
        payload = torch.zeros(16, dtype=torch.int64, device="cuda")
        graph = torch.cuda.CUDAGraph()
        torch.cuda.current_stream().wait_stream(consumer)
        with torch.cuda.graph(graph):
            lengths, indices = build_oasiskv_commit(rows, 3, "cuda", buffers=buffers)
            committed = payload[indices].flatten()
            seq_lens = lengths + 100
        for step in range(3):
            payload.copy_(torch.arange(16, device="cuda") + step * 100)
            graph.replay()
            assert committed.tolist() == [step * 100 + i for i in (0, 2, 4)]
            assert seq_lens.tolist() == [101] * 3


def test_draft_extend_plan_waits_for_target_producers(monkeypatch):
    import sglang.srt.speculative.eagle_worker_v2 as module

    events = []
    main_stream = object()
    plan_stream = SimpleNamespace(
        wait_stream=lambda stream: events.append(("wait", stream))
    )
    worker = SimpleNamespace(
        server_args=SimpleNamespace(is_oasiskv_lookahead=True),
        speculative_num_draft_tokens=2,
        device="cpu",
        plan_stream=plan_stream,
        plan_stream_ctx=nullcontext(),
        draft_runner=object(),
        cuda_graph_runner_for_draft_extend=None,
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([10, 20]),
        out_cache_loc=torch.arange(4),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.arange(4).view(4, 1)),
        accept_lens=torch.ones(2, dtype=torch.int32),
        next_token_ids=torch.tensor([5, 7], dtype=torch.int32),
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _: SimpleNamespace(current_stream=lambda: main_stream),
    )

    class Staged(Exception):
        pass

    def prepare(inputs, batch, tokens, *args, **kwargs):
        assert events == [("wait", main_stream)]
        assert inputs.hidden_states.flatten().tolist() == [0, 2]
        assert tokens.tolist() == [5, 7]
        assert batch.out_cache_loc.tolist() == [0, 2]
        raise Staged

    monkeypatch.setattr(module, "prepare_for_draft_extend", prepare)
    with pytest.raises(Staged):
        module.EagleDraftWorker._draft_extend_for_decode(worker, batch, result)
    assert batch.out_cache_loc.tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_extend_stages_delayed_target_outputs_on_plan_stream(monkeypatch):
    import sglang.srt.speculative.eagle_worker_v2 as module

    main, plan = torch.cuda.Stream(), torch.cuda.Stream()
    staged = []
    retained = []
    done = torch.cuda.Event()

    class Staged(Exception):
        pass

    def prepare(inputs, batch, tokens, *args, **kwargs):
        assert torch.cuda.current_stream() == plan
        # Exercise the copies that production graph-input staging performs on
        # the plan stream. Preserve all source owners until the copies finish.
        retained.extend((inputs, tokens, batch.out_cache_loc))
        staged.extend(
            (inputs.hidden_states.clone(), tokens.clone(), batch.out_cache_loc.clone())
        )
        done.record()
        raise Staged

    monkeypatch.setattr(module, "prepare_for_draft_extend", prepare)
    worker = SimpleNamespace(
        server_args=SimpleNamespace(is_oasiskv_lookahead=True),
        speculative_num_draft_tokens=2,
        device="cuda",
        plan_stream=plan,
        plan_stream_ctx=torch.cuda.stream(plan),
        draft_runner=object(),
        cuda_graph_runner_for_draft_extend=None,
    )
    with torch.cuda.stream(main):
        hidden = torch.zeros(4, 1, device="cuda")
        tokens = torch.zeros(2, dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], device="cuda"),
            out_cache_loc=torch.arange(4, device="cuda"),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden),
            accept_lens=torch.ones(2, dtype=torch.int32, device="cuda"),
            next_token_ids=tokens,
        )
    main.synchronize()  # Setup only; no synchronization during the tested step.
    with torch.cuda.stream(main):
        torch.cuda._sleep(20_000_000)
        hidden.fill_(17)
        tokens.fill_(31)
        with pytest.raises(Staged):
            module.EagleDraftWorker._draft_extend_for_decode(worker, batch, result)
    done.synchronize()
    assert staged[0].flatten().tolist() == [17, 17]
    assert staged[1].tolist() == [31, 31]
    assert staged[2].tolist() == [0, 2]
