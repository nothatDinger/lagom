"""Replay publication and external stream ordering for Lagom's CSA forecasts."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.attention.dsv4.lagom import resolve_lagom_topk


def _runner_method(filename, class_name, method_name, namespace=None):
    # Exercise the production method without importing CUDA-only runner/model
    # dependencies into the CPU test suite.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/runner"
        / filename
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    namespace = {} if namespace is None else namespace
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method.name]


def _publish():
    return _runner_method(
        "decode_cuda_graph_runner.py",
        "DecodeCudaGraphRunner",
        "_publish_lagom_graph_prefetch",
    )


def test_target_autotune_uses_real_demand_window_and_does_not_queue_dummy_requests():
    """Exercise BaseRunner warmup through the real CSA routing helper."""
    batch, decode_width, prediction_width, topk = 2, 3, 2, 4
    total = decode_width + prediction_width
    physical = torch.full((batch * total, topk), 33, dtype=torch.int32)
    coordinator = SimpleNamespace(
        swap_in_lagom_pages=Mock(return_value=physical),
        queue_lagom_prefetch=Mock(),
    )
    seen = {}

    def forward(_ids, _positions, forward_batch, **_kwargs):
        seen["batch"] = forward_batch
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=forward_batch,
            layer_id=1,
            raw_indices=torch.arange(batch * total * topk).reshape(batch * total, topk),
            compressed_seq_lens=torch.ones(batch * total, dtype=torch.int32),
            output_buffer=physical,
        )

    args = SimpleNamespace(
        is_lagom=True,
        speculative_num_draft_tokens=decode_width,
        disaggregation_mode="null",
        enable_lora=False,
        pp_size=1,
    )
    algo = SimpleNamespace(
        is_speculative=lambda: True,
        is_eagle=lambda: False,
        is_standalone=lambda: False,
    )
    mr = SimpleNamespace(
        device="cpu",
        server_args=args,
        is_draft_worker=False,
        is_generation=True,
        spec_algorithm=algo,
        decode_num_tokens_per_req=lambda: total,
        model=SimpleNamespace(forward=forward),
        prepare_dummy_forward_batch=lambda fb: fb,
        attn_backend=SimpleNamespace(init_forward_metadata=Mock()),
        tp_group=SimpleNamespace(barrier=Mock()),
    )
    buffers = SimpleNamespace(
        input_ids=torch.zeros(batch * total, dtype=torch.int64),
        positions=torch.zeros(batch * total, dtype=torch.int64),
        out_cache_loc=torch.zeros(batch * total, dtype=torch.int64),
        mrope_positions=torch.zeros(3, batch * total, dtype=torch.int64),
        num_token_non_padded=torch.zeros(1, dtype=torch.int32),
        req_pool_indices=torch.tensor([1, 2]),
        seq_lens=torch.ones(batch, dtype=torch.int64),
        seq_lens_cpu=torch.ones(batch, dtype=torch.int64),
        next_token_logits_buffer=None,
        encoder_lens=None,
        custom_mask=None,
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
        ngram_embedding_info=None,
    )
    dummy = _runner_method(
        "base_runner.py",
        "BaseRunner",
        "_dummy_run",
        dict(
            torch=torch,
            ForwardMode=SimpleNamespace(DECODE=1, EXTEND=2, TARGET_VERIFY=3),
            CaptureHiddenMode=SimpleNamespace(NULL=0, FULL=1),
            get_server_return_hidden_states_mode=lambda: 0,
            get_flags=lambda: SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=False)
            ),
            require_mlp_tp_gather=lambda _a: False,
            require_attn_tp_gather=lambda _a: False,
            require_gathered_buffer=lambda _a: False,
            create_dummy_verify_input=lambda *_a: SimpleNamespace(
                capture_hidden_mode=1
            ),
            ForwardBatch=lambda **kw: SimpleNamespace(lagom_prefix_lens_cpu=None, **kw),
            DpPaddingMode=SimpleNamespace(
                get_default_mode_in_cuda_graph=lambda: SimpleNamespace(
                    is_max_len=lambda: False
                )
            ),
            set_dp_buffer_len=lambda *_a: None,
            set_is_extend_in_batch=lambda *_a: None,
            forward_context=lambda *_a: nullcontext(),
            ForwardContext=SimpleNamespace,
            empty_context=nullcontext,
        ),
    )
    dummy(SimpleNamespace(model_runner=mr), batch_size=batch, buffers=buffers)
    captured = seen["batch"]
    assert captured.input_ids.numel() == batch * total
    assert (
        coordinator.swap_in_lagom_pages.call_args.kwargs["decode_width"] == decode_width
    )
    assert captured._lagom_pending_prefetch[1][1].shape == (
        batch,
        prediction_width,
        topk,
    )
    coordinator.queue_lagom_prefetch.assert_not_called()


@pytest.mark.parametrize("is_draft,physical_width", [(False, 15), (True, 14)])
def test_dummy_spec_metadata_uses_physical_lagom_width(
    monkeypatch, is_draft, physical_width
):
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/spec_info.py"
    )
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_dummy_verify_input"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = dict(
        get_spec_config=lambda: SimpleNamespace(
            is_lagom=True, speculative_num_draft_tokens=8
        )
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.forward_batch_info",
        SimpleNamespace(CaptureHiddenMode=SimpleNamespace(NULL=0, FULL=1)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.speculative.dflash_info",
        SimpleNamespace(DFlashVerifyInput=SimpleNamespace),
    )
    algo = SimpleNamespace(
        is_eagle=lambda: False,
        is_standalone=lambda: False,
        is_dflash_family=lambda: True,
    )
    result = namespace[method.name](algo, None, physical_width, is_draft)
    assert result.draft_token_num == physical_width


def test_replay_publishes_live_request_order_and_excludes_bucket_padding():
    coordinator = SimpleNamespace(queue_lagom_prefetch=Mock())
    predictions = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    runner = SimpleNamespace(
        _replay_graph_key="bucket4",
        _lagom_graph_prefetch={"bucket4": {7: (coordinator, predictions)}},
    )
    live = SimpleNamespace(
        lagom_decode_width=3,
        batch_size=2,
        req_pool_indices=torch.tensor([11, 4]),
        req_pool_indices_cpu=torch.tensor([11, 4]),
        lagom_prefix_lens_cpu=torch.tensor([4096, 2048]),
    )
    publish = _publish()
    publish(runner, live)
    call = coordinator.queue_lagom_prefetch.call_args.kwargs
    assert call["layer_id"] == 7
    assert call["req_pool_indices"].tolist() == [11, 4]
    assert call["source_committed_lens_cpu"].tolist() == [4096, 2048]
    torch.testing.assert_close(call["predicted_c4_entries"], predictions[:2])

    # The graph tensor is reused, but the next descriptor must take live
    # identities from the new batch rather than capture-time or previous IDs.
    live.req_pool_indices = torch.tensor([8])
    live.req_pool_indices_cpu = torch.tensor([8])
    live.lagom_prefix_lens_cpu = torch.tensor([8192])
    live.batch_size = 1
    predictions.add_(100)
    publish(runner, live)
    call = coordinator.queue_lagom_prefetch.call_args.kwargs
    assert call["req_pool_indices"].tolist() == [8]
    assert call["predicted_c4_entries"].shape == (1, 2, 3)
    torch.testing.assert_close(call["predicted_c4_entries"], predictions[:1])


def test_idle_graph_does_not_queue_or_require_a_prediction_template():
    runner = SimpleNamespace(_lagom_graph_prefetch={})
    _publish()(runner, SimpleNamespace(lagom_decode_width=3, batch_size=0))


def test_active_lagom_graph_fails_if_predictions_were_not_captured():
    runner = SimpleNamespace(_replay_graph_key=2, _lagom_graph_prefetch={})
    with pytest.raises(RuntimeError, match="did not capture"):
        _publish()(runner, SimpleNamespace(lagom_decode_width=3, batch_size=2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph runtime")
def test_external_event_wait_and_prediction_tensor_survive_multiple_replays():
    """Run the actual capture helper with an externally recorded layer event."""
    device = "cuda"
    producer = torch.cuda.Stream()
    ready = torch.cuda.Event(external=True)
    ready.record()
    inputs = torch.zeros((2 * 5, 3), dtype=torch.int32, device=device)
    output = torch.empty_like(inputs)
    resident = torch.zeros((), dtype=torch.int32, device=device)
    predictions = []
    resolved = []

    def swap(**kwargs):
        ready.wait(torch.cuda.current_stream())
        kwargs["output_buffer"].copy_(kwargs["top_k_result"] + resident)
        return kwargs["output_buffer"]

    def queue(**kwargs):
        predictions.append(kwargs["predicted_c4_entries"].clone())

    coordinator = SimpleNamespace(swap_in_lagom_pages=swap, queue_lagom_prefetch=queue)
    capture_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1, 2], device=device),
        lagom_decode_width=3,
        lagom_prefix_lens_cpu=None,
        is_lagom_graph_capture=True,
    )
    lengths = torch.full((10,), 1024, device=device)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        raw = inputs + 1
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=capture_batch,
            layer_id=0,
            raw_indices=raw,
            compressed_seq_lens=lengths,
            output_buffer=output,
        )
    runner = SimpleNamespace(
        _replay_graph_key=2,
        _lagom_graph_prefetch={2: capture_batch._lagom_pending_prefetch},
    )
    live = SimpleNamespace(
        lagom_decode_width=3,
        batch_size=1,
        req_pool_indices=capture_batch.req_pool_indices[:1],
        req_pool_indices_cpu=torch.tensor([1]),
        lagom_prefix_lens_cpu=torch.tensor([4096]),
    )
    for value in (5, 9):
        inputs.fill_(value)
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            resident.fill_(value)
            ready.record(producer)
        graph.replay()
        _publish()(runner, live)
        resolved.append(output.clone())
    torch.cuda.synchronize()
    assert len(predictions) == 2
    assert predictions[0].shape == (1, 2, 3)
    assert torch.all(predictions[0] == 6)
    assert torch.all(predictions[1] == 10)
    assert torch.all(resolved[0] == 11)
    assert torch.all(resolved[1] == 19)
