"""Independent side graphs and per-layer ordering for Lagom's CSA forecasts."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.attention.dsv4.lagom import resolve_lagom_topk
from sglang.srt.managers.lagom_graph import LagomPrefetchGraphs


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


def _replay():
    return _runner_method(
        "decode_cuda_graph_runner.py",
        "DecodeCudaGraphRunner",
        "_replay_lagom_prefetch",
    )


def test_target_autotune_registers_split_plans_with_inactive_dummy_requests():
    """Exercise BaseRunner warmup through the real CSA routing helper."""
    batch, decode_width, prediction_width, topk = 2, 3, 2, 4
    total = decode_width + prediction_width
    physical = torch.full((batch * total, topk), 33, dtype=torch.int32)
    coordinator = SimpleNamespace(
        swap_in_lagom_pages=Mock(return_value=physical),
        num_real_reqs=torch.ones(1, dtype=torch.int32),
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
        hisparse_coordinator=coordinator,
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
            ForwardBatch=lambda **kw: SimpleNamespace(**kw),
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
    kwargs = coordinator.swap_in_lagom_pages.call_args.kwargs
    assert kwargs["capturing"]
    assert kwargs["source_prefix_lens"].tolist() == [0, 0]
    assert coordinator.num_real_reqs.item() == 0


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


@pytest.mark.parametrize("live_size", [0, 1, 3])
def test_replay_launches_captured_bucket_without_cpu_metadata(live_size):
    pipeline = Mock()
    runner = SimpleNamespace(
        bs=4,
        model_runner=SimpleNamespace(
            hisparse_coordinator=SimpleNamespace(_lagom_graphs=pipeline)
        ),
    )
    _replay()(runner, SimpleNamespace(lagom_decode_width=3, batch_size=live_size))
    pipeline.replay.assert_called_once_with(4)
    pipeline.wait.assert_not_called()
    pipeline.synchronize.assert_not_called()


def test_non_lagom_graph_does_not_launch_prefetch():
    _replay()(SimpleNamespace(), SimpleNamespace())


@pytest.mark.parametrize("fail_target", [False, True])
def test_runner_launches_side_graph_immediately_after_asynchronous_target(fail_target):
    calls = []
    error = RuntimeError("target graph launch failed")
    pipeline = SimpleNamespace(
        check_healthy=lambda: calls.append("health"),
        replay=lambda _bs: calls.append("side"),
        fail=Mock(),
    )

    def target(*_args):
        calls.append("target")
        if fail_target:
            raise error
        return "logits"

    runner = SimpleNamespace(
        bs=4,
        model_runner=SimpleNamespace(
            device_timer=None,
            hisparse_coordinator=SimpleNamespace(_lagom_graphs=pipeline),
        ),
        backend=SimpleNamespace(replay_session=nullcontext, replay=target),
        _resolve_shared_read_ends=lambda *_a: None,
        _replay_attn_backend=lambda: None,
        load_batch=lambda *_a: None,
        _prepare_oasiskv_graph_replay=lambda *_a: None,
        _prepare_ema_graph_replay=lambda: None,
        _replay_graph_key=4,
        _publish_oasiskv_graph_prefetch=lambda *_a: calls.append("other_publish"),
        _submit_ema_graph_prefetch=lambda *_a: None,
        _submit_previous_graph_prefetch=lambda *_a: None,
        _slice_replay_output=lambda output, _batch: output,
    )
    runner._replay_lagom_prefetch = lambda batch: _replay()(runner, batch)
    execute = _runner_method(
        "decode_cuda_graph_runner.py",
        "DecodeCudaGraphRunner",
        "execute",
        dict(
            device_timer_ctx=lambda *_a: nullcontext(),
            envs=SimpleNamespace(
                SGLANG_LOG_DECODE_GRAPH_KEY=SimpleNamespace(get=lambda: False)
            ),
            SharedReadEnds=SimpleNamespace(PRE_REPLAY=1, IN_REPLAY=2, POST_REPLAY=3),
        ),
    )
    batch = SimpleNamespace(
        lagom_decode_width=3, forward_mode=SimpleNamespace(name="TARGET_VERIFY")
    )
    if fail_target:
        with pytest.raises(RuntimeError, match="target graph"):
            execute(runner, batch)
        assert calls == ["health", "target"]
        pipeline.fail.assert_called_once_with(error)
    else:
        assert execute(runner, batch) == "logits"
        assert calls == ["health", "target", "side", "other_publish"]


class _FakeCuda:
    def __init__(self):
        self.calls = []
        self.main = self.Stream("main", self)
        self.current = self.main
        self.capturing = False
        self.event_count = 0

    class Stream:
        def __init__(self, name, cuda):
            self.name, self.cuda = name, cuda

        def wait_stream(self, other):
            self.cuda.calls.append(("wait_stream", self.name, other.name))

        def synchronize(self):
            self.cuda.calls.append(("stream_sync", self.name))

    def current_stream(self):
        return self.current

    def is_current_stream_capturing(self):
        return self.capturing

    def Event(self, *, external):
        assert external
        name = self.event_count
        self.event_count += 1
        return SimpleNamespace(
            record=lambda stream=None: self.calls.append(
                ("record", name, (stream or self.current).name)
            ),
            wait=lambda stream=None: self.calls.append(
                ("wait", name, (stream or self.current).name)
            ),
            synchronize=lambda: self.calls.append(("event_sync", name)),
        )

    def stream(self, stream):
        cuda = self

        class Context:
            def __enter__(self):
                self.previous = cuda.current
                cuda.current = stream

            def __exit__(self, *_exc):
                cuda.current = self.previous

        return Context()

    def CUDAGraph(self):
        return SimpleNamespace(
            replay=lambda: self.calls.append(("replay", self.current.name))
        )

    def graph(self, graph, *, stream):
        return self.stream(stream)


@pytest.fixture
def fake_pipeline(monkeypatch):
    import sglang.srt.managers.lagom_graph as module

    cuda = _FakeCuda()
    monkeypatch.setattr(module, "torch", SimpleNamespace(cuda=cuda))
    pipeline = LagomPrefetchGraphs(cuda.Stream("side", cuda))
    return cuda, pipeline


def test_eager_launch_forks_at_plan_readiness_without_target_tail_wait(fake_pipeline):
    cuda, pipeline = fake_pipeline
    pipeline.consume_layer(0)
    cuda.calls.clear()
    pipeline.plan_ready(
        0,
        2,
        lambda: cuda.calls.append(("copy_future", cuda.current.name)),
        capturing=False,
    )
    cuda.calls.append(("copy_demand", cuda.current.name))
    assert cuda.calls == [
        ("record", 0, "main"),
        ("wait", 0, "side"),
        ("copy_future", "side"),
        ("record", 1, "side"),
        ("copy_demand", "main"),
    ]
    assert pipeline.pending
    pipeline.consume_layer(0)
    assert cuda.calls[-1] == ("wait", 1, "main")
    pipeline.synchronize()
    assert cuda.calls[-1] == ("event_sync", 1)
    assert not pipeline.pending


def test_graph_transfer_waits_each_layer_and_replays_without_foreground_join(
    fake_pipeline,
):
    cuda, pipeline = fake_pipeline
    for layer in (0, 2):
        pipeline.consume_layer(layer)
        pipeline.plan_ready(
            layer,
            4,
            lambda layer=layer: cuda.calls.append(
                ("copy_future", layer, cuda.current.name)
            ),
            capturing=True,
        )
    assert not any(call[0] == "copy_future" for call in cuda.calls)
    pipeline.capture(4)
    assert pipeline.has_shape(4)
    # The captured copy sequence has independent layer-ready dependencies.
    assert cuda.calls[-6:] == [
        ("wait", 0, "side"),
        ("copy_future", 0, "side"),
        ("record", 1, "side"),
        ("wait", 2, "side"),
        ("copy_future", 2, "side"),
        ("record", 3, "side"),
    ]
    cuda.calls.clear()
    pipeline.replay(4)
    assert cuda.calls == [("replay", "side")]
    pipeline.synchronize()
    assert cuda.calls[-2:] == [("event_sync", 1), ("event_sync", 3)]


@pytest.mark.parametrize("graph", [False, True])
def test_failed_transfer_remains_drainable_but_cache_cannot_be_reused(
    fake_pipeline, graph
):
    cuda, pipeline = fake_pipeline
    pipeline.consume_layer(0)
    error = RuntimeError("injected launch failure")

    def fail():
        raise error

    if graph:
        pipeline.graphs[2] = (SimpleNamespace(replay=fail), (0,))
        launch = lambda: pipeline.replay(2)
    else:
        launch = lambda: pipeline.plan_ready(0, 2, fail, capturing=False)
    with pytest.raises(RuntimeError, match="injected"):
        launch()
    assert pipeline.pending
    assert cuda.calls[-1] == ("record", 1, "side")
    for operation in (
        lambda: pipeline.consume_layer(0),
        lambda: pipeline.plan_ready(0, 2, lambda: None, capturing=False),
        lambda: pipeline.capture(2),
        lambda: pipeline.replay(2),
    ):
        with pytest.raises(RuntimeError, match="rebuild") as exc:
            operation()
        assert exc.value.__cause__ is error
    pipeline.synchronize()
    assert cuda.calls[-1] == ("event_sync", 1)
    assert not pipeline.pending


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph runtime")
def test_independent_transfer_graph_consumes_live_plans_across_replays():
    """Real external-event edges join independent target/side graph launches."""
    pipeline = LagomPrefetchGraphs(torch.cuda.Stream())
    inputs = torch.zeros((2, 3), dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    plan = torch.empty_like(inputs)
    plan_count = torch.empty_like(count)
    current = torch.empty_like(inputs)
    future = torch.empty_like(inputs)
    rows = torch.arange(2, device="cuda")[:, None]

    def transfer():
        future.copy_(torch.where(rows < plan_count, plan + 10, -1))

    def target():
        pipeline.consume_layer(0)
        plan.copy_(inputs)
        plan_count.copy_(count)
        pipeline.plan_ready(0, 2, transfer, capturing=True)
        current.copy_(inputs + 1)

    target()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target()
    pipeline.capture(2)
    results = []
    for value, live_count in ((5, 2), (9, 1), (12, 0)):
        inputs.fill_(value)
        count.fill_(live_count)
        graph.replay()
        pipeline.replay(2)
        pipeline.wait()
        results.append((current.clone(), future.clone()))
    pipeline.synchronize()
    torch.cuda.synchronize()
    for (value, live_count), (real, prefetched) in zip(
        ((5, 2), (9, 1), (12, 0)), results
    ):
        assert torch.all(real == value + 1)
        assert torch.all(prefetched[:live_count] == value + 10)
        assert torch.all(prefetched[live_count:] == -1)
