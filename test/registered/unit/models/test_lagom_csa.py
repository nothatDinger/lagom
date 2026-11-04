from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.dsv4.lagom import (
    lagom_prediction_geometry,
    resolve_lagom_topk,
)


def make_batch(prediction_width=2):
    return SimpleNamespace(
        req_pool_indices=torch.tensor([3, 8]),
        lagom_decode_width=3,
        lagom_prediction_tokens=torch.zeros((2, prediction_width), dtype=torch.int64),
        lagom_prediction_indices=torch.arange(2 * prediction_width * 4).reshape(-1, 4),
        lagom_prediction_ready=None,
        is_lagom_graph_capture=False,
        positions=(torch.tensor([1024, 2048])[:, None] + torch.arange(3)).reshape(-1),
    )


@pytest.mark.parametrize("prediction_width", [1, 2, 4])
def test_lagom_csa_combines_predictions_but_returns_only_real_locations(
    prediction_width,
):
    batch = make_batch(prediction_width)
    logical_ids = torch.arange(6 * 4).reshape(6, 4) + 100
    full_width = 3 + prediction_width
    physical_locs = torch.arange(2 * full_width * 4, dtype=torch.int32).reshape(-1, 4)
    output = torch.empty((6, 4), dtype=torch.int32)
    coordinator = SimpleNamespace(
        compress_ratio=4,
        swap_in_lagom_pages=Mock(return_value=physical_locs),
    )
    result = resolve_lagom_topk(
        coordinator=coordinator,
        forward_batch=batch,
        layer_id=7,
        raw_indices=logical_ids,
        compressed_seq_lens=torch.arange(6),
        output_buffer=output,
    )
    assert result is output
    expected = physical_locs.reshape(2, full_width, 4)[:, :3].reshape(6, 4)
    torch.testing.assert_close(result, expected)
    swap = coordinator.swap_in_lagom_pages.call_args.kwargs
    union = swap["top_k_result"].reshape(2, full_width, 4)
    torch.testing.assert_close(union[:, :3].reshape(6, 4), logical_ids)
    torch.testing.assert_close(
        union[:, 3:].reshape(-1, 4), batch.lagom_prediction_indices
    )
    assert swap["source_prefix_lens"].tolist() == [1024, 2048]
    assert swap["decode_width"] == 3
    lengths = swap["compressed_seq_lens"].reshape(2, full_width)
    assert lengths[:, 3:].tolist() == [
        [256] * prediction_width,
        [512] * prediction_width,
    ]
    assert swap["req_pool_indices"] is batch.req_pool_indices
    assert swap["layer_id"] == 7
    assert not swap["capturing"]


@pytest.mark.parametrize("width,rows,positions", [(5, 6, 6), (3, 5, 6), (3, 6, 5)])
def test_invalid_lagom_geometry_cannot_launch_prefetch(width, rows, positions):
    batch = make_batch()
    batch.lagom_decode_width = width
    batch.positions = torch.arange(positions)
    coordinator = Mock()
    with pytest.raises(ValueError):
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=batch,
            layer_id=0,
            raw_indices=torch.zeros((rows, 4), dtype=torch.int32),
            compressed_seq_lens=torch.zeros(rows),
            output_buffer=torch.empty((rows, 4), dtype=torch.int32),
        )
    coordinator.swap_in_lagom_pages.assert_not_called()


def test_graph_capture_uses_live_device_prefix_without_cpu_mirrors():
    batch = make_batch()
    batch.is_lagom_graph_capture = True
    coordinator = SimpleNamespace(
        compress_ratio=4,
        swap_in_lagom_pages=Mock(return_value=torch.zeros(10, 4, dtype=torch.int32)),
    )
    for fill in (11, 22):
        batch.positions.add_(fill)
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=batch,
            layer_id=1,
            raw_indices=torch.full((6, 4), fill, dtype=torch.int32),
            compressed_seq_lens=torch.zeros(6),
            output_buffer=torch.empty((6, 4), dtype=torch.int32),
        )
    args = coordinator.swap_in_lagom_pages.call_args.kwargs
    assert args["capturing"]
    assert args["source_prefix_lens"].tolist() == [1057, 2081]


@pytest.mark.parametrize("prediction_width", [1, 2, 4])
def test_prediction_queries_use_future_positions_and_only_committed_keys(
    prediction_width,
):
    batch = make_batch(prediction_width)
    pages = torch.tensor([[3, 4]] * 3 + [[8, 9]] * 3)
    for offset in (0, 5):
        batch.positions.add_(offset)
        positions, lengths, prediction_pages = lagom_prediction_geometry(batch, pages)
        prefixes = [1024 + offset, 2048 + offset]
        assert positions.reshape(2, -1).tolist() == [
            list(range(prefix + 3, prefix + 3 + prediction_width))
            for prefix in prefixes
        ]
        assert lengths.reshape(2, -1).tolist() == [
            [prefix // 4] * prediction_width for prefix in prefixes
        ]
        assert prediction_pages.tolist() == (
            [[3, 4]] * prediction_width + [[8, 9]] * prediction_width
        )


def _indexer_method(name):
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/attention/dsv4/indexer.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "C4IndexerBackendMixin"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return path, ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )


def test_prediction_indexer_never_writes_compressor_or_attention_metadata():
    import ast
    from types import MethodType

    source_indices = torch.full((6, 4), 99, dtype=torch.int32)
    core = SimpleNamespace(
        positions=torch.arange(6),
        c4_sparse_page_indices=source_indices,
        page_table=torch.ones(6, 1, dtype=torch.int32),
    )
    prediction = SimpleNamespace(
        c4_seq_lens=torch.tensor([8, 8, 16, 16], dtype=torch.int32),
        page_table=torch.ones(4, 1, dtype=torch.int32),
        deep_gemm_metadata=None,
        max_c4_seq_len=64,
        c4_page_size=64,
    )
    flags = SimpleNamespace(
        **{
            name: SimpleNamespace(get=lambda value=value: value)
            for name, value in {
                "SGLANG_OPT_USE_TILELANG_INDEXER": False,
                "SGLANG_OPT_USE_AITER_INDEXER": False,
                "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": True,
            }.items()
        }
    )
    logits = Mock(return_value=torch.zeros(4, 64))
    capture = Mock(
        side_effect=AssertionError("predictions must not capture real indices")
    )

    def topk(_scores, _lengths, _pages, physical, _page_size, raw):
        physical.fill_(123)
        raw.copy_(torch.arange(16, dtype=torch.int32).reshape(4, 4))

    namespace = dict(
        torch=torch,
        TYPE_CHECKING=False,
        PagedIndexerMetadata=SimpleNamespace,
        envs=flags,
        is_sm120_supported=lambda: False,
        fp8_paged_mqa_logits_torch=logits,
        get_global_indexer_capturer=capture,
        topk_transform_512_pytorch_vectorized=topk,
    )
    methods = {}
    for name in ("_forward_prepare_normal", "forward_c4_indexer"):
        path, module = _indexer_method(name)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        methods[name] = namespace[name]
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(core_metadata=core, indexer_metadata=None),
        token_to_kv_pool=SimpleNamespace(
            get_index_k_with_scale_buffer=lambda **_: torch.zeros(1, 64 * 132)
        ),
        hisparse_coordinator=Mock(),
        _get_nonpaged_indexer_plan=Mock(return_value=None),
        dsa_topk_backend=SimpleNamespace(is_torch=lambda: True),
        forward_indexer_compressor=Mock(
            side_effect=AssertionError("P must not write indexer keys")
        ),
    )
    backend._forward_prepare_normal = MethodType(
        methods["_forward_prepare_normal"], backend
    )
    indexer = SimpleNamespace(
        use_fp4_indexer=False,
        index_topk=4,
        layer_id=1,
        compute_weights=lambda x, **_: torch.ones(x.shape[0], 2),
        compute_q=lambda q, pos, weights: (
            torch.ones(q.shape[0], 2, 128),
            weights.unsqueeze(-1),
        ),
    )
    result = methods["forward_c4_indexer"](
        backend,
        x=torch.ones(4, 16),
        q_lora=torch.ones(4, 8),
        c4_indexer=indexer,
        forward_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_idle=lambda: False,
                is_decode=lambda: False,
                is_target_verify=lambda: True,
            )
        ),
        skip_compressor=True,
        prediction_metadata=prediction,
        prediction_positions=torch.tensor([35, 36, 67, 68]),
    )
    assert result.tolist() == torch.arange(16).reshape(4, 4).tolist()
    assert core.c4_sparse_page_indices is source_indices
    assert (source_indices == 99).all()
    backend.forward_indexer_compressor.assert_not_called()
    backend.hisparse_coordinator.assert_not_called()
    assert backend.hisparse_coordinator.method_calls == []
    capture.assert_not_called()
    torch.testing.assert_close(
        logits.call_args.args[3].flatten(), prediction.c4_seq_lens
    )


def test_each_csa_prediction_reuses_frozen_input_and_joins_before_union(monkeypatch):
    from contextlib import nullcontext

    from sglang.srt.layers.attention.dsv4.lagom import launch_lagom_prediction

    main = object()
    ready = Mock()
    stream = Mock(record_event=Mock(return_value=ready))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    frozen = torch.arange(16).reshape(4, 4)
    snapshot = frozen.clone()
    batch = make_batch()
    prediction = torch.zeros(4, 4, dtype=torch.int32)
    backend = SimpleNamespace(forward_c4_indexer=Mock(return_value=prediction))
    for layer_id in (1, 7):
        attention = SimpleNamespace(
            fuse_wqa_wkv=False,
            _compute_q_a=Mock(return_value=torch.ones(4, 2)),
            indexer=SimpleNamespace(layer_id=layer_id),
        )
        indices, event = launch_lagom_prediction(
            attention=attention,
            hidden_states=frozen,
            positions=torch.arange(4),
            metadata=object(),
            forward_batch=batch,
            attn_backend=backend,
            stream=stream,
        )
        assert backend.forward_c4_indexer.call_args.kwargs["x"] is frozen
        assert backend.forward_c4_indexer.call_args.kwargs["skip_compressor"]
        assert indices is prediction
        batch.lagom_prediction_indices = indices
        batch.lagom_prediction_ready = event
        torch.testing.assert_close(frozen, snapshot)

    def plan(**_kwargs):
        ready.wait.assert_called_once_with(main)
        return torch.zeros(10, 4, dtype=torch.int32)

    resolve_lagom_topk(
        coordinator=SimpleNamespace(compress_ratio=4, swap_in_lagom_pages=plan),
        forward_batch=batch,
        layer_id=7,
        raw_indices=torch.zeros(6, 4, dtype=torch.int32),
        compressed_seq_lens=torch.ones(6, dtype=torch.int32),
        output_buffer=torch.empty(6, 4, dtype=torch.int32),
    )
