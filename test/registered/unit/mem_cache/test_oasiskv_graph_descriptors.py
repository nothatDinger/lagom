"""Stable replay skips per-layer Python work while retaining ownership fences."""

import os
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_graph import OasisKVGraphState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def populated(layers=3):
    cache = OasisKVCache(
        OasisKVConfig(token_budget=8, block_size=4), "cpu", max_requests=3
    )
    graph = OasisKVGraphState(cache, 3, 512, layers)
    key = torch.zeros(33, 1, 8)
    for layer in range(layers):
        graph.get_layer(layer, key, key, key)
        for slot in (7, 101):
            state = cache.get(slot, 0, layer, key, key)
            state.append(key, key, 0)
            state._grow(128)
    return cache, graph, key


def test_stable_reorder_and_padding_do_not_rebuild_layer_descriptors(monkeypatch):
    cache, graph, _ = populated()
    graph.prepare([7, 101], [0, 0], [33, 33], paired=True)
    before = graph.tables.clone()
    descriptors = graph.inputs.descriptors.clone()
    graph.finish_replay()
    for state in cache.layers.values():
        monkeypatch.setattr(
            state, "graph_row", lambda *a: pytest.fail("address queried")
        )
        monkeypatch.setattr(
            state, "prepare_append", lambda *a: pytest.fail("layer revisited")
        )
    graph.prepare([101, 7], [0, 0], [34, 34], paired=True)
    torch.testing.assert_close(graph.inputs.descriptors, descriptors)
    for field in (kernels.HOST, kernels.LOW, kernels.RESIDENT, kernels.PAGES):
        torch.testing.assert_close(graph.tables[:, :2, field], before[:, [1, 0], field])
    assert graph.tables[..., kernels.PREFIX].tolist() == [[34, 34, -1]] * 3
    assert graph.tables[..., kernels.STEP].tolist() == [[1, 1, 0]] * 3
    graph.finish_replay()
    graph.prepare([7], [0], [35], paired=False)
    assert graph.tables[..., kernels.PREFIX].tolist() == [[35, -1, -1]] * 3
    assert graph.tables[..., kernels.STEP].tolist() == [[2, 0, 0]] * 3


def test_stable_prepare_has_one_input_fence_and_no_per_layer_event_calls(monkeypatch):
    cache, graph, _ = populated(layers=36)
    graph.prepare([7], [0], [33], paired=True)
    graph.finish_replay()
    uploaded = Mock()
    uploaded.query.return_value = True
    graph.inputs.uploads = [(*entry[:4], uploaded) for entry in graph.inputs.uploads]
    consumed, main = object(), Mock()
    graph.inputs.consumed = consumed
    graph.cuda = True  # Simulate the CUDA API boundary, retaining real CPU buffers.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: main)
    for state in cache.layers.values():
        monkeypatch.setattr(
            state, "prepare_append", lambda *a: pytest.fail("slow path")
        )
    for layout in graph.layers.values():
        layout.background = Mock()
    graph.prepare([7], [0], [34], paired=True)
    main.wait_event.assert_called_once_with(consumed)
    uploaded.record.assert_called_once_with(main)
    uploaded.synchronize.assert_not_called()
    for layout in graph.layers.values():
        layout.background.inputs_ready.record.assert_not_called()


def test_request_release_reuse_and_eager_append_invalidate_cached_binding():
    cache, graph, key = populated()
    graph.prepare([7], [0], [33], paired=True)
    graph.finish_replay()
    old_index = graph.inputs.requests[7].index
    for layer in range(3):
        cache.layers[(7, 0, layer)].append(key[:1], key[:1], 34)
    assert graph.inputs.requests[7].dirty
    graph.prepare([7], [0], [35], paired=False)
    assert graph.tables[..., kernels.PREFIX].tolist() == [[35, -1, -1]] * 3
    graph.finish_replay()
    cache.release(7)
    assert 7 not in graph.inputs.requests
    for layer in range(3):
        cache.get(7, 1, layer, key, key).append(key[:3], key[:3], 0)
    graph.prepare([7], [1], [3], paired=False)
    assert graph.inputs.requests[7].index == old_index
    assert graph.tables[..., kernels.PREFIX].tolist() == [[3, -1, -1]] * 3
    assert not graph.tables[:, 0, kernels.HOST].any()
    assert (graph.inputs.descriptors[old_index, :, -1] == 1).all()


def test_external_capacity_growth_refreshes_pointers_before_replay():
    cache, graph, _ = populated()
    graph.prepare([7], [0], [33], paired=False)
    graph.finish_replay()
    old = [cache.layers[(7, 0, layer)].host for layer in range(3)]
    for layer in range(3):
        cache.layers[(7, 0, layer)]._grow(256)
    graph.prepare([7], [0], [34], paired=False)
    for layer in range(3):
        state = cache.layers[(7, 0, layer)]
        assert (
            graph.tables[layer, 0, kernels.HOST]
            == state.host.data_ptr()
            != old[layer].data_ptr()
        )
        assert graph.tables[layer, 0, kernels.CAPACITY] == 256


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="CUDA or Triton interpreter required",
)
@pytest.mark.parametrize("batch_size", [4, 19, 256])
def test_descriptor_gather_kernel_masks_stale_generation_and_invalid_indices(
    batch_size,
):
    device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    descriptors = torch.arange(3 * 2 * 12, dtype=torch.int64, device=device).reshape(
        3, 2, 12
    )
    descriptors[..., -1] = 5
    inputs = (
        torch.tensor(
            [[2, 5, 101, 7], [1, 4, 200, 9], [-1, 5, 10, 0], [3, 5, 10, 0]],
            device=device,
        )
        .repeat((batch_size + 3) // 4, 1)[:batch_size]
        .contiguous()
    )
    tables = torch.full((2, batch_size + 3, 11), 999, dtype=torch.int64, device=device)
    kernels.prepare_request_tables[(2, (batch_size + 15) // 16)](
        descriptors,
        inputs,
        tables,
        REQUESTS=3,
        LAYERS=2,
        MAX_BATCH=batch_size + 3,
        BATCH=batch_size,
    )
    expected = torch.zeros_like(tables)
    expected[..., kernels.PREFIX] = -1
    expected[:, :batch_size:4] = descriptors[2, :, None, :11]
    expected[:, :batch_size:4, kernels.PREFIX] = 101
    expected[:, :batch_size:4, kernels.STEP] += 7
    expected[:, batch_size:] = 999
    torch.testing.assert_close(tables, expected)
