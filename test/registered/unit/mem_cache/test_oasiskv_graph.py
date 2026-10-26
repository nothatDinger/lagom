"""Graph ownership/metadata tests that do not require a CUDA device."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.oasiskv_backend import OasisKVAttnBackend
from sglang.srt.layers.attention.triton_ops.oasiskv import (
    CAPACITY,
    HOST,
    PREFIX,
    RESIDENT,
)
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_graph import (
    GraphPlan,
    OasisKVGraphState,
    copy_launch_geometry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("batch", [1, 8, 32, 256])
@pytest.mark.parametrize("storage", [256, 960])
@pytest.mark.parametrize("ctas_per_sm", [1, 8])
def test_copy_grid_caps_total_ctas_even_when_batch_exceeds_budget(
    batch, storage, ctas_per_sm
):
    tile, grid = copy_launch_geometry(
        batch=batch,
        heads=8,
        slots=128,
        block=16,
        storage=storage,
        element_size=2,
        sm_count=132,
        ctas_per_sm=ctas_per_sm,
    )
    assert tile == 1024
    assert 0 < grid[0] * grid[1] <= 132 * ctas_per_sm
    if batch == 8:
        assert grid == (64, 16 if ctas_per_sm == 8 else 2)
    if batch == 256:
        assert grid == (132 * ctas_per_sm, 1)


def test_copy_geometry_caps_workers_for_tiny_records():
    assert copy_launch_geometry(
        batch=1,
        heads=1,
        slots=3,
        block=1,
        storage=8,
        element_size=4,
        sm_count=132,
        ctas_per_sm=8,
    ) == (8, (1, 3))


def test_copy_kernel_can_be_switched_for_full_server_ab():
    assert OasisKVConfig.from_json("{}").copy_kernel == "tiled"
    assert OasisKVConfig.from_json('{"copy_kernel":"legacy"}').copy_kernel == "legacy"
    with pytest.raises(ValueError, match="copy_kernel"):
        OasisKVConfig.from_json('{"copy_kernel":"unknown"}')


@pytest.mark.parametrize("name", ["refill_ctas_per_sm", "prefetch_ctas_per_sm"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_copy_budget_requires_positive_integer(name, value):
    import json

    with pytest.raises(ValueError, match=name):
        OasisKVConfig.from_json(json.dumps({name: value}))


def test_refill_and_prefetch_dispatch_separate_budgets_and_publish_after_copy(
    monkeypatch,
):
    from sglang.srt.mem_cache import oasiskv_graph

    launches = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((self.name, grid, kwargs))

    monkeypatch.setattr(oasiskv_graph.kernels, "copy_misses", Kernel("copy"))
    monkeypatch.setattr(oasiskv_graph.kernels, "copy_misses_legacy", Kernel("legacy"))
    monkeypatch.setattr(oasiskv_graph.kernels, "publish_plan", Kernel("publish"))
    plan = GraphPlan(8, 8, 128, "cpu")
    plan.sm_count = 132
    table = torch.empty((8, 9), dtype=torch.int64)
    for background in (False, True):
        plan.transfer(
            table, OasisKVConfig(), 256, torch.bfloat16, background=background
        )
    assert [(name, grid) for name, grid, _ in launches] == [
        ("copy", (64, 16)),
        ("publish", (8, 8)),
        ("copy", (64, 2)),
        ("publish", (8, 8)),
    ]
    launches.clear()
    plan.transfer(
        table, OasisKVConfig(copy_kernel="legacy"), 256, torch.bfloat16, background=True
    )
    assert launches[0][:2] == ("legacy", (8, 8, 4))


def setup_graph():
    cache = OasisKVCache(OasisKVConfig(token_budget=8, block_size=2), "cpu")
    graph = OasisKVGraphState(cache, 3, 64, 2)
    key, value, query = (
        torch.randn(12, 2, 4),
        torch.randn(12, 2, 3),
        torch.randn(12, 4, 4),
    )
    for layer in range(2):
        graph.get_layer(layer, query, key, value)
        for slot, length in ((1, 3), (2, 8)):
            cache.get(slot, 0, layer, key, value).append(
                key[:length], value[:length], 0
            )
    return cache, graph, (query, key, value)


def test_replay_tables_keep_addresses_across_reorder_padding_and_growth():
    cache, graph, _ = setup_graph()
    table_address = graph.tables.data_ptr()
    states = [cache.layers[(slot, 0, 0)] for slot in (1, 2)]
    resident_addresses = [state.resident.data_ptr() for state in states]
    old_host = states[1].host  # Keep old allocation alive to prove the pointer changes.
    graph.prepare([2, 1], [0, 0], [8, 3], paired=True)
    assert graph.tables[0, :, PREFIX].tolist() == [8, 3, -1]
    assert old_host is None  # Dense requests have no host KV replica yet.
    assert graph.tables[0, 0, HOST] == states[1].host.data_ptr() != 0
    assert graph.tables[0, 0, CAPACITY] == states[1].capacity
    assert graph.tables[0, 0, RESIDENT] == resident_addresses[1]
    graph.finish_replay()
    assert [state.length for state in states] == [4, 9]
    assert [state.step for state in states] == [1, 1]
    graph.prepare([1], [0], [4], paired=False)
    assert graph.tables.data_ptr() == table_address
    assert graph.tables[0, :, PREFIX].tolist() == [4, -1, -1]
    graph.finish_replay()
    assert states[0].length == 5 and states[0].step == 1
    assert states[0].resident.data_ptr() == resident_addresses[0]


def test_capture_and_idle_do_not_modify_live_cache_or_allocate_fake_requests():
    cache, graph, _ = setup_graph()
    original = [(key, state.length) for key, state in cache.layers.items()]
    backend = OasisKVAttnBackend.__new__(OasisKVAttnBackend)
    backend.graph_state, backend.dsa = graph, None
    fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    backend.init_forward_metadata_out_graph(fb, in_capture=True)
    assert (graph.tables[..., PREFIX] == -1).all()
    fb.actual_forward_mode = ForwardMode.IDLE
    backend.init_forward_metadata_out_graph(fb)
    backend.finish_cuda_graph_replay(fb)
    assert original == [(key, state.length) for key, state in cache.layers.items()]


def test_recycled_request_is_not_bound_to_previous_generations_storage():
    cache, graph, (_, key, value) = setup_graph()
    old = cache.layers[(1, 0, 0)]
    cache.release(1)
    for layer in range(2):
        cache.get(1, 1, layer, key, value).append(key[:2], value[:2], 0)
    graph.prepare([1], [1], [2], paired=True)
    assert graph.tables[0, 0, HOST] == 0  # Recycled request starts dense.
    assert graph.tables[0, 0, RESIDENT] != old.resident.data_ptr()
    graph.finish_replay()
    assert old.length == 3 and cache.layers[(1, 1, 0)].length == 3


@pytest.mark.parametrize(
    "slots,generations,prefixes",
    [([1, 1], [0, 0], [3, 3]), ([1], [0], [4]), ([3], [0], [2]), ([1], [0], [64])],
)
def test_replay_rejects_stale_duplicate_or_missing_request_state(
    slots, generations, prefixes
):
    _, graph, _ = setup_graph()
    with pytest.raises(ValueError):
        graph.prepare(slots, generations, prefixes, paired=True)


def test_eager_refill_and_prefetch_update_resident_tables_in_place():
    cache, _, (query, _, _) = setup_graph()
    state = cache.layers[(2, 0, 0)]
    addresses = state.resident.data_ptr(), state.ages.data_ptr()
    state.prepare_attention(query[0])
    state.prefetch(query[1])
    assert addresses == (state.resident.data_ptr(), state.ages.data_ptr())


def test_replay_gates_each_layer_independently(monkeypatch):
    cache, graph, _ = setup_graph()
    event = object()
    waits = []
    for layer_id in range(2):
        state = cache.layers[(1, 0, layer_id)]
        state.pending = state.backup_done = state.refill_done = event
    graph.cuda = True
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args: SimpleNamespace(wait_event=waits.append),
    )
    graph.prepare([1], [0], [3], paired=True)
    assert waits == [event, event]


def test_backend_replay_uses_live_cpu_slots_and_only_unpadded_lengths():
    cache, graph, _ = setup_graph()
    backend = OasisKVAttnBackend.__new__(OasisKVAttnBackend)
    backend.graph_state, backend.dsa = graph, None
    backend.req_pool = SimpleNamespace(req_generation=torch.zeros(4, dtype=torch.int64))
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=3,
        num_padding=1,
        req_pool_indices_cpu=torch.tensor([2, 1]),
        seq_lens_cpu=torch.tensor([8, 3, 1]),
    )
    backend.init_forward_metadata_out_graph(fb)
    assert graph.tables[0, :, PREFIX].tolist() == [8, 3, -1]
    backend.finish_cuda_graph_replay(fb)
    assert cache.layers[(1, 0, 0)].length == 4


def test_replay_view_preserves_the_live_cpu_request_identity():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        build_replay_fb_view,
    )

    slots = torch.tensor([2, 1])
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices_cpu=slots,
        seq_lens_cpu=torch.tensor([8, 3]),
        seq_lens_sum=11,
        spec_info=None,
    )
    buffers = SimpleNamespace(
        input_ids=torch.zeros(6, dtype=torch.int64),
        positions=torch.zeros(6, dtype=torch.int64),
        req_pool_indices=torch.tensor([2, 1, 0]),
        seq_lens=torch.tensor([8, 3, 1]),
        seq_lens_cpu=torch.tensor([8, 3, 1]),
        mamba_track_indices=None,
    )
    view = build_replay_fb_view(
        fb, buffers, 3, 2, 6, 1, ForwardMode.TARGET_VERIFY, False
    )
    assert view.req_pool_indices_cpu is slots
    assert view.num_padding == 1
    assert view.seq_lens_cpu.tolist() == [8, 3, 1]


def test_replay_rejects_a_prefill_layout_that_would_misinterpret_indirect_pointers():
    cache, graph, _ = setup_graph()
    cache.layers[(1, 0, 0)].key_dim += 1
    with pytest.raises(ValueError, match="geometry"):
        graph.prepare([1], [0], [3], paired=True)
