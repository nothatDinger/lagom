"""Same-step demand refill/prefetch I/O must be disjoint by logical block.

Run with CUDA, or TRITON_INTERPRET=1 SGLANG_USE_CPU_ENGINE=0 for the actual
Triton planners/copy kernels on CPU. Kernel launch count is not an I/O count.
"""

import os

import pytest
import torch
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_graph import OasisKVGraphState

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


def _sources(plan, head, batch=None):
    count = plan.counts[head] if batch is None else plan.counts[batch, head]
    sources = plan.sources[head] if batch is None else plan.sources[batch, head]
    return set(sources[: int(count)].tolist())


def _state(cache, slot, heads=2):
    key = torch.arange(
        64 * heads * 4, dtype=torch.float32, device=cache.pipeline.device
    ).reshape(64, heads, 4)
    state = cache.get(slot, 0, 0, key, key)
    state.append(key, key, 0)
    state.resident.copy_(torch.tensor([[0, 1, 2, 3]] * heads, device=key.device))
    state.ages.fill_(0)
    if state.backup_done is not None:
        state.backup_done.synchronize()
    # Make device bytes consistent with the initial metadata, including hits.
    for page in range(4):
        state.block(page).copy_(state.host[:, page].to(key.device))
    return state, key


@pytest.mark.parametrize(
    "draft,expected",
    [
        ([4, 5, 6, 7], set()),  # everything has just been refilled
        ([4, 5, 8, 9], {8, 9}),  # two refill hits and two genuine prediction misses
        ([3, 4, 5, 8], {8}),  # old hit plus two new refill hits
    ],
)
@pytest.mark.parametrize("cap", [None, 1, 0])
def test_eager_prefetch_subtracts_complete_post_refill_set(draft, expected, cap):
    cache = OasisKVCache(
        OasisKVConfig(
            token_budget=4, block_size=1, native_sparse=True, max_new_blocks=cap
        ),
        "cpu",
    )
    state, key = _state(cache, 0)
    root = [3, 4, 5, 6] if 3 in draft else [4, 5, 6, 7]
    calls = []
    copy = state._copy_plan

    def record(plan):
        calls.append([_sources(plan, h) for h in range(2)])
        copy(plan)

    state._copy_plan = record
    root_ids = torch.tensor([root, root])
    state.prepare_attention(key[0], root_ids)
    base = state.resident.clone()
    plan = state.plan_prefetch(torch.tensor([draft, draft]))
    state.submit_prefetch(plan)
    admitted = set(sorted(expected)[:cap]) if cap is not None else expected
    for head in range(2):
        assert calls[1][head] == admitted
        assert calls[0][head].isdisjoint(calls[1][head])
        # Removing overlap from candidates without preserving hits would let
        # LRU evict these wanted blocks while admitting new predictions.
        hits = set(draft) & set(base[head].tolist())
        assert hits <= set(state.resident[head].tolist())


@pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="CUDA or Triton interpreter required",
)
@pytest.mark.parametrize("copy_kernel", ["tiled", "legacy"])
@pytest.mark.parametrize("cap", [None, 1, 0])
@pytest.mark.parametrize(
    "draft,expected", [([4, 5, 6, 7], set()), ([4, 5, 8, 9], {8, 9})]
)
def test_graph_prefetch_uses_layer_owned_refill_result(
    draft, expected, copy_kernel, cap
):
    config = OasisKVConfig(
        token_budget=4,
        block_size=1,
        native_sparse=True,
        copy_kernel=copy_kernel,
        max_new_blocks=cap,
    )
    expected = set(sorted(expected)[:cap]) if cap is not None else expected
    cache = OasisKVCache(config, DEVICE)
    state, key = _state(cache, 7)
    graph = OasisKVGraphState(cache, 2, 128, 2)
    layout = graph.get_layer(0, key, key, key)
    next_layer = graph.get_layer(1, key, key, key)
    assert layout.workspace is next_layer.workspace
    state2 = cache.get(7, 0, 1, key, key)
    state2.append(key, key, 0)
    graph.prepare([7], [0], [64], paired=True)
    graph.inputs.materialize(2)
    table = graph.tables[0]
    w = layout.workspace
    w.candidates.fill_(99)  # padded request must not generate I/O
    w.candidates[0].copy_(torch.tensor([[4, 5, 6, 7]] * 2, device=DEVICE))
    w.real.select(table, w.candidates, cap=None, background=False)
    foreground = [_sources(w.real, h, 0) for h in range(2)]
    assert foreground == [{4, 5, 6, 7}] * 2
    w.real.transfer(
        table,
        config,
        state.storage_dim,
        key.dtype,
        background=False,
        next_plan=layout.plan,
    )
    base = state.resident.clone()
    # A later layer can overwrite all shared foreground planning scratch while
    # this layer's side graph is still queued. The handoff must not alias it.
    next_layer.workspace.real.ids.fill_(80)
    next_layer.workspace.real.ages.fill_(80)
    next_layer.workspace.real.sources.fill_(80)
    layout.draft.fill_(99)
    layout.draft[0].copy_(torch.tensor([draft] * 2, device=DEVICE))
    layout.plan.select(
        table, layout.draft, cap=cap, background=True, use_refill_base=True
    )
    assert not layout.plan.counts[1].any()
    background = [_sources(layout.plan, h, 0) for h in range(2)]
    assert background == [expected] * 2
    assert all(a.isdisjoint(b) for a, b in zip(foreground, background))
    before = state.data.clone()
    layout.plan.transfer(table, config, state.storage_dim, key.dtype, background=True)
    if not expected:
        # The copy node still launches, but counts are zero and no bytes change.
        torch.testing.assert_close(state.data, before)
    for head in range(2):
        hits = set(draft) & set(base[head].tolist())
        assert hits <= set(state.resident[head].tolist())
        for slot, block in enumerate(state.resident[head].tolist()):
            torch.testing.assert_close(
                state.block(slot)[head].cpu(), state.host[head, block]
            )
    actual_bytes = (
        int(layout.plan.counts.sum())
        * config.block_size
        * state.storage_dim
        * key.element_size()
    )
    assert (
        actual_bytes
        == len(expected)
        * 2
        * config.block_size
        * state.storage_dim
        * key.element_size()
    )
