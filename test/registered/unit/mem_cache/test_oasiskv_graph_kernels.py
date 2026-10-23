"""Numerical tests for the actual kernels (CUDA or TRITON_INTERPRET=1).

CPU invocation:
  TRITON_INTERPRET=1 SGLANG_USE_CPU_ENGINE=0 PYTHONPATH=python pytest -q <this file>
The interpreter exercises indirect addressing and updates, not CUDA scheduling.
"""

import os

import pytest
import torch
import triton
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache, plan_delta, quest_scores
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_graph import OasisKVGraphState

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="CUDA or Triton interpreter required",
)
DEVICE = "cpu" if INTERPRET else "cuda"


@pytest.mark.parametrize(
    "slots,cap", [(3, 0), (3, 1), (8, None), (128, 10), (2048, 128)]
)
def test_delta_matches_reference_including_empty_slots_hits_and_lru_ties(slots, cap):
    torch.manual_seed(51)
    config = OasisKVConfig(token_budget=slots, block_size=1, native_sparse=True)
    cache = OasisKVCache(config, DEVICE)
    graph = OasisKVGraphState(cache, 2, 10000, 1)
    key, value, query = (
        torch.zeros(1, 2, 4, device=DEVICE),
        torch.zeros(1, 2, 3, device=DEVICE),
        torch.zeros(1, 4, 4, device=DEVICE),
    )
    layout = graph.get_layer(0, query, key, value)
    state = cache.get(1, 0, 0, key, value)
    state.append(key, value, 0)
    resident = torch.stack([torch.randperm(slots * 2)[:slots] for _ in range(2)]).to(
        DEVICE
    )
    resident[:, -1] = -1
    ages = torch.randint(0, 5, (2, slots), device=DEVICE)
    candidates = torch.stack([torch.randperm(slots * 2)[:slots] for _ in range(2)]).to(
        DEVICE
    )
    candidates[:, -1] = -1
    state.resident.copy_(resident)
    state.ages.copy_(ages)
    state.step = 7
    graph.prepare([1], [0], [1], paired=True)
    layout.draft[0].copy_(candidates)
    expected = plan_delta(candidates, resident, ages, 8, cap)
    layout.plan.select(graph.tables[0], layout.draft, cap=cap, background=True)
    torch.testing.assert_close(layout.plan.ids[0], expected.block_ids)
    torch.testing.assert_close(layout.plan.ages[0], expected.last_selected)
    torch.testing.assert_close(layout.plan.counts[0], expected.counts)
    assert not layout.plan.counts[1].any()
    # Selection must not publish the new table before the transfer dependency.
    torch.testing.assert_close(state.resident, resident)


@pytest.mark.parametrize(
    "native,heads,group,kd,vd,block,softcap,budget,prefill",
    [
        (False, 2, 1, 8, 8, 4, 0, 16, 15),
        (False, 2, 3, 8, 5, 4, 2, 16, 15),
        (False, 1, 4, 12, 8, 4, 0, 16, 15),
        (True, 1, 4, 12, 8, 1, 0, 4, 15),
        (False, 2, 3, 8, 5, 4, 0, 16, 0),
        (True, 1, 2, 12, 8, 1, 0, 4, 0),
        (False, 2, 3, 16, 12, 16, 0, 512, 700),
        (True, 1, 2, 16, 12, 1, 0, 512, 700),
    ],
)
def test_token_append_refill_attention_and_prefetch_match_eager_across_boundaries(
    native, heads, group, kd, vd, block, softcap, budget, prefill
):
    torch.manual_seed(71)
    config = OasisKVConfig(
        token_budget=budget,
        block_size=block,
        native_sparse=native,
        max_new_blocks=1,
    )
    cache, reference_cache = OasisKVCache(config, DEVICE), OasisKVCache(config, DEVICE)
    graph = OasisKVGraphState(cache, 2, max(64, prefill + 16), 1)
    keys = torch.randn(prefill + 6, heads, kd, device=DEVICE)
    values = torch.randn(prefill + 6, heads, vd, device=DEVICE)
    queries = torch.randn(prefill + 6, heads * group, kd, device=DEVICE)
    layout = graph.get_layer(0, queries, keys, values)
    state = cache.get(1, 0, 0, keys, values)
    reference = reference_cache.get(1, 0, 0, keys, values)
    for item in (state, reference):
        item.append(keys[:prefill], values[:prefill], 0)
    w = layout.workspace
    for pos in range(prefill, prefill + 5):
        graph.prepare([1], [0], [pos], paired=True)
        table = graph.tables[0]
        q, k, v = [
            torch.cat((x[pos : pos + 2], torch.zeros_like(x[:2])), 0).contiguous()
            for x in (queries, keys, values)
        ]
        kernels.append_token[(2, heads)](
            table,
            k,
            v,
            HEADS=heads,
            KD=kd,
            VD=vd,
            STORAGE=layout.storage_dim,
            SLOTS=config.history_blocks,
            BLOCK=block,
            WIDTH=2,
            NATIVE=native,
            D=triton.next_power_of_2(layout.storage_dim),
        )
        reference.append(k[:1], v[:1], pos)
        if native:
            root_ids = list(dict.fromkeys([0, 2, pos - 3, -1]))
            draft_ids = list(dict.fromkeys([1, 3, pos - 2, pos]))
            root = torch.tensor(root_ids + [-1] * (4 - len(root_ids)), device=DEVICE)
            draft = torch.tensor(draft_ids + [-1] * (4 - len(draft_ids)), device=DEVICE)
            topk = torch.stack((root, draft, root, draft)).int()
            kernels.native_candidates[(2, heads)](
                table,
                topk,
                w.candidates,
                layout.draft,
                HEADS=heads,
                SLOTS=config.history_blocks,
                TOPK=4,
                WIDTH=2,
                N=triton.next_power_of_2(config.history_blocks),
            )
        else:
            root = draft = None
            kernels.score_blocks[(2 * heads, 2, triton.cdiv(graph.max_blocks, 16))](
                table,
                q,
                w.scores,
                HEADS=heads,
                GROUP=group,
                KD=kd,
                BLOCK=block,
                MAX_BLOCKS=graph.max_blocks,
                WIDTH=2,
                D=triton.next_power_of_2(kd),
                TILE=16,
            )
            expected_scores = quest_scores(
                q[0],
                reference.key_min[:, : pos // block],
                reference.key_max[:, : pos // block],
            )
            torch.testing.assert_close(
                w.scores[0, 0, :, : pos // block], expected_scores
            )
            scores, ids = w.scores.topk(config.history_blocks, -1)
            ids.masked_fill_(~scores.isfinite(), -1)
            w.candidates.copy_(ids[:, 0])
            layout.draft.copy_(ids[:, 1])
        reference.prepare_attention(q[0], root)
        expected = reference.attend(
            q[:2], kd**-0.5, draft_key=k[1], draft_value=v[1], logit_cap=softcap
        )
        w.real.select(table, w.candidates, cap=None, background=False)
        w.real.transfer(table, config, layout.storage_dim, k.dtype)
        layout.plan.select(
            table, layout.draft, cap=config.max_new_blocks, background=True
        )
        output = torch.empty((4, heads * group, vd), device=DEVICE)
        kernels.attention_partial[(4, heads * group, w.splits)](
            table,
            q,
            k,
            v,
            w.real.selected,
            w.partial,
            HEADS=heads,
            GROUP=group,
            KD=kd,
            VD=vd,
            STORAGE=layout.storage_dim,
            SLOTS=config.history_blocks,
            BLOCK=block,
            WIDTH=2,
            SCALE=kd**-0.5,
            CAP=softcap,
            SPLITS=w.splits,
            PER_SPLIT=w.per_split,
            DK=triton.next_power_of_2(kd),
            DV=triton.next_power_of_2(vd),
            TILE=32,
        )
        kernels.attention_reduce[(4, heads * group)](
            w.partial,
            output,
            QHEADS=heads * group,
            VD=vd,
            SPLITS=w.splits,
            NS=triton.next_power_of_2(w.splits),
            DV=triton.next_power_of_2(vd),
        )
        torch.testing.assert_close(output[:2], expected, atol=3e-5, rtol=3e-5)
        assert not output[2:].any()
        layout.plan.transfer(table, config, layout.storage_dim, k.dtype)
        reference.prefetch(q[1], draft)
        reference.join()
        graph.finish_replay()
        torch.testing.assert_close(state.resident, reference.resident)
        torch.testing.assert_close(state.ages, reference.ages)
        # Padding bytes/tail tokens are intentionally unspecified. Compare all
        # committed tokens and summaries, including the new active block.
        torch.testing.assert_close(
            state.host[:, : state.capacity].flatten(1, 2)[:, : pos + 1, : kd + vd],
            reference.host.flatten(1, 2)[:, : pos + 1, : kd + vd],
        )
