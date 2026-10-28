"""Grouped physical-page decode, runnable on CUDA or TRITON_INTERPRET=1."""

import os
from types import SimpleNamespace

import pytest
import torch
import triton
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
from sglang.srt.mem_cache.oasiskv_attention import (
    GroupedAttentionWorkspace,
    attention_geometry,
    use_grouped_attention,
)
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_graph import GraphPlan

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
needs_kernel = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="CUDA or interpreter required",
)


def test_dispatch_and_static_geometry():
    config = OasisKVConfig()
    assert use_grouped_attention(config, torch.bfloat16, 4, 128, 128)
    for dtype, group, kd, vd in [
        (torch.float32, 4, 128, 128),
        (torch.float16, 1, 128, 128),
        (torch.bfloat16, 64, 576, 512),
    ]:
        assert not use_grouped_attention(config, dtype, group, kd, vd)
    assert not use_grouped_attention(
        OasisKVConfig.from_json('{"attention_kernel":"legacy"}'),
        torch.bfloat16,
        4,
        128,
        128,
    )
    with pytest.raises(ValueError, match="attention_kernel"):
        OasisKVConfig.from_json('{"attention_kernel":"invalid"}')
    for batch in (1, 8, 64, 256):
        g = attention_geometry(batch, 8, 4, 2, 128, 16, 132)
        assert g.splits <= 32
        assert g.splits * g.tiles_per_split * g.tile >= 2048
    assert (
        attention_geometry(1, 8, 4, 2, 128, 16, 132).splits
        > attention_geometry(256, 8, 4, 2, 128, 16, 132).splits
    )
    w = GroupedAttentionWorkspace(256, 8, 4, 128, 128, config, "cpu")
    for batch in (1, 2, 4, 7, 8, 32, 128, 256):
        for width in (1, 2):
            g = attention_geometry(batch, 8, 4, width, 128, 16, 132)
            assert batch * width * 32 * g.splits * 130 <= w.partial.numel()
    assert w.partial.numel() < 256 * 2 * 32 * 32 * 130


@needs_kernel
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "block,group,kd,vd,cap,paged",
    [
        (16, 4, 128, 128, 0, True),
        (4, 3, 24, 19, 2, True),
        (1, 16, 64, 32, 0, True),
        (32, 2, 48, 40, 2, False),
        (64, 7, 256, 192, 0, True),
    ],
)
def test_matches_reference_and_legacy_with_padding_and_replaced_pages(
    width, dtype, block, group, kd, vd, cap, paged
):
    if INTERPRET and dtype == torch.bfloat16:
        pytest.skip(
            "Triton interpreter dot treats BF16 bit patterns as uint16; run on CUDA"
        )
    torch.manual_seed(552)
    batch, heads, slots = 3, 2, 5
    config = OasisKVConfig(token_budget=slots * block, block_size=block)
    storage = triton.cdiv((kd + vd) * 2, 16) * 8
    workspace = GroupedAttentionWorkspace(batch, heads, group, kd, vd, config, DEVICE)
    table = torch.zeros((batch, kernels.FIELDS), dtype=torch.int64, device=DEVICE)
    ids = torch.tensor([[[3, -1, 0, 1, 2], [1, 2, -1, 3, 0]]] * batch, device=DEVICE)
    selected = torch.tensor(
        [[[True, False, True, False, True], [False, True, False, True, True]]] * batch,
        device=DEVICE,
    )
    q = torch.randn(batch * width, heads * group, kd, dtype=dtype, device=DEVICE)
    k = torch.randn(batch * width, heads, kd, dtype=dtype, device=DEVICE)
    v = torch.randn(batch * width, heads, vd, dtype=dtype, device=DEVICE)
    data = torch.randn(
        batch, heads, slots + 1, block, storage, dtype=dtype, device=DEVICE
    )
    # Separate slabs with different physical addresses; neither attention path
    # may assume adjacent slots or requests have contiguous KV storage.
    replacement = torch.randn_like(data)
    pointers = torch.empty((batch, heads, slots + 1), dtype=torch.int64, device=DEVICE)
    output, legacy = [
        torch.empty(batch * width, heads * group, vd, dtype=dtype, device=DEVICE)
        for _ in range(2)
    ]
    pages_address = workspace.pages.data_ptr()
    for step, backing in enumerate((data, replacement)):
        positions = (
            [4 * block + block - 1, 2 * block, -1]
            if step == 0
            else [-1, 4 * block + 1, 3 * block]
        )
        # Padding can contain unspecified values from an upstream graph.
        for x in (q, k, v):
            x.normal_()
            pad = positions.index(-1)
            x[pad * width : (pad + 1) * width].fill_(float("nan"))
        for b in range(batch):
            table[b, kernels.PREFIX] = positions[b]
            table[b, kernels.RESIDENT] = ids[b].data_ptr()
            table[b, kernels.DATA] = backing[b].data_ptr()
            for h in range(heads):
                for s in range(slots + 1):
                    pointers[b, h, s] = backing[b, h, s].data_ptr()
            if paged:
                table[b, kernels.PAGES] = pointers[b].data_ptr()
        workspace.prepare(table, ids, selected, storage, dtype)
        workspace.forward(
            q,
            k,
            v,
            output,
            batch=batch,
            width=width,
            storage=storage,
            scale=kd**-0.5,
            cap=cap,
        )
        # Compare against a separate Torch reference, not just another kernel.
        expected = torch.zeros_like(output)
        for b, pos in enumerate(positions):
            if pos < 0:
                continue
            for h in range(heads):
                history = []
                for s in range(slots):
                    logical = int(ids[b, h, s])
                    length = min(block, max(0, pos - logical * block + 1))
                    if logical >= 0 and selected[b, h, s]:
                        history.append(backing[b, h, s, :length, : kd + vd])
                history.append(backing[b, h, slots, : pos % block + 1, : kd + vd])
                packed = torch.cat(history)
                for lane in range(width):
                    keys, values = packed[:, :kd], packed[:, kd : kd + vd]
                    if lane == 1:
                        keys = torch.cat((keys, k[b * width + lane, h : h + 1]))
                        values = torch.cat((values, v[b * width + lane, h : h + 1]))
                    queries = q[b * width + lane, h * group : (h + 1) * group].float()
                    logits = queries @ keys.float().T * kd**-0.5
                    if cap:
                        logits = cap * torch.tanh(logits / cap)
                    expected[b * width + lane, h * group : (h + 1) * group] = (
                        logits.softmax(-1) @ values.float()
                    ).to(dtype)
        tokens = (slots + 1) * block + 1
        splits = min(8, triton.cdiv(tokens, 256))
        partial = torch.empty(
            batch * width, heads * group, splits, vd + 2, device=DEVICE
        )
        kernels.attention_partial[(batch, heads * group, splits)](
            table,
            q,
            k,
            v,
            selected,
            partial,
            HEADS=heads,
            GROUP=group,
            KD=kd,
            VD=vd,
            STORAGE=storage,
            SLOTS=slots,
            BLOCK=block,
            WIDTH=width,
            SCALE=kd**-0.5,
            CAP=cap,
            SPLITS=splits,
            PER_SPLIT=triton.cdiv(tokens, splits * 32) * 32,
            DK=triton.next_power_of_2(kd),
            DV=triton.next_power_of_2(vd),
            TILE=32,
        )
        kernels.attention_reduce[(batch * width, heads * group)](
            partial,
            legacy,
            QHEADS=heads * group,
            VD=vd,
            SPLITS=splits,
            NS=triton.next_power_of_2(splits),
            DV=triton.next_power_of_2(vd),
        )
        tol = 2e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(legacy, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(output, expected, atol=tol, rtol=tol)
        assert workspace.pages.data_ptr() == pages_address
        assert not output.reshape(batch, width, heads * group, vd)[
            positions.index(-1)
        ].any()


@needs_kernel
def test_foreground_publication_fuses_descriptors_and_background_leaves_them_untouched():
    batch, heads, slots, block = 2, 1, 4, 16
    config = OasisKVConfig(token_budget=slots * block, block_size=block)
    table = torch.zeros(batch, kernels.FIELDS, dtype=torch.int64, device=DEVICE)
    data = torch.zeros(heads, slots + 1, block, 64, dtype=torch.float16, device=DEVICE)
    ids = torch.tensor([[3, 0, -1, 1]], device=DEVICE)
    ages = torch.zeros_like(ids)
    table[:, kernels.PREFIX] = -1
    table[0, kernels.PREFIX] = 64
    table[0, kernels.DATA], table[0, kernels.RESIDENT], table[0, kernels.AGES] = (
        data.data_ptr(),
        ids.data_ptr(),
        ages.data_ptr(),
    )
    plan = GraphPlan(batch, heads, slots, DEVICE)
    plan.ids.copy_(ids.expand_as(plan.ids))
    plan.selected.fill_(True)
    plan.ages.fill_(1)
    plan.counts.zero_()
    w = GroupedAttentionWorkspace(batch, heads, 4, 32, 32, config, DEVICE)
    plan.transfer(
        table, config, 64, torch.float16, background=False, attention_pages=w.pages
    )
    expected = w.pages.clone()
    w.prepare(table, plan.ids, plan.selected, 64, torch.float16)
    torch.testing.assert_close(w.pages, expected)
    assert not w.pages[1].any()
    plan.ids[0, 0, 0] = 2
    plan.transfer(table, config, 64, torch.float16, background=True)
    torch.testing.assert_close(w.pages, expected)


@needs_kernel
def test_dispatch_through_graph_state_and_eager_paged_attention():
    from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
    from sglang.srt.mem_cache.oasiskv_graph import OasisKVGraphState

    torch.manual_seed(218)
    config = OasisKVConfig(token_budget=64, block_size=16)
    cache, reference_cache = OasisKVCache(config, DEVICE), OasisKVCache(config, "cpu")
    graph = OasisKVGraphState(cache, 2, 128, 1)
    layer = SimpleNamespace(layer_id=0, scaling=32**-0.5, logit_cap=2)
    history_k, history_v = [
        torch.randn(93, 2, 32, dtype=torch.float16) for _ in range(2)
    ]
    q = torch.randn(2, 8, 32, dtype=torch.float16, device=DEVICE)
    k, v = [torch.randn(2, 2, 32, dtype=torch.float16, device=DEVICE) for _ in range(2)]
    assert not graph.forward(q, k, v, layer, 2, paired=False).any()
    assert graph.layers[0].workspace.grouped is not None
    state = cache.get(2, 0, 0, history_k.to(DEVICE), history_v.to(DEVICE))
    ref = reference_cache.get(2, 0, 0, history_k, history_v)
    state.append(history_k.to(DEVICE), history_v.to(DEVICE), 0)
    ref.append(history_k, history_v, 0)
    for _ in range(2):
        prefix = ref.length
        graph.prepare([2], [0], [prefix], paired=False)
        ref.append(k[:1].cpu(), v[:1].cpu(), prefix)
        ref.prepare_attention(q[0].cpu())
        expected = ref.attend(q[:1].cpu(), layer.scaling, logit_cap=layer.logit_cap)
        actual = graph.forward(q, k, v, layer, 2, paired=False)
        graph.finish_replay()
        torch.testing.assert_close(
            actual[:1].reshape_as(expected).cpu(), expected, atol=2e-3, rtol=2e-3
        )
        assert not actual[1:].any()
        eager = state._attend_paged(q[:1], layer.scaling, None, None, layer.logit_cap)
        torch.testing.assert_close(eager.cpu(), expected, atol=2e-3, rtol=2e-3)
