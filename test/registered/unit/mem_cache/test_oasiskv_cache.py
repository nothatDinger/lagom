import pytest
import torch
import torch.nn.functional as F
from sglang.srt.mem_cache.oasiskv_cache import (
    OasisKVCache,
    OasisKVLayerCache,
    OasisKVPipeline,
    plan_delta,
    quest_scores,
)
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def dense(q, k, v, scale, prefix=0):
    h = q.shape[1]
    k = k.repeat_interleave(h // k.shape[1], dim=1)
    v = v.repeat_interleave(h // v.shape[1], dim=1)
    mask = torch.arange(k.shape[0])[None] <= prefix + torch.arange(q.shape[0])[:, None]
    return F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask.to(q.device),
        scale=scale,
    ).transpose(0, 1)


def test_quest_gqa_opposing_heads_do_not_cancel():
    q = torch.tensor([[2.0], [-2.0]])
    lo = torch.tensor([[[-4.0], [1.0]]])
    hi = torch.tensor([[[-3.0], [2.0]]])
    assert quest_scores(q, lo, hi).tolist() == [[8.0, 4.0]]


def test_capped_delta_retains_hits_and_evicts_oldest_only():
    resident = torch.tensor([[1, 2, 3, 4]])
    ages = torch.tensor([[9, 2, 4, 0]])
    predicted = torch.tensor([[2, 7, 8, -1]])
    plan = plan_delta(predicted, resident, ages, step=10, cap=1)
    assert plan.counts.tolist() == [1]
    assert plan.block_ids.tolist() == [[1, 2, 3, 7]]
    assert plan.last_selected.tolist() == [[9, 10, 4, 10]]
    assert plan.sources[0, 0] == 7
    assert plan.destinations[0, 0] == 3
    frozen = plan_delta(predicted, resident, ages, step=10, cap=0)
    assert torch.equal(frozen.block_ids, resident)


@pytest.mark.parametrize(
    "heads,group,kdim,vdim", [(2, 1, 8, 8), (2, 3, 8, 8), (1, 4, 12, 8)]
)
@pytest.mark.parametrize("block_size", [1, 4])
def test_paired_attention_matches_dense_and_never_commits_draft(
    heads, group, kdim, vdim, block_size
):
    torch.manual_seed(43)
    config = OasisKVConfig(token_budget=16, block_size=block_size)
    state = OasisKVLayerCache(
        config, heads, kdim, vdim, torch.float32, "cpu", OasisKVPipeline("cpu")
    )
    keys = torch.randn(12, heads, kdim)
    values = torch.randn(12, heads, vdim)
    queries = torch.randn(12, heads * group, kdim)
    scale = kdim**-0.5
    # Chunked prefill, including a partial summary block.
    state.append(keys[:3], values[:3], 0)
    torch.testing.assert_close(
        state.dense_prefill(queries[:3], 0, scale),
        dense(queries[:3], keys[:3], values[:3], scale),
    )
    state.append(keys[3:7], values[3:7], 3)
    torch.testing.assert_close(
        state.dense_prefill(queries[3:7], 3, scale),
        dense(queries[3:7], keys[:7], values[:7], scale, 3),
    )
    for pos in range(7, 11):
        state.append(keys[pos : pos + 1], values[pos : pos + 1], pos)
        state.seed(queries[pos])
        out = state.attend(
            queries[pos : pos + 2],
            scale,
            draft_key=keys[pos + 1],
            draft_value=values[pos + 1],
        )
        expected = dense(
            queries[pos : pos + 2], keys[: pos + 2], values[: pos + 2], scale, pos
        )
        torch.testing.assert_close(out, expected)
        state.prefetch(queries[pos + 1])
        assert state.length == pos + 1
    assert state.prefetch_submissions == 4


def test_native_dsa_prefetch_uses_draft_indices_not_root_selection():
    cfg = OasisKVConfig(
        token_budget=2, block_size=1, max_new_blocks=1, native_sparse=True
    )
    state = OasisKVLayerCache(
        cfg, 1, 4, 3, torch.float32, "cpu", OasisKVPipeline("cpu")
    )
    key, value = torch.randn(8, 1, 4), torch.randn(8, 1, 3)
    state.append(key, value, 0)
    assert state.key_min is None and state.key_max is None
    state.seed(torch.randn(2, 4), torch.tensor([0, 1]))
    state.prefetch(torch.randn(2, 4), torch.tensor([5, 6]))
    assert state.resident.tolist() == [[5, 1]]
    assert state.prefetch_submissions == 1
    torch.testing.assert_close(state.data[0, 0, 0, :4], key[5, 0])


def test_native_dsa_prefill_padding_does_not_erase_token_zero():
    cfg = OasisKVConfig(token_budget=2, block_size=1, native_sparse=True)
    state = OasisKVLayerCache(
        cfg, 1, 4, 3, torch.float32, "cpu", OasisKVPipeline("cpu")
    )
    key, value = torch.randn(4, 1, 4), torch.randn(4, 1, 3)
    query = torch.randn(4, 2, 4)
    state.append(key, value, 0)
    indices = torch.tensor([[0, -1, -1], [0, 1, -1], [0, 2, -1], [1, 3, -1]])
    output = state.dense_prefill(query, 0, 0.5, indices=indices)
    for row, selected in enumerate(([0], [0, 1], [0, 2], [1, 3])):
        expected = F.scaled_dot_product_attention(
            query[row].unsqueeze(1),
            key[selected].transpose(0, 1),
            value[selected].transpose(0, 1),
            scale=0.5,
        ).squeeze(1)
        torch.testing.assert_close(output[row], expected)
    with pytest.raises(ValueError, match="logical Indexer Top-K"):
        state.prefetch(query[-1])


def test_generation_reuse_release_and_bounded_gpu_payload():
    cfg = OasisKVConfig(token_budget=8, block_size=4)
    cache = OasisKVCache(cfg, "cpu")
    key, value = torch.randn(128, 2, 8), torch.randn(128, 2, 8)
    state = cache.get(2, 1, 0, key, value)
    state.append(key, value, 0)
    assert state.data.shape == (2, 3, 4, 16)
    assert state.host.shape[1] >= 32
    replacement = cache.get(2, 2, 0, key[:1], value[:1])
    assert replacement.length == 0
    assert len(cache.layers) == 1
    cache.release(2)
    assert not cache.layers


@pytest.mark.parametrize(
    "kwargs",
    [{"block_size": 0}, {"token_budget": 3, "block_size": 2}, {"max_new_blocks": -1}],
)
def test_config_rejects_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        OasisKVConfig(**kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA UVA copy kernel"
)
@pytest.mark.parametrize(
    "kdim,vdim,dtype",
    [(8, 8, torch.float32), (5, 2, torch.bfloat16), (576, 512, torch.bfloat16)],
)
def test_cuda_pipeline_copies_only_selected_block_heads(kdim, vdim, dtype):
    cfg = OasisKVConfig(token_budget=4, block_size=2, max_new_blocks=1)
    state = OasisKVLayerCache(
        cfg, 2, kdim, vdim, dtype, "cuda", OasisKVPipeline("cuda")
    )
    key, value = (
        torch.randn(16, 2, kdim, dtype=dtype, device="cuda"),
        torch.randn(16, 2, vdim, dtype=dtype, device="cuda"),
    )
    state.append(key, value, 0)
    state.seed(torch.randn(4, kdim, dtype=dtype, device="cuda"))
    state.prefetch(torch.randn(4, kdim, dtype=dtype, device="cuda"))
    state.join(reclaim=True)
    for h in range(2):
        for slot, block in enumerate(state.resident[h].tolist()):
            if block >= 0:
                torch.testing.assert_close(
                    state.data[h, slot, :, :kdim], key[block * 2 : block * 2 + 2, h]
                )
                torch.testing.assert_close(
                    state.data[h, slot, :, kdim : kdim + vdim],
                    value[block * 2 : block * 2 + 2, h],
                )
