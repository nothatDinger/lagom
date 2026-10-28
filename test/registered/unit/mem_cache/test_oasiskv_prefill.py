"""Current-chunk KV stays on GPU; only the old prefix may be read from host."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.oasiskv_backend import OasisKVAttnBackend
from sglang.srt.mem_cache import oasiskv_prefill
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def reference(query, key, value, prefix, scale, cap=0, indices=None):
    group = query.shape[1] // key.shape[1]
    q = query.float().transpose(0, 1)
    k = key.float().repeat_interleave(group, 1).transpose(0, 1)
    v = value.float().repeat_interleave(group, 1).transpose(0, 1)
    scores = q @ k.transpose(-1, -2) * scale
    if cap:
        scores = cap * torch.tanh(scores / cap)
    positions = torch.arange(key.shape[0], device=query.device)
    mask = (
        positions[None]
        <= prefix + torch.arange(query.shape[0], device=query.device)[:, None]
    )
    if indices is not None:
        mask &= (positions[None, :, None] == indices[:, None, :]).any(-1)
    weights = scores.masked_fill(~mask, -float("inf")).softmax(-1)
    return (weights @ v).transpose(0, 1).to(query.dtype)


@pytest.mark.parametrize("chunks", [(3, 4, 7, 9), (11, 5, 7)])
@pytest.mark.parametrize(
    "heads,group,kd,vd,native,cap",
    [
        (2, 1, 8, 8, False, 0),
        (2, 3, 8, 8, False, 2),
        (1, 4, 12, 8, False, 0),
        (1, 4, 12, 8, True, 0),
    ],
)
def test_chunked_prefill_matches_full_attention_and_decode_cache(
    monkeypatch, chunks, heads, group, kd, vd, native, cap
):
    torch.manual_seed(72)
    config = OasisKVConfig(
        token_budget=8, block_size=1 if native else 4, native_sparse=native
    )
    cache, old_cache = OasisKVCache(config, "cpu"), OasisKVCache(config, "cpu")
    state = cache.new_layer(0, heads, kd, vd, torch.float32, "cpu")
    old = old_cache.new_layer(0, heads, kd, vd, torch.float32, "cpu")
    key, value, query = (
        torch.randn(sum(chunks) + 2, heads, kd),
        torch.randn(sum(chunks) + 2, heads, vd),
        torch.randn(sum(chunks) + 2, heads * group, kd),
    )
    reads = []
    read_prefix = state._prefill_prefix

    def only_old_prefix(prefix):
        assert prefix == state.length  # current chunk has not been committed
        reads.append((prefix, state.host is not None))
        return read_prefix(prefix)

    monkeypatch.setattr(state, "_prefill_prefix", only_old_prefix)
    monkeypatch.setattr(
        state, "materialize_history", lambda: pytest.fail("full history readback")
    )
    for count in chunks:
        start, end = state.length, state.length + count
        indices = (
            torch.stack(
                (torch.zeros(count, dtype=torch.long), torch.arange(start, end)), 1
            )
            if native
            else None
        )
        output = state.prefill(
            query[start:end],
            key[start:end],
            value[start:end],
            start,
            kd**-0.5,
            indices=indices,
            logit_cap=cap,
        )
        torch.testing.assert_close(
            output,
            reference(
                query[start:end], key[:end], value[:end], start, kd**-0.5, cap, indices
            ),
        )
        old.append(key[start:end], value[start:end], start)
        assert state.length == old.length and state.capacity == old.capacity
        assert state.sparse == old.sparse
        if state.host is not None:
            torch.testing.assert_close(
                state.host.flatten(1, 2)[:, :end], old.host.flatten(1, 2)[:, :end]
            )
        for name in ("key_min", "key_max", "resident"):
            if getattr(state, name) is not None:
                torch.testing.assert_close(getattr(state, name), getattr(old, name))
        torch.testing.assert_close(
            state.block(config.history_blocks), old.block(config.history_blocks)
        )
    assert [p for p, _ in reads] == list(torch.tensor(chunks).cumsum(0)[:-1])
    if not native and chunks[0] == 3:
        assert reads == [(3, False), (7, False), (14, True)]
    # The first real decode plus its disposable probe must see the same cache.
    pos = sum(chunks)
    selected = torch.arange(config.history_blocks) if native else None
    for item in (state, old):
        item.append(key[pos : pos + 1], value[pos : pos + 1], pos)
        item.prepare_attention(query[pos], selected)
    torch.testing.assert_close(
        state.attend(
            query[pos : pos + 2],
            kd**-0.5,
            draft_key=key[pos + 1],
            draft_value=value[pos + 1],
        ),
        old.attend(
            query[pos : pos + 2],
            kd**-0.5,
            draft_key=key[pos + 1],
            draft_value=value[pos + 1],
        ),
    )


def test_backend_flash_prefill_uses_current_projections_before_backup(monkeypatch):
    from sglang.kernels.ops.attention import flash_attention

    cache = OasisKVCache(
        OasisKVConfig(token_budget=8, block_size=4, prefill_chunk_size=2), "cpu"
    )
    backend = OasisKVAttnBackend(
        SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_generation=torch.tensor([0])),
            token_to_kv_pool=SimpleNamespace(oasiskv_cache=cache),
        )
    )
    layer = SimpleNamespace(
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        scaling=8**-0.5,
        logit_cap=2,
        is_cross_attention=False,
        layer_id=0,
    )
    query, key, value = (
        torch.randn(19, 4, 8),
        torch.randn(19, 2, 8),
        torch.randn(19, 2, 8),
    )
    calls = []

    def flash(**kw):
        start = kw["max_seqlen_k"] - kw["max_seqlen_q"]
        end = kw["max_seqlen_k"]
        state = cache.layers[(0, 0, 0)]
        assert state.length == start  # no current-chunk D2H before attention
        assert kw["cu_seqlens_q"].tolist() == [0, end - start]
        assert kw["cu_seqlens_k"].tolist() == [0, end]
        assert kw["causal"] and kw["ver"] == 3 and kw["softcap"] == 2
        torch.testing.assert_close(kw["k"], key[:end])
        torch.testing.assert_close(kw["v"], value[:end])
        if not start:
            assert kw["k"].data_ptr() == key.data_ptr()
            assert kw["v"].data_ptr() == value.data_ptr()
        calls.append((start, end))
        return reference(
            kw["q"], kw["k"], kw["v"], start, kw["softmax_scale"], kw["softcap"]
        )

    monkeypatch.setattr(oasiskv_prefill, "use_flash_prefill", lambda *args: True)
    monkeypatch.setattr(flash_attention, "flash_attn_varlen_func", flash)
    for start, end in ((0, 11), (11, 19)):
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_idle=lambda: False, is_decode=lambda: False
            ),
            req_pool_indices_cpu=torch.tensor([0]),
            extend_prefix_lens_cpu=[start],
            extend_seq_lens_cpu=[end - start],
        )
        backend.init_forward_metadata(fb)
        actual = backend.forward_extend(
            query[start:end], key[start:end], value[start:end], layer, fb
        )
        torch.testing.assert_close(
            actual.view(-1, 4, 8),
            reference(
                query[start:end], key[:end], value[:end], start, layer.scaling, 2
            ),
        )
    assert calls == [(0, 11), (11, 19)]  # one FA3 call per request chunk
    assert cache.layers[(0, 0, 0)].length == 19


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cap", [0, 2])
def test_cuda_fa3_chunked_prefill(dtype, cap, monkeypatch):
    torch.manual_seed(45)
    query = torch.randn(57, 8, 128, device="cuda", dtype=dtype)
    key = torch.randn(57, 2, 128, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    if not oasiskv_prefill.use_flash_prefill(query, key, value, None):
        pytest.skip("FA3 not available on this architecture")
    cache = OasisKVCache(OasisKVConfig(token_budget=16, block_size=4), "cuda")
    state = cache.new_layer(0, 2, 128, 128, dtype, "cuda")
    monkeypatch.setattr(
        state, "materialize_history", lambda: pytest.fail("full readback")
    )
    for start, end in ((0, 7), (7, 19), (19, 57)):
        actual = state.prefill(
            query[start:end],
            key[start:end],
            value[start:end],
            start,
            128**-0.5,
            logit_cap=cap,
        )
        expected = reference(
            query[start:end], key[:end], value[:end], start, 128**-0.5, cap
        )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    cache.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_dma_waits_on_stream_without_cpu_event_sync(monkeypatch):
    query = torch.randn(19, 4, 32, device="cuda", dtype=torch.float16)
    key = torch.randn(19, 2, 32, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    cache = OasisKVCache(OasisKVConfig(token_budget=8, block_size=4), "cuda")
    state = cache.new_layer(0, 2, 32, 32, key.dtype, "cuda")
    state.prefill(query[:17], key[:17], value[:17], 0, 32**-0.5)
    # The next chunk fits the existing 20-token capacity: reclamation is not
    # needed. Read host history from another stream before the CPU has waited
    # for the previous chunk's D2H; only GPU event dependencies make it safe.
    other = torch.cuda.Stream()
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.cuda.Event,
            "synchronize",
            lambda self: pytest.fail("prefill blocked the CPU on an event"),
        )
        with torch.cuda.stream(other):
            actual = state.prefill(query[17:], key[17:], value[17:], 17, 32**-0.5)
    torch.cuda.current_stream().wait_stream(other)
    torch.testing.assert_close(
        actual, reference(query[17:], key, value, 17, 32**-0.5), rtol=0.01, atol=0.01
    )
    cache.clear()
