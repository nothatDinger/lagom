"""Chunk append versus the old per-block implementation, CUDA or Triton interpreter."""

import os

import pytest
import torch
from sglang.srt.mem_cache.oasiskv_cache import (
    OasisKVCache,
    OasisKVLayerCache,
    OasisKVPipeline,
)
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="CUDA or Triton interpreter required",
)


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize(
    "heads,kd,vd,block,native",
    [
        (2, 8, 5, 4, False),
        (2, 17, 5, 6, False),
        (1, 12, 8, 1, True),
        (2, 128, 128, 16, False),
        (1, 192, 128, 16, False),
    ],
)
def test_chunk_append_preserves_history_summaries_and_pages(
    monkeypatch, pooled, heads, kd, vd, block, native
):
    torch.manual_seed(27)
    config = OasisKVConfig(
        token_budget=block * 3, block_size=block, native_sparse=native
    )
    caches = [OasisKVCache(config, DEVICE) for _ in range(2)]
    states = [
        (
            cache.new_layer(0, heads, kd, vd, torch.float32, DEVICE)
            if pooled
            else OasisKVLayerCache(
                config, heads, kd, vd, torch.float32, DEVICE, OasisKVPipeline(DEVICE)
            )
        )
        for cache in caches
    ]
    fast, reference = states
    if INTERPRET:
        monkeypatch.setattr(fast, "_append_reference", fast._append_batched)
    else:
        monkeypatch.setattr(reference, "_append_batched", reference._append_reference)
    chunks = [1, block + 1, 2, block * 3 + 2, 1, block + 3]
    # Strided model projections; packing must not assume contiguous QKV inputs.
    key = torch.randn(sum(chunks), heads, kd * 2, device=DEVICE)[..., ::2]
    value = torch.randn(sum(chunks), heads, vd * 2, device=DEVICE)[..., ::2]
    prefix = 0
    for count in chunks:
        end = prefix + count
        for state in states:
            state.append(key[prefix:end], value[prefix:end], prefix)
            state.join(reclaim=True)
        assert fast.length == reference.length == end
        assert fast.sparse == reference.sparse
        if fast.host is not None:
            torch.testing.assert_close(
                fast.host.flatten(1, 2)[:, :end], reference.host.flatten(1, 2)[:, :end]
            )
        for name in ("key_min", "key_max", "resident", "ages"):
            if getattr(fast, name) is not None:
                torch.testing.assert_close(
                    getattr(fast, name), getattr(reference, name)
                )
        # Includes unchanged sparse slots and the active block's cleared tail.
        torch.testing.assert_close(fast.data, reference.data)
        prefix = end
    for state in states:
        state.release()


@pytest.mark.skipif(INTERPRET, reason="Real DMA lifetime test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_bulk_backup_precedes_cross_stream_consumption(dtype):
    config = OasisKVConfig(token_budget=32, block_size=16)
    cache = OasisKVCache(config, "cuda")
    key = torch.randn(513, 2, 128, dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    state = cache.get(0, 0, 0, key, value)
    state.append(key[:511], value[:511], 0)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        state.append(key[511:], value[511:], 511)
        history = state._prefill_prefix(513)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(history, torch.cat((key, value), -1).transpose(0, 1))
    cache.clear()


def test_packed_promotion_keeps_physical_page_dtype(monkeypatch):
    config = OasisKVConfig(token_budget=8, block_size=4)
    caches = [OasisKVCache(config, DEVICE) for _ in range(2)]
    key = torch.randn(13, 2, 8, device=DEVICE, dtype=torch.float16)
    value = torch.randn(13, 2, 5, device=DEVICE, dtype=torch.float32)
    fast, reference = [cache.get(0, 0, 0, key, value) for cache in caches]
    if INTERPRET:
        monkeypatch.setattr(fast, "_append_reference", fast._append_batched)
    else:
        monkeypatch.setattr(reference, "_append_batched", reference._append_reference)
    for state in (fast, reference):
        state.append(key, value, 0)
        state.join(reclaim=True)
    torch.testing.assert_close(fast.data, reference.data)
    torch.testing.assert_close(
        fast.host.flatten(1, 2)[:, :13], reference.host.flatten(1, 2)[:, :13]
    )
    torch.testing.assert_close(fast.key_min, reference.key_min)
    torch.testing.assert_close(fast.key_max, reference.key_max)
    for cache in caches:
        cache.clear()
