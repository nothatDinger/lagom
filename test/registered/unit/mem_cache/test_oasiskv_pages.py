"""Physical page sharing and threshold transitions, independent of CUDA."""

import pytest
import torch
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_dense_growth_transition_keeps_pages_and_recycles_between_requests():
    torch.manual_seed(24)
    config = OasisKVConfig(token_budget=8, block_size=2)
    cache = OasisKVCache(config, "cpu", max_requests=3)
    key, value = torch.randn(14, 2, 8), torch.randn(14, 2, 5)
    dense = cache.get(0, 0, 0, key, value)
    other = cache.get(1, 0, 0, key, value)
    other.append(key[:3], value[:3], 0)
    assert dense.page_table.pool is other.page_table.pool
    pool = dense.page_table.pool
    dense.append(key[:1], value[:1], 0)
    first = dense.page_table.ids[-1]
    table_address = dense.page_table.pointers.data_ptr()
    assert len(pool.allocated) == 3  # 1 + ceil(3/2); not two full sparse budgets
    for pos in range(1, 8):
        dense.append(key[pos : pos + 1], value[pos : pos + 1], pos)
    assert dense.host is None and not dense.sparse
    assert dense.page_table.ids[0] == first
    old_pages = {p for p in dense.page_table.ids if p >= 0}
    dense.append(key[8:9], value[8:9], 8)
    assert dense.sparse and dense.host is not None
    assert old_pages <= set(dense.page_table.ids)
    assert dense.page_table.pointers.data_ptr() == table_address
    assert not other.sparse and other.host is None
    dense.append(key[9:], value[9:], 9)
    torch.testing.assert_close(
        dense.materialize_history()[..., :13],
        torch.cat((key, value), -1).transpose(0, 1),
    )
    # Transition/growth of one request must not alter the other request's pages.
    torch.testing.assert_close(
        other.materialize_history()[..., :13],
        torch.cat((key[:3], value[:3]), -1).transpose(0, 1),
    )
    ids = set(dense.page_table.ids)
    cache.release(0)
    assert ids.isdisjoint(pool.allocated)
    replacement = cache.get(0, 1, 0, key, value)
    replacement.append(key[:2], value[:2], 0)
    assert replacement.page_table.ids[-1] in ids
    cache.clear()
    assert not pool.allocated


def test_transition_during_chunked_prefill_preserves_partial_active_block():
    cache = OasisKVCache(OasisKVConfig(token_budget=8, block_size=4), "cpu")
    key, value = torch.randn(13, 2, 4), torch.randn(13, 2, 3)
    state = cache.get(0, 0, 0, key, value)
    state.append(key[:6], value[:6], 0)
    state.append(key[6:], value[6:], 6)
    torch.testing.assert_close(
        state.materialize_history()[..., :7],
        torch.cat((key, value), -1).transpose(0, 1),
    )
    state.prepare_attention(key[-1])
    assert state.sparse and state.initialized


def test_sparse_block_advance_does_not_add_a_foreground_prefetch_fence(monkeypatch):
    cache = OasisKVCache(OasisKVConfig(token_budget=4, block_size=2), "cpu")
    key = torch.randn(9, 1, 4)
    state = cache.get(0, 0, 0, key, key)
    state.append(key, key, 0)
    # Reserve summary/host capacity, so the only changing value is active index.
    state._grow(20)
    calls = []
    monkeypatch.setattr(state, "join", lambda **kw: calls.append(kw))
    state.prepare_append(11)
    assert calls == []


def test_pool_refuses_unallocated_page_and_double_release():
    cache = OasisKVCache(OasisKVConfig(token_budget=4, block_size=2), "cpu")
    key = torch.zeros(1, 1, 4)
    state = cache.get(0, 0, 0, key, key)
    with pytest.raises(RuntimeError, match="not allocated"):
        state.block(0)
    state.append(key, key, 0)
    page = state.page_table.ids[-1]
    cache.release(0)
    with pytest.raises(RuntimeError, match="double free"):
        state.page_table.pool.release([page])


def test_page_reclamation_waits_for_attention_readers_before_reuse():
    from unittest.mock import MagicMock

    cache = OasisKVCache(OasisKVConfig(token_budget=4, block_size=2), "cpu")
    k = torch.zeros(1, 1, 4)
    state = cache.get(0, 0, 0, k, k)
    state.append(k, k, 0)
    page = state.page_table.ids[-1]
    event = MagicMock()

    def check_owned():
        assert page in state.page_table.pool.allocated

    event.synchronize.side_effect = check_owned
    state.read_done = event
    cache.release(0)
    event.synchronize.assert_called_once_with()
    assert page not in state.page_table.pool.allocated
