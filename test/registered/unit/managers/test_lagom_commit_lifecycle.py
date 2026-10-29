"""C4 promotion remains independent of Lagom's pending history transfers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseDSparkWindow,
    dspark_completed_c4_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _transaction(monkeypatch, prefix, *, hot=8, width=15):
    """Exercise production commit with token-granular copies backed by CPU KV."""
    positions = dspark_completed_c4_positions(prefix, width)
    count = len(positions)
    device_kv = torch.full((2, 160), -1, dtype=torch.int64)
    host_kv = torch.full_like(device_kv, -1)
    canonical = torch.arange(1, hot + 2, dtype=torch.int64)
    history = min(prefix // 4, hot)
    device_kv[:, canonical[:history]] = torch.arange(history)[None, :]
    host_kv[:, : prefix // 4] = torch.arange(prefix // 4)[None, :]
    scratch = torch.arange(100, 100 + count, dtype=torch.int64)
    device_kv[:, scratch] = torch.tensor(positions)[None, :]
    transfers = []

    def transfer(*, src_ptrs, dst_ptrs, src_indices, dst_indices):
        # A clone preserves copy semantics even if tensors share storage.
        dst_ptrs[:, dst_indices] = src_ptrs[:, src_indices].clone()
        transfers.append((dst_ptrs is device_kv, dst_indices.clone()))

    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.transfer_cache_dsv4_mla",
        transfer,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.device_module",
        SimpleNamespace(current_stream=lambda: "target"),
    )
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.compress_ratio, c.device_buffer_size, c.device = 4, hot, "cpu"
    c.req_to_host_pool = torch.full((1, 160), -1, dtype=torch.int64)
    c.req_to_host_pool_allocated_len = [prefix // 4]
    c.req_to_device_buffer = canonical[None, :]
    c.req_device_buffer_tokens = torch.arange(hot + 1, dtype=torch.int32)[
        None, None, :
    ].repeat(2, 1, 1)
    c.mem_pool_host = SimpleNamespace(
        device_ptrs=device_kv,
        data_ptrs=host_kv,
        alloc_paged_token_slots=lambda _mapping, _lengths, _req, pos, _count: (
            torch.tensor([pos], dtype=torch.int64)
        ),
    )
    c.mem_pool_device = SimpleNamespace(
        full_to_hisparse_device_index_mapping=scratch.clone()
    )
    c._dspark_scratch_valid_mapping = torch.ones(count, dtype=torch.bool)
    c._dspark_commit_done_event = Mock()
    c._has_pending_dspark_commit = False
    # A global side-stream join would unnecessarily serialize accepted C4
    # promotion with unrelated historical KV transfers.
    c._lagom_graphs = SimpleNamespace(
        wait=Mock(side_effect=AssertionError("commit must not join all prefetch")),
        synchronize=Mock(
            side_effect=AssertionError("commit must not synchronize prefetch")
        ),
    )
    window = HiSparseDSparkWindow(
        compressed_locs=torch.arange(count),
        device_locs=scratch,
        previous_device_mapping=torch.full((count,), -1, dtype=torch.int64),
        req_offsets=[0] * count,
        req_pool_indices_cpu=[0],
        c4_positions=positions,
        prefix_lens_cpu=[prefix],
    )
    c._active_dspark_window = window
    return c, window, device_kv, host_kv, transfers


@pytest.mark.parametrize("prefix", [32, 33, 34, 35])
@pytest.mark.parametrize("accepted_tokens", [0, 1, 4, 8])
@pytest.mark.parametrize("prefetch_first", [False, True])
def test_long_c4_commit_commutes_with_pending_history_h2d(
    monkeypatch, prefix, accepted_tokens, prefetch_first
):
    c, window, device, host, transfers = _transaction(monkeypatch, prefix)
    # A union plan can choose any hot physical location, but cannot choose the
    # reserved newest slot. Emulate a side writer that completes on either side
    # of the actual commit; the result must be identical.
    prefetched_token, destination = 2, int(c.req_to_device_buffer[0, 5])

    def prefetch():
        device[:, destination] = host[:, prefetched_token]

    if prefetch_first:
        prefetch()
    c.commit_dspark_verify_window(window, torch.tensor([accepted_tokens]))
    if not prefetch_first:
        prefetch()

    assert (device[:, destination] == prefetched_token).all()
    first_new = prefix // 4
    committed_end = (prefix + accepted_tokens) // 4
    if committed_end > first_new:
        torch.testing.assert_close(
            host[:, first_new:committed_end],
            torch.arange(first_new, committed_end)[None, :].expand(2, -1),
        )
        assert (device[:, c.req_to_device_buffer[0, -1]] == committed_end - 1).all()
    for device_destination, destinations in transfers:
        if device_destination:
            assert destinations.tolist() == [int(c.req_to_device_buffer[0, -1])]
    assert not c._dspark_scratch_valid_mapping.any()
    assert c._active_dspark_window is None


@pytest.mark.parametrize("prefix", [27, 28, 29, 30, 31])
@pytest.mark.parametrize("accepted_tokens", [0, 1, 4, 8])
def test_c4_commit_crossing_hot_boundary_preserves_historical_slots(
    monkeypatch, prefix, accepted_tokens
):
    c, window, device, host, transfers = _transaction(monkeypatch, prefix)
    # With a short prefix, all historical KV fits and is already resident.
    # Promotion may write new canonical slots, then the reserved newest slot,
    # but must preserve every slot the history-only prefetch can reference.
    historical_slots = c.req_to_device_buffer[0, : prefix // 4]
    historical_bytes = device[:, historical_slots].clone()
    c.commit_dspark_verify_window(window, torch.tensor([accepted_tokens]))
    torch.testing.assert_close(device[:, historical_slots], historical_bytes)

    first_new, committed_end = prefix // 4, (prefix + accepted_tokens) // 4
    for logical in range(first_new, min(committed_end, c.device_buffer_size)):
        assert (device[:, c.req_to_device_buffer[0, logical]] == logical).all()
    if committed_end > c.device_buffer_size:
        assert (device[:, c.req_to_device_buffer[0, -1]] == committed_end - 1).all()
    for device_destination, destinations in transfers:
        if device_destination:
            assert not torch.isin(destinations, historical_slots).any()
    assert not c._dspark_scratch_valid_mapping.any()
