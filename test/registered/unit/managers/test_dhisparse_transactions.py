from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.dhisparse_coordinator import AdaptiveHiSparse


def transaction_fixture():
    n, hot, width = 3, 4, 4
    stride = hot + width + 1
    payload = torch.arange(n * stride * 8).view(n * stride, 8).clone()
    host_payload = torch.full((n * 32, 8), -1)
    host_map = torch.arange(n * 32).view(n, 32)
    mapping = torch.zeros(64, dtype=torch.int64)
    pool = SimpleNamespace(full_to_hisparse_device_index_mapping=mapping)
    pool.transfer_values_on_device = lambda dst, src: payload.__setitem__(
        dst, payload[src].clone()
    )
    c = SimpleNamespace(
        is_dsv4_hisparse=False,
        device="cpu",
        device_buffer_size=hot,
        padded_buffer_size=stride,
        mem_pool_device=pool,
        req_to_device_buffer=torch.arange(n * stride).view(n, stride),
        req_device_buffer_tokens=torch.full((1, n, stride), -1, dtype=torch.int32),
        req_device_buffer_token_locs=torch.arange(n * stride, dtype=torch.int32).view(
            1, n, stride
        ),
        req_to_host_pool=host_map,
        req_to_host_pool_allocated_len=torch.zeros(n),
        _eager_backup_previous_token=Mock(),
        wait_for_pending_backup=Mock(),
        _backup_done_event=SimpleNamespace(record=Mock()),
    )
    c.mem_pool_host = SimpleNamespace(
        alloc_paged_token_slots=lambda table, lens, rid, start, count: table[
            rid, start : start + count
        ]
    )
    adaptive = AdaptiveHiSparse.__new__(AdaptiveHiSparse)
    adaptive.c, adaptive.active_verify = c, None
    adaptive.device_module = SimpleNamespace(current_stream=lambda: None)
    adaptive._backup = lambda dst, src: host_payload.__setitem__(dst, payload[src])
    return adaptive, payload, host_payload, mapping


@pytest.mark.parametrize("commit", [False, True])
def test_verify_commit_only_accepts_prefix_and_rollback_restores_mapping(commit):
    adaptive, payload, host, mapping = transaction_fixture()
    c = adaptive.c
    requests, prefixes = torch.tensor([2, 1]), torch.tensor([7, 5])
    logical = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    before_mapping = mapping.clone()
    canonical = c.req_to_device_buffer[requests, c.device_buffer_size]
    before_tail = payload[canonical].clone()
    adaptive.prepare_verify(requests, prefixes, logical)
    physical = mapping[logical].clone()
    values = torch.arange(64).view(2, 4, 8) + 1000
    payload[physical] = values
    if commit:
        adaptive.commit_verify(torch.tensor([3, 1]))
        for row, count in enumerate((3, 1)):
            rid, prefix = int(requests[row]), int(prefixes[row])
            host_ids = c.req_to_host_pool[rid, prefix : prefix + 4]
            torch.testing.assert_close(host[host_ids[:count]], values[row, :count])
            assert (host[host_ids[count:]] == -1).all()
            torch.testing.assert_close(payload[canonical[row]], values[row, count - 1])
    else:
        adaptive.rollback_verify()
        torch.testing.assert_close(payload[canonical], before_tail)
        assert (host == -1).all()
    assert adaptive.active_verify is None
    torch.testing.assert_close(mapping, before_mapping)
    assert (c.req_device_buffer_tokens[:, requests, 5:] == -1).all()


def test_second_verify_cannot_overwrite_live_transaction():
    adaptive, *_ = transaction_fixture()
    adaptive.prepare_verify(
        torch.tensor([1]), torch.tensor([3]), torch.tensor([[10, 11]])
    )
    with pytest.raises(RuntimeError, match="Unfinished"):
        adaptive.prepare_verify(
            torch.tensor([1]), torch.tensor([3]), torch.tensor([[10, 11]])
        )
    adaptive.rollback_verify()
