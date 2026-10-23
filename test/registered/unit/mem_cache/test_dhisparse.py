from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.kvcache.dhisparse import allocate_workspace, promote, resolve
from sglang.srt.mem_cache.dhisparse import (
    DHiSparseConfig,
    DraftRingLayout,
    ResidencyManager,
    VerificationLayout,
    allocate_payload_slab,
    staging_groups,
)


def test_draft_ring_keeps_every_live_position_and_alignment_fringe_disjoint():
    layout = DraftRingLayout(2048, 8, 64)
    assert layout.rows == 2176
    for frontier in (0, 2047, 2048, 2175, 2176, 100_000):
        start = max(0, frontier - layout.window) // 64 * 64
        positions = torch.arange(start, frontier + layout.max_verify_width)
        locations = layout.locations(torch.zeros_like(positions), positions)
        assert locations.unique().numel() == positions.numel()
        other = layout.locations(torch.ones_like(positions), positions)
        assert not torch.isin(locations, other).any()


def test_shared_arenas_alias_only_across_non_live_layer_groups():
    slots, count = staging_groups([False, True, True, False, True, False])
    assert slots == [0, 1, 2, 0, 1, 0] and count == 3
    layout = VerificationLayout(2, 4, 3, 4, tuple(slots))
    slab, views, offsets = allocate_payload_slab(
        owned_units=20, unit_shape=(8,), dtype=torch.uint8, device="cpu", layout=layout
    )
    for layer, view in enumerate(views):
        view[:20].fill_(layer + 1)
    views[0][offsets[0]].fill_(77)
    assert torch.equal(views[3][offsets[3]], views[0][offsets[0]])
    assert not (views[1][offsets[1]] == 77).any()
    for layer, view in enumerate(views):
        assert (view[:20] == layer + 1).all()
    assert slab.numel() == (6 * 20 + 3 * 2 * 12) * 8


@pytest.mark.parametrize("width", [1, 4, 8])
def test_union_overflow_exact_payload_order_causality_and_warm_cache(width):
    torch.manual_seed(4)
    n, hot, k, rows, d = 3, 4, 4, 64, 8
    capacity, tail = width * k, 2
    host = torch.arange(n * rows * d).view(n * rows, d)
    host_locs = torch.arange(n * rows).view(n, rows)
    device = torch.full((n * (hot + tail) + n * capacity, d), -999)
    locations = torch.arange(n * (hot + tail), dtype=torch.int32).view(n, hot + tail)
    tokens = torch.full((n, hot + tail), -1, dtype=torch.int32)
    optional = torch.full((n, rows), -1, dtype=torch.int32)
    ages, epochs = torch.zeros(n, hot, dtype=torch.int64), torch.zeros(
        n, dtype=torch.int64
    )
    request_ids = torch.tensor([2, 0, 1])
    selected = torch.randint(0, rows, (n, width, k), dtype=torch.int32)
    selected[0, :, 0] = 5  # cross-query duplicate
    selected[0, 0, 1] = -1
    lengths = torch.full((n, width), rows, dtype=torch.int32)
    lengths[0, 0] = 7  # preserve per-query causal boundary
    for request in (0, 2):
        tokens[request, hot] = 5
        device[locations[request, hot]] = host[host_locs[request, 5]]
        optional[request, 6] = locations[request, hot + 1]
        device[locations[request, hot + 1]] = host[host_locs[request, 6]]
    workspace = allocate_workspace(n, capacity, hot, "cpu")
    out = torch.empty_like(selected)
    args = dict(
        request_ids=request_ids,
        buffer_tokens=tokens,
        buffer_locs=locations,
        hot_ages=ages,
        epochs=epochs,
        device_buffer=device,
        staging_offset=n * (hot + tail),
        staging_stride=capacity,
        hot_size=hot,
        workspace=workspace,
        num_real_requests=torch.tensor([2], dtype=torch.int32),
        item_bytes=d * 8,
    )
    resolve(
        selections=selected,
        seq_lens=lengths,
        optional_locs=optional,
        host_locs=host_locs,
        host=host,
        output=out,
        **args,
    )
    assert (out[2] == -1).all()  # padded request cannot alter its state
    for batch, request in enumerate(request_ids[:2]):
        valid = (selected[batch] >= 0) & (selected[batch] < lengths[batch, :, None])
        torch.testing.assert_close(
            device[out[batch][valid].long()],
            host[host_locs[request, selected[batch][valid].long()]],
        )
        assert (out[batch][~valid] == -1).all()
        unique = selected[batch][valid].unique()
        expected_misses = sum(int(x) not in (5, 6) for x in unique)
        assert int((workspace["miss_src"][request] >= 0).sum()) == expected_misses
    before = device[out[0][out[0] >= 0].long()].clone()
    promote(**args)
    torch.testing.assert_close(device[out[0][out[0] >= 0].long()], before)
    # Arena can now be overwritten without damaging the promoted hot payload.
    device[n * (hot + tail) :].fill_(-777)
    for request in (0, 2):
        for slot, token in enumerate(tokens[request, :hot]):
            if token >= 0:
                torch.testing.assert_close(
                    device[locations[request, slot]], host[host_locs[request, token]]
                )
    assert epochs[1] == 0


def test_missing_host_backing_is_an_error_not_a_silent_invalid_selection():
    workspace = allocate_workspace(1, 1, 2, "cpu")
    with pytest.raises(RuntimeError, match="neither ready GPU nor host"):
        resolve(
            selections=torch.tensor([[[0]]], dtype=torch.int32),
            request_ids=torch.tensor([0]),
            seq_lens=torch.tensor([[1]]),
            optional_locs=torch.tensor([[-1]], dtype=torch.int32),
            host_locs=torch.tensor([[-1]]),
            buffer_tokens=torch.full((1, 3), -1, dtype=torch.int32),
            buffer_locs=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            hot_ages=torch.zeros(1, 2, dtype=torch.int64),
            epochs=torch.zeros(1, dtype=torch.int64),
            host=torch.zeros(1, 8),
            device_buffer=torch.zeros(4, 8),
            staging_offset=3,
            staging_stride=1,
            hot_size=2,
            output=torch.empty(1, 1, 1, dtype=torch.int32),
            workspace=workspace,
            num_real_requests=torch.tensor([1]),
            item_bytes=32,
        )


def test_residency_backup_before_reclaim_publish_after_completion_and_refill_fairness():
    free, log, events, visible = [24], [], [], {}

    def alloc(count):
        free[0] -= count
        return torch.arange(count)

    def retire(pages):
        log.append("retire")
        free[0] += len(pages)

    def copy(request, start, pages):
        event = SimpleNamespace(done=False)
        event.query = lambda: event.done
        event.synchronize = lambda: setattr(event, "done", True)
        events.append(event)
        return event

    manager = ResidencyManager(
        page_size=4,
        config=DHiSparseConfig(admission_reserve_pages=2, refill_pages_per_step=2),
        available=lambda: free[0],
        allocate=alloc,
        copy=copy,
        publish=lambda request, start, pages: visible.setdefault(request, []).extend(
            pages.tolist()
        ),
        unpublish=lambda request: visible.pop(request, None),
        backup=lambda state: log.append("backup"),
        retire=retire,
    )
    manager.register(2, 17, alloc(16), backed=False)
    manager.register(0, 12, alloc(8), backed=True)
    assert manager.reclaim(12)
    assert log == ["backup", "retire"]
    assert 2 not in visible and 0 in visible  # reclaim whole request; stop at deficit
    manager.refill()
    assert 2 not in visible  # pending bytes must not be visible
    assert free[0] == 8  # admission reserve remains free
    events[0].done = True
    manager.poll()
    assert len(visible[2]) == 8
    manager.release(0)
    manager.refill()
    manager.reclaim(24)  # drains pending copies before retirement
    assert free[0] == 24 and manager.reclaimable_rows == 0


def test_budget_prices_staging_once_per_live_group():
    layout = VerificationLayout(35, 8, 2048, 64, tuple(i % 4 for i in range(78)))
    assert layout.rows == 4 * 35 * 8 * 2048
    assert layout.rows * 576 < 1.3 * 1024**3
    with pytest.raises(ValueError, match="exceed the KV budget"):
        layout.owned_rows(4096, 576, 4096, 100_000)


def test_draft_ring_compact_tables_preserve_order_through_wrap():
    layout = DraftRingLayout(2048, 8, 64)
    reqs = torch.tensor([2, 1])
    prefixes = torch.tensor([100_000, 31])
    starts = torch.clamp(prefixes - layout.window, min=0) // 64 * 64
    visible = prefixes - starts
    table = torch.full((3, layout.rows), -1, dtype=torch.int32)
    layout.rebuild(table, reqs, prefixes, visible, 8)
    for row, rid in enumerate(reqs):
        count = int(visible[row]) + 8
        expected = layout.locations(rid, torch.arange(starts[row], prefixes[row] + 8))
        torch.testing.assert_close(table[rid, :count].long(), expected)
        assert expected.unique().numel() == count
        for offset in range(0, count - 64, 64):
            assert torch.equal(
                table[rid, offset : offset + 64], table[rid, offset] + torch.arange(64)
            )
    assert (table[0] == -1).all()


def test_packed_page_copies_include_every_value_and_scale_byte_without_adjacent_rows():
    from sglang.kernels.ops.kvcache.dhisparse import _copy_cpu

    page_bytes = 65 * 576
    source = torch.randint(0, 256, (3, page_bytes), dtype=torch.uint8)
    target = torch.full_like(source, 211)
    for src, dst in ((63, 64), (128, 7)):
        before = target.clone()
        _copy_cpu(source, target, src, dst, True)
        expected = before.clone()
        for offset, width in ((0, 576), (64 * 576, 8)):
            expected[
                dst // 64, offset + (dst % 64) * width : offset + (dst % 64 + 1) * width
            ] = source[
                src // 64, offset + (src % 64) * width : offset + (src % 64 + 1) * width
            ]
        torch.testing.assert_close(target, expected)


def test_adaptive_config_reserves_verification_independently_from_hot_capacity():
    from sglang.srt.mem_cache.sparsity import (
        parse_hisparse_config,
        resolve_dspark_device_buffer_size,
    )

    config = parse_hisparse_config(SimpleNamespace(hisparse_config=None))
    assert config.dhisparse.enabled
    config = resolve_dspark_device_buffer_size(
        config, raw_hisparse_config=None, verify_width=8, effective_top_k=512
    )
    assert config.device_buffer_size == 1024
    for raw in ('{"force_h2d_each_step": true}', '{"prefetcher": "previous"}'):
        with pytest.raises(ValueError, match="Adaptive DHiSparse"):
            parse_hisparse_config(SimpleNamespace(hisparse_config=raw))


def test_real_dsa_pool_registers_only_owned_bytes_and_accounts_shared_slab_once():
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

    pool = HiSparseDSATokenToKVPool(
        size=2048,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=2,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        dhisparse_layout=VerificationLayout(2, 4, 4, 64, (0, 0)),
        dhisparse_hot=64,
        dhisparse_context=256,
    )
    ptrs, lengths, _ = pool.get_contiguous_buf_infos()
    assert ptrs[0] + lengths[0] == ptrs[1]
    assert (
        ptrs[1] + lengths[1]
        <= pool.kv_buffer[0].data_ptr() + pool._dhisparse_slab.nbytes
    )
    assert pool.get_kv_size_bytes() == pool._dhisparse_slab.nbytes + sum(
        x.nbytes for x in pool.index_k_with_scale_buffer
    )
    assert pool.size < 2048
