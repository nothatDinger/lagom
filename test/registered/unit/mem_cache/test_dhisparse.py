from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.kvcache.dhisparse import allocate_workspace, promote, resolve
from sglang.srt.mem_cache.dhisparse import (
    DHiSparseConfig,
    ResidencyManager,
    VerificationLayout,
    allocate_payload_slab,
)


def test_shared_arenas_alias_only_across_non_live_layer_groups():
    slots = [0, 1, 2, 0, 1, 0]
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
        assert int((workspace["miss_src"][batch] >= 0).sum()) == expected_misses
    before = device[out[0][out[0] >= 0].long()].clone()
    promote(lru_slots=torch.arange(hot, dtype=torch.int32).repeat(n, 1), **args)
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
    assert config.device_buffer_size == 8192
    for width in (1, 7, 32):
        raw = '{"device_buffer_size": 64, "force_h2d_each_step": true}'
        cfg = parse_hisparse_config(SimpleNamespace(hisparse_config=raw))
        assert (
            resolve_dspark_device_buffer_size(
                cfg, raw_hisparse_config=raw, verify_width=width, effective_top_k=512
            ).device_buffer_size
            == 64
        )


def test_c4_pool_reserves_one_physical_arena_and_keeps_it_outside_allocatable_pages():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

    layout = VerificationLayout(2, 8, 64, 64, (0, 0, 0), request_slots=10)
    pool = HiSparseC4DevicePool(
        size=8192,
        page_size=64,
        dtype=torch.uint8,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=3,
        device="cpu",
        enable_memory_saver=False,
        dhisparse_layout=layout,
        dhisparse_hot=128,
        dhisparse_context=256,
    )
    owned = pool.size // 64 + 1
    assert pool.size < 8192
    assert pool._dhisparse_slab.shape == (3 * owned + 2 * 8, 65 * 576)
    assert (
        len(
            {
                view[offset // 64].data_ptr()
                for view, offset in zip(pool.kv_buffer, pool.dhisparse_staging_offsets)
            }
        )
        == 1
    )
    assert layout.slots == 10 and layout.rows == 1024


def test_extra_h2d_uses_an_unused_arena_row_and_preserves_victims_and_read_addresses():
    from sglang.kernels.ops.kvcache.dhisparse import reread

    def step(force, *, drop_planned_hit=False):
        # Slot 6 is active at batch row 0; PD reservation slots do not own arenas.
        request = 6
        host = torch.arange(64 * 8, dtype=torch.uint8).view(64, 8)
        payload = torch.full((10, 8), 251, dtype=torch.uint8)
        locs = torch.full((7, 3), -1, dtype=torch.int32)
        tokens = torch.full_like(locs, -1)
        locs[request] = torch.tensor([0, 1, 2])
        tokens[request] = torch.tensor([4, 3, 19])
        payload[:3] = host[[4, 3, 19]]
        optional = torch.full((7, 32), -1, dtype=torch.int32)
        optional[request, 18] = 3
        payload[3] = host[18]
        host_locs = torch.arange(32).repeat(7, 1)
        host_locs[request, 18:20] = -1  # optional and tail have no host backup
        ages = torch.zeros((7, 2), dtype=torch.long)
        epochs = torch.zeros(7, dtype=torch.long)
        workspace = allocate_workspace(1, 6, 2, "cpu", track_hits=True)
        counts = torch.zeros(5, dtype=torch.int32)
        args = dict(
            request_ids=torch.tensor([request]),
            buffer_tokens=tokens,
            buffer_locs=locs,
            hot_ages=ages,
            epochs=epochs,
            device_buffer=payload,
            staging_offset=4,
            staging_stride=6,
            hot_size=2,
            workspace=workspace,
            counts=counts,
            num_real_requests=torch.tensor([1]),
            item_bytes=8,
        )
        selected = torch.tensor([[[4, 8, 18], [4, 9, 19]]], dtype=torch.int32)
        out = torch.empty_like(selected)
        resolve(
            selections=selected,
            seq_lens=torch.tensor([[19, 20]]),
            optional_locs=optional,
            host_locs=host_locs,
            host=host,
            output=out,
            **args,
        )
        assert counts.tolist() == [2, 1, 0, 0, 0]
        if drop_planned_hit:
            # Simulate a copy kernel missing an elected row. Eligibility must
            # survive even though no destination is written.
            workspace["hit_src"].fill_(-1)
        if force:
            reread(
                request_ids=args["request_ids"],
                host=host,
                device_buffer=payload,
                workspace=workspace,
                num_real_requests=args["num_real_requests"],
                staging_offset=4,
                staging_stride=6,
                item_bytes=8,
                counts=counts,
            )
        torch.testing.assert_close(payload[out.long()], host[selected.long()])
        assert tokens[request].tolist() == [4, 3, 19]  # reader has not completed
        copied = int(force and not drop_planned_hit)
        assert counts.tolist() == [2, 1, 0, 0, copied]
        expected_hit = host[4] if copied else torch.full((8,), 251, dtype=torch.uint8)
        torch.testing.assert_close(payload[4], expected_hit)
        promote(lru_slots=torch.arange(2, dtype=torch.int32).repeat(7, 1), **args)
        # Selected resident 4 stays hot. Only unselected slot 1 can admit a miss.
        assert tokens[request].tolist() == [4, 8, 19]
        assert counts.tolist() == [2, 1, 0, 1, copied]
        return out, tokens, ages, counts[:4], payload[:4]

    for control, forced in zip(step(False), step(True)):
        torch.testing.assert_close(control, forced)
    step(True, drop_planned_hit=True)


def test_pd_admission_publishes_only_complete_backed_pages_after_copy():
    free = [16]
    visible, events = {}, []

    def alloc(count):
        free[0] -= count
        return torch.arange(count)

    class Event:
        done = False

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True

    def copy(request, start, pages):
        event = Event()
        events.append((request, start, len(pages), event))
        return event

    manager = ResidencyManager(
        page_size=4,
        config=DHiSparseConfig(admission_reserve_pages=1),
        available=lambda: free[0],
        allocate=alloc,
        copy=copy,
        publish=lambda rid, start, pages: visible.update({rid: len(pages)}),
        unpublish=lambda rid: visible.pop(rid, None),
        backup=lambda state: pytest.fail("already backed"),
        retire=lambda pages: free.__setitem__(0, free[0] + len(pages)),
    )
    for rid in (4, 9):
        manager.register(rid, 15, torch.empty(0, dtype=torch.long), backed=True)
        manager.populate_on_admission(rid)
    assert [(r, i, n) for r, i, n, _ in events] == [(4, 0, 12)]
    assert free[0] == 4 and not visible
    manager.poll()
    assert not visible
    events[0][3].done = True
    manager.poll()
    assert visible == {4: 12}
    assert not manager.reclaim(17) and visible == {4: 12}
    assert manager.reclaim(8) and not visible and free[0] == 16


def test_serving_attention_releases_the_shared_arena_after_the_last_read():
    from sglang.srt.managers.dhisparse_coordinator import (
        AdaptiveHiSparse,
        finish_dhisparse_attention,
    )

    # Execute the serving resolve/attention callback with real packed CPU pages.
    layout = VerificationLayout(1, 2, 2, 64, (0, 0), request_slots=4)
    _, views, offsets = allocate_payload_slab(
        owned_units=4,
        unit_shape=(65 * 576,),
        dtype=torch.uint8,
        device="cpu",
        layout=layout,
        rows_per_unit=64,
    )
    locs = torch.zeros(2, 4, 4, dtype=torch.int32)
    locs[:, 3] = torch.tensor([64, 65, 66, 67])
    host = torch.arange(65 * 576).remainder(251).to(torch.uint8).view(1, -1)
    host_locs = torch.arange(64).repeat(4, 1)
    pool = SimpleNamespace(
        kv_buffer=views,
        dhisparse_staging_offsets=offsets,
        layer_mapping=[(4, 0, None), (4, 1, None)],
    )
    c = SimpleNamespace(
        device_buffer_size=2,
        force_h2d_each_step=False,
        device="cpu",
        req_device_buffer_tokens=torch.full_like(locs, -1),
        req_device_buffer_token_locs=locs,
        req_to_host_pool=host_locs,
        lru_slots=torch.arange(2, dtype=torch.int32).repeat(2, 4, 1),
        num_real_reqs=torch.tensor([1]),
        is_dsv4_hisparse=True,
        item_size_bytes=584,
        mem_pool_host=SimpleNamespace(kv_buffer=[host, host]),
        mem_pool_device=pool,
    )
    a = AdaptiveHiSparse.__new__(AdaptiveHiSparse)
    c.adaptive = a
    a.c, a.layout, a.trace, a.pending = c, layout, None, {}
    a.ages = torch.zeros(2, 4, 2, dtype=torch.long)
    a.epochs = torch.zeros(2, 4, dtype=torch.long)
    a.workspaces = [allocate_workspace(1, 4, 2, "cpu")]
    a.io_counts = [a.workspaces[0]["empty_counts"]]
    a.outputs = [torch.empty(1, 4, dtype=torch.int32)]
    a.optional_locs = torch.full_like(host_locs, -1, dtype=torch.int32)
    selected = torch.tensor([[[3, 1], [3, 7]]], dtype=torch.int32)
    reqs, lens = torch.tensor([3]), torch.tensor([[4, 8]], dtype=torch.int32)
    result = torch.empty((2, 2), dtype=torch.int32)
    backend = SimpleNamespace(token_to_kv_pool=pool, hisparse_coordinator=c)

    @finish_dhisparse_attention
    def reader(self, q, k, v, layer, batch):
        assert (c.req_device_buffer_tokens[layer.layer_id] == -1).all()
        return result.clone()

    for layer in (0, 1):
        assert a.resolve(reqs, lens, selected, layer, output=result) is result
        assert result[0, 0] == result[1, 0]
        with pytest.raises(RuntimeError, match="attention reader"):
            a.resolve(reqs, lens, selected, 1 - layer)
        read = reader(backend, None, None, None, SimpleNamespace(layer_id=layer), None)
        assert torch.equal(read, result) and not a.pending
        assert c.req_device_buffer_tokens[layer, 3, :2].tolist() == [3, 1]
    assert (a.epochs[:, :3] == 0).all()


def test_pd_pool_constructor_uses_active_batch_for_the_arena_and_transfer_slots_for_maps(
    monkeypatch,
):
    from sglang.srt.mem_cache import kv_cache_configurator as mod

    cfg = SimpleNamespace(
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=3,
        max_speculative_num_draft_tokens=6,
        hisparse_config='{"device_buffer_size": 128}',
    )
    kvc = SimpleNamespace(
        server_args=cfg,
        pool_page_size=256,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
        layer_info=SimpleNamespace(start_layer=2, end_layer=5, num_effective_layers=3),
        model_config=SimpleNamespace(
            compress_ratios=[4, 4, 4, 128, 4],
            hf_text_config=SimpleNamespace(index_topk=512),
            window_size=128,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
        ),
        kv_cache_dtype=torch.uint8,
        device="cpu",
    )
    reqs = SimpleNamespace(max_context_len=32768, req_to_token=torch.empty(9, 1))
    monkeypatch.setattr(
        mod, "get_memory", lambda: SimpleNamespace(enable_hisparse=True)
    )
    monkeypatch.setattr(
        mod,
        "get_exec",
        lambda: SimpleNamespace(features=SimpleNamespace(enable_memory_saver=False)),
    )
    monkeypatch.setattr(mod, "get_schedule", lambda: SimpleNamespace(page_size=256))
    monkeypatch.setattr(mod, "max_speculative_num_draft_tokens", lambda: 6)
    monkeypatch.setattr(mod, "DeepSeekV4TokenToKVPool", lambda **kwargs: kwargs)
    from types import MethodType

    kvc._dsv4_verify_pool_kwargs = MethodType(
        mod.KVCacheConfigurator._dsv4_verify_pool_kwargs, kvc
    )
    kwargs = mod.KVCacheConfigurator._build_dsv4_kv_pool(
        kvc,
        max_running_requests=3,
        swa_max_total_num_tokens=32_768,
        c4_max_total_num_tokens=65_536,
        c128_max_total_num_tokens=8_192,
        c4_state_pool_size=512,
        c128_state_pool_size=16,
        c4_state_dtype=None,
        c128_state_dtype=None,
        req_to_token_pool=reqs,
    )
    payload = kwargs["dhisparse_pool_kwargs"]
    assert payload["dhisparse_hot"] == 128 and payload["dhisparse_context"] == 8192 + 64
    layout = payload["dhisparse_layout"]
    assert (
        layout.rows == 3 * 6 * 512
        and layout.slots == 9
        and layout.layer_slots == (0, 0)
    )


def test_automatic_concurrency_fits_both_resident_rows_and_verification_storage():
    from dataclasses import replace

    layout = VerificationLayout(
        512, 6, 512, 64, (0, 0, 0), request_slots=522, reserve_pages=4
    )
    args = (100_000, layout.packed_row_bytes, 2048, 8192)
    n = layout.fitting_requests(*args)
    assert 1 < n < layout.num_requests
    fitting = replace(layout, num_requests=n, request_slots=n + 10)
    assert fitting.owned_rows(*args) >= n * (2048 + 64) + 4 * 64
    with pytest.raises(ValueError, match="exceed the KV budget"):
        replace(fitting, num_requests=n + 1, request_slots=n + 11).owned_rows(*args)


@pytest.mark.parametrize("fail_backup", [False, True])
def test_local_admission_owns_complete_imported_pages_and_recovers_failed_copy(
    monkeypatch, fail_backup
):
    from unittest.mock import Mock

    from sglang.kernels.ops.kvcache import hisparse
    from sglang.srt.managers.dhisparse_coordinator import AdaptiveHiSparse
    from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin

    class Host(HiSparseHostPoolMixin):
        page_size = 4
        data_ptrs = "host"

        def alloc_page(self, count):
            return torch.arange(count * 4)

        free = Mock()

    rid = 1
    mapping = torch.arange(4, 13)
    original = mapping.clone()
    dpool = SimpleNamespace(
        data_ptrs="device",
        full_to_hisparse_device_index_mapping=mapping,
        translate_loc_from_full_to_compressed=lambda x: x,
        translate_loc_to_hisparse_device=lambda x: mapping[x],
    )
    required = torch.arange(16, 24)
    raw = SimpleNamespace(
        available_size=lambda: 24,
        alloc=lambda size: required,
        free=Mock(),
        is_not_in_free_group=False,
    )
    host = Host()
    c = SimpleNamespace(
        page_size=4,
        device="cpu",
        mem_pool_device=dpool,
        device_buffer_size=4,
        padded_buffer_size=8,
        mem_pool_host=host,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(9).repeat(2, 1)),
        req_to_device_buffer=torch.zeros(2, 8, dtype=torch.long),
        req_device_buffer_size=torch.zeros(2, dtype=torch.long),
        req_device_buffer_token_locs=torch.full((1, 2, 8), -1, dtype=torch.int32),
        req_device_buffer_tokens=torch.full((1, 2, 8), -1, dtype=torch.int32),
        req_to_host_pool=torch.full((2, 12), -1, dtype=torch.long),
        req_to_host_pool_allocated_len=torch.zeros(2, dtype=torch.long),
        decode_producer_stream=Mock(),
        write_staging_stream=Mock(),
        ack_staging_queue=[],
    )
    a = AdaptiveHiSparse.__new__(AdaptiveHiSparse)
    a.c, a.raw_allocator, a.config = c, raw, DHiSparseConfig()
    a.optional_locs = torch.full((2, 12), -1, dtype=torch.int32)
    a.device_module, a.prepared_requests = Mock(), set()
    gpu_bytes = torch.arange(32) + 100
    host_bytes = torch.full((12,), -1)

    def transfer(*, src_ptrs, dst_ptrs, src_indices, dst_indices):
        assert src_ptrs == "device"
        if fail_backup and dst_ptrs == "host":
            raise RuntimeError("transfer failed")
        target = host_bytes if dst_ptrs == "host" else gpu_bytes
        target[dst_indices] = gpu_bytes[src_indices].clone()

    monkeypatch.setattr(hisparse, "transfer_cache_dsv4_mla", transfer)
    a.residency = ResidencyManager(
        page_size=4,
        config=a.config,
        available=raw.available_size,
        allocate=raw.alloc,
        copy=Mock(),
        publish=a._publish,
        unpublish=a._unpublish,
        backup=a._backup_history,
        retire=a._retire,
    )
    req = SimpleNamespace(req_pool_idx=rid, extend_range=SimpleNamespace(end=9))
    if fail_backup:
        with pytest.raises(RuntimeError, match="transfer failed"):
            a.admit_resident(req)
        assert torch.equal(mapping, original) and not a.residency.requests
        assert (
            c.req_to_host_pool_allocated_len[rid] == 0
            and (c.req_to_host_pool[rid] == -1).all()
        )
        assert torch.equal(raw.free.call_args.args[0], required)
        assert host.free.call_count == 1 and not c.ack_staging_queue
    else:
        assert a.admit_resident(req) and c.ack_staging_queue
        assert a.optional_locs[rid].tolist() == original[:8].tolist() + [-1] * 4
        assert not a.residency.requests[rid].backed and (host_bytes[:8] == -1).all()
        assert host_bytes[8] == gpu_bytes[required[0]] == gpu_bytes[required[4]] == 112
        assert c.req_device_buffer_tokens[0, rid].tolist() == [
            8,
            -1,
            -1,
            -1,
            8,
            -1,
            -1,
            -1,
        ]
        a._backup_history(a.residency.requests[rid])
        assert torch.equal(host_bytes[:8], gpu_bytes[original[:8]])
    assert raw.is_not_in_free_group is False
