"""V4 paged KV: real refill + captured side streams + root-only C4 commit.

No model checkpoint is needed. This exercises the actual HiSparse CUDA kernels,
Triton metadata kernels and C4 commit copies, with byte-exact attention reads.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseDSparkWindow,
)
from sglang.srt.managers.hisparse_prefetcher import OasisKVPrefetcher
from sglang.srt.managers.oasiskv_v4_graph import OasisKVV4GraphState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PAGE = 64
VALUE = 576
SCALE = 8
PAGE_BYTES = ((PAGE * (VALUE + SCALE) + VALUE - 1) // VALUE) * VALUE


def _byte_indices(locs):
    byte = torch.arange(VALUE + SCALE, device=locs.device)
    base = (locs // PAGE)[..., None] * PAGE_BYTES
    offset = (locs % PAGE)[..., None]
    return base + torch.where(
        byte < VALUE,
        offset * VALUE + byte,
        PAGE * VALUE + offset * SCALE + byte - VALUE,
    )


def _read(cache, locs):
    return cache.view(-1)[_byte_indices(locs)]


def _write(cache, locs, payload):
    cache.view(-1)[_byte_indices(locs)] = payload.to(cache.device)


@pytest.mark.parametrize("hot,k,long_history", [(4, 2, 8), (512, 512, 700)])
def test_v4_graph_refill_pipeline_padding_reorder_c4_commit_and_recycle(
    hot, k, long_history
):
    layers, requests = 2, 3
    host_stride = ((long_history + 32 + PAGE - 1) // PAGE) * PAGE
    device_entries = requests * (hot + 1) + requests
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.device, c.compress_ratio, c.is_dsv4_hisparse = "cuda", 4, True
    c.prefetcher_name, c.prefetcher = "oasiskv", OasisKVPrefetcher(k)
    c.device_buffer_size, c.padded_buffer_size = hot, hot + 1
    c.top_k, c.swap_in_block_size, c.item_size_bytes = k, 256, VALUE + SCALE
    c.skip_io = False
    c.num_real_reqs = torch.zeros(1, dtype=torch.int32, device="cuda")
    c.req_to_device_buffer = torch.arange(requests * (hot + 1), device="cuda").view(
        requests, -1
    )
    c.req_device_buffer_token_locs = c.req_to_device_buffer.to(torch.int32).repeat(
        layers, 1, 1
    )
    c.req_device_buffer_tokens = torch.full(
        (layers, requests, hot + 1), -1, dtype=torch.int32, device="cuda"
    )
    c.lru_slots = torch.arange(hot, dtype=torch.int16, device="cuda").repeat(
        layers, requests, 1
    )
    c.req_to_host_pool = torch.full(
        (requests, host_stride), -1, dtype=torch.int64, device="cuda"
    )
    c.req_to_host_pool_allocated_len = torch.zeros(requests, dtype=torch.int64)
    c._oasiskv_select_stream, c._previous_prefetch_stream = (
        torch.cuda.Stream(),
        torch.cuda.Stream(),
    )
    c._oasiskv_ring = [
        [
            SimpleNamespace(
                predicted_entries=torch.empty(
                    requests, k, dtype=torch.int32, device="cuda"
                ),
                device_locs=torch.empty(requests, k, dtype=torch.int32, device="cuda"),
                miss_src=torch.empty(requests, k, dtype=torch.int64, device="cuda"),
                miss_dst=torch.empty(requests, k, dtype=torch.int32, device="cuda"),
                miss_count=torch.empty(requests, dtype=torch.int32, device="cuda"),
                planned_tokens=torch.empty(
                    requests, hot, dtype=torch.int32, device="cuda"
                ),
                planned_lru=torch.empty(
                    requests, hot, dtype=torch.int16, device="cuda"
                ),
                selection_event=torch.cuda.Event(),
                pending_event=None,
            )
        ]
        for _ in range(layers)
    ]
    host = [
        torch.zeros(
            requests * host_stride // PAGE,
            PAGE_BYTES,
            dtype=torch.uint8,
            pin_memory=True,
        )
        for _ in range(layers)
    ]
    device = [
        torch.zeros(
            (device_entries + PAGE - 1) // PAGE,
            PAGE_BYTES,
            dtype=torch.uint8,
            device="cuda",
        )
        for _ in range(layers)
    ]
    for layer in range(layers):
        locs = torch.arange(requests * host_stride)
        payload = (
            (locs[:, None] * 7 + torch.arange(VALUE + SCALE) + layer * 31) % 251
        ).byte()
        _write(host[layer], locs, payload)
    device_ptrs = torch.tensor(
        [x.data_ptr() for x in device], dtype=torch.uint64, device="cuda"
    )
    host_ptrs = torch.tensor(
        [x.data_ptr() for x in host], dtype=torch.uint64, device="cuda"
    )

    def alloc_host(_mapping, _lengths, req, pos, count):
        assert count == 1
        loc = torch.tensor([req * host_stride + pos], device="cuda")
        c.req_to_host_pool[req, pos] = loc[0]
        return loc

    c.mem_pool_host = SimpleNamespace(
        kv_buffer=host,
        device_ptrs=device_ptrs,
        data_ptrs=host_ptrs,
        alloc_paged_token_slots=alloc_host,
    )
    c.mem_pool_device = SimpleNamespace(
        kv_buffer=device,
        full_to_hisparse_device_index_mapping=torch.zeros(
            requests * host_stride, dtype=torch.int64, device="cuda"
        ),
    )
    c._dspark_scratch_valid_mapping = torch.zeros(
        requests * host_stride, dtype=torch.bool, device="cuda"
    )
    c._dspark_commit_done_event = torch.cuda.Event()
    c._has_pending_dspark_commit = False
    c._active_dspark_window = None
    state = c.oasiskv_graph_state = OasisKVV4GraphState(c)

    def capture(bs):
        fb = SimpleNamespace(
            req_pool_indices=torch.zeros(bs, dtype=torch.int64, device="cuda"),
            seq_lens=torch.zeros(bs, dtype=torch.int32, device="cuda"),
            oasiskv_draft_valid=torch.ones(bs, dtype=torch.bool, device="cuda"),
        )
        real = torch.full((bs, k), -1, dtype=torch.int32, device="cuda")
        predicted = torch.full_like(real, -1)
        output_locs = torch.full_like(real, -1)

        def run():
            normal_lens = (fb.seq_lens + 1) // 4
            reads = []
            for layer in range(layers):
                state.consume_layer(layer)
                # The native indexer supplies real/predicted logical C4 indices.
                swapped = c.swap_in_selected_pages_spec(
                    fb.req_pool_indices,
                    normal_lens,
                    real,
                    layer,
                    output_buffer=output_locs,
                )
                logical = fb.req_pool_indices[:, None] * host_stride + real.clamp_min(0)
                mapped = c.mem_pool_device.full_to_hisparse_device_index_mapping[
                    logical
                ].to(torch.int32)
                resolved = c.select_dspark_scratch_locs(logical, swapped, mapped)
                state.plan(
                    fb,
                    layer_id=layer,
                    predicted_c4_entries=predicted,
                    compressed_seq_lens=normal_lens,
                )
                # Snapshot every KV byte attention may read. A premature transfer
                # or stale real-query refill produces a different payload here.
                payload = _read(device[layer], resolved.clamp_min(0))
                valid = (real >= 0) & (
                    torch.arange(bs, device="cuda")[:, None] < c.num_real_reqs
                )
                reads.append(torch.where(valid[..., None], payload, 0))
                state.transfer(fb, layer_id=layer)
            state.finish_in_graph()
            return reads

        for _ in range(2):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            reads = run()
        return graph, fb, real, predicted, reads

    captures = {bs: capture(bs) for bs in (2, 3)}
    assert (c.req_device_buffer_tokens == -1).all()
    assert all(not buf.any() for buf in device)
    addresses = [
        x.data_ptr()
        for x in (c.req_to_host_pool, c.req_device_buffer_tokens, c.lru_slots)
    ]
    prefixes = {0: (hot - 1) * 4 + 2, 1: long_history * 4 + 2, 2: 0}

    def prefill(req):
        length = prefixes[req] // 4
        c.req_to_host_pool[req].fill_(-1)
        c.req_to_host_pool[req, :length] = torch.arange(
            req * host_stride, req * host_stride + length, device="cuda"
        )
        c.req_device_buffer_tokens[:, req].fill_(-1)
        c.lru_slots[:, req] = torch.arange(hot, dtype=torch.int16, device="cuda")
        indices = list(range(min(length, hot)))
        fixed = indices.copy()
        if length > hot:
            indices.append(length - 1)
            fixed.append(hot)
        for layer in range(layers):
            c.req_device_buffer_tokens[layer, req, fixed] = torch.tensor(
                indices, dtype=torch.int32, device="cuda"
            )
            _write(
                device[layer],
                c.req_to_device_buffer[req, fixed],
                _read(
                    host[layer],
                    req * host_stride + torch.tensor(indices, dtype=torch.int64),
                ),
            )

    prefill(0)
    prefill(1)
    saw_miss = False
    schedules = [
        (3, [0, 1]),
        (2, [1, 0]),
        (3, [1]),
        (3, [0, 1]),
        (2, [1, 0]),
        (3, [0]),
        (2, [0, 1]),
        (3, [1, 0]),
        (3, []),
        (2, [1, 0]),
    ]
    for step, (bucket, slots) in enumerate(schedules):
        if step == 5:
            state.drain_request(0)
            prefixes[0] = 2  # scheduler slot reused for a different request
            prefill(0)
        state.prepare_replay()
        graph, fb, real, predicted, reads = captures[bucket]
        state.replay_batch_size = bucket
        bs = len(slots)
        c.num_real_reqs.fill_(bs)
        fb.req_pool_indices.fill_(slots[0] if slots else 0)  # padding aliases a live ID
        fb.req_pool_indices[:bs] = torch.tensor(slots, device="cuda")
        fb.seq_lens.zero_()
        fb.seq_lens[:bs] = torch.tensor([prefixes[r] for r in slots], device="cuda")
        real.fill_(-1)
        predicted.fill_(-1)
        entries, offsets, positions, scratch = [], [], [], []
        expected = [[] for _ in range(layers)]
        for row, req in enumerate(slots):
            prefix = prefixes[req]
            normal, draft = (prefix + 1) // 4, (prefix + 2) // 4
            selected = (torch.arange(min(k, normal)) + step * 17) % max(1, normal)
            prediction = (torch.arange(min(k, draft)) + max(0, draft - k // 2)) % max(
                1, draft
            )
            if step == 1:
                # Selection must see this same step's real refill. A perfect
                # overlapping prediction cannot schedule those H2D copies again.
                prediction = selected.clone()
            real[row, : len(selected)] = selected.to("cuda", torch.int32)
            predicted[row, : len(prediction)] = prediction.to("cuda", torch.int32)
            for layer in range(layers):
                wanted = torch.zeros(k, VALUE + SCALE, dtype=torch.uint8)
                wanted[: len(selected)] = _read(
                    host[layer], req * host_stride + selected
                )
                expected[layer].append(wanted)
            if draft > prefix // 4:
                pos = prefix // 4
                logical = req * host_stride + pos
                physical = requests * (hot + 1) + req
                entries.append(logical)
                offsets.append(row)
                positions.append(pos)
                scratch.append(physical)
                for layer in range(layers):
                    _write(
                        device[layer],
                        torch.tensor([physical], device="cuda"),
                        _read(host[layer], torch.tensor([logical])),
                    )
        compressed = torch.tensor(entries, dtype=torch.int64, device="cuda")
        mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
        window = HiSparseDSparkWindow(
            compressed,
            torch.tensor(scratch, dtype=torch.int64, device="cuda"),
            mapping[compressed].clone(),
            offsets,
            slots,
            positions,
            [prefixes[r] for r in slots],
        )
        mapping[compressed] = window.device_locs
        c._dspark_scratch_valid_mapping[compressed] = True
        c._active_dspark_window = window if entries else None
        untouched = c.req_device_buffer_tokens[:, 2].clone()
        graph.replay()
        state.finish_replay(SimpleNamespace(req_pool_indices_cpu=torch.tensor(slots)))
        c.commit_dspark_verify_window(
            window, torch.ones(bs, dtype=torch.int32, device="cuda")
        )
        torch.cuda.synchronize()
        for layer in range(layers):
            if bs:
                torch.testing.assert_close(
                    reads[layer][:bs].cpu(), torch.stack(expected[layer])
                )
            assert not reads[layer][bs:].any()
            task = c._oasiskv_ring[layer][0]
            saw_miss |= bool(task.miss_count[:bs].any())
            if step == 1:
                assert not task.miss_count[:bs].any()
            assert not task.miss_count[bs:bucket].any()
            for row, req in enumerate(slots):
                assert (task.predicted_entries[row] < prefixes[req] // 4).all()
                ids = c.req_device_buffer_tokens[layer, req]
                valid = ids >= 0
                payload = _read(device[layer], c.req_to_device_buffer[req, valid]).cpu()
                torch.testing.assert_close(
                    payload,
                    _read(host[layer], req * host_stride + ids[valid].cpu().long()),
                )
                assert sorted(c.lru_slots[layer, req].cpu().tolist()) == list(
                    range(hot)
                )
        torch.testing.assert_close(c.req_device_buffer_tokens[:, 2], untouched)
        for req in slots:
            prefixes[req] += 1
            history = prefixes[req] // 4
            if history:
                latest_slot = min(history - 1, hot)
                assert (
                    c.req_device_buffer_tokens[:, req, latest_slot] == history - 1
                ).all()
            assert c.req_to_host_pool[req, history] == -1  # draft C4 discarded
        assert addresses == [
            x.data_ptr()
            for x in (c.req_to_host_pool, c.req_device_buffer_tokens, c.lru_slots)
        ]
    assert saw_miss


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
