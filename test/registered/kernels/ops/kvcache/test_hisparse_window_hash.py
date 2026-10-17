"""KV-content oracle tests for parallel whole-window hash/CLOCK planning."""

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
from sglang.kernels.ops.kvcache.hisparse_window_hash import (
    allocate_window_hash_states,
    initialize_window_hash_state,
    plan_cache_window_hash_mla,
)
from sglang.srt.mem_cache.hisparse_window_hash import HiSparseWindowHashLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Whole-window hash tests require NVIDIA CUDA.",
)


def _inputs(hot, width, k, num_req_slots=3):
    layout = HiSparseWindowHashLayout(hot, width, k)
    host_len = 2 * hot + 8
    host = (
        torch.arange(num_req_slots * host_len * 8, dtype=torch.float32)
        .view(-1, 1, 8)
        .pin_memory()
    )
    locs = torch.arange(
        num_req_slots * (hot + 1), device="cuda", dtype=torch.int32
    ).view(num_req_slots, -1)
    tokens = torch.full_like(locs, -1)
    tokens[:, :hot] = torch.arange(hot, device="cuda", dtype=torch.int32)
    device = (
        host[torch.arange(num_req_slots).view(-1, 1) * host_len + torch.arange(hot + 1)]
        .reshape(-1, 1, 8)
        .cuda()
    )
    args = {
        "top_k_tokens": torch.full((3, width, k), -1, device="cuda", dtype=torch.int32),
        "top_k_device_locs": torch.full(
            (3, width, k), -1, device="cuda", dtype=torch.int32
        ),
        "device_buffer_tokens": tokens,
        "device_buffer_locs": locs,
        "host_cache_locs": torch.arange(
            num_req_slots * host_len, device="cuda", dtype=torch.int64
        ).view(num_req_slots, -1),
        "host_cache": host,
        "device_buffer": device,
        "req_pool_indices": torch.tensor([2, 0, 999], device="cuda", dtype=torch.int64),
        "seq_lens": torch.full(
            (3 * width,), host_len, device="cuda", dtype=torch.int32
        ),
        "state": allocate_window_hash_states(
            layout, num_layers=1, num_req_slots=num_req_slots, device="cuda"
        )[0],
        "num_real_reqs": torch.tensor([2], device="cuda", dtype=torch.int32),
        "miss_src": torch.empty((3, width * k), device="cuda", dtype=torch.int64),
        "miss_dst": torch.empty((3, width * k), device="cuda", dtype=torch.int32),
        "miss_count": torch.zeros(3, device="cuda", dtype=torch.int32),
    }
    _initialize(args)
    return args


def _initialize(args):
    initialize_window_hash_state(
        device_buffer_tokens=args["device_buffer_tokens"],
        req_pool_indices=args["req_pool_indices"][:2],
        state=args["state"],
    )


def _copy(args):
    copy_cache_planned_mla(
        **{
            key: args[key]
            for key in (
                "miss_src",
                "miss_dst",
                "miss_count",
                "num_real_reqs",
                "host_cache",
                "device_buffer",
            )
        },
        item_size_bytes=32,
    )


def _check_contents(args):
    topk = args["top_k_tokens"].cpu()
    output = args["top_k_device_locs"].cpu()
    lengths = args["seq_lens"].view(3, -1).cpu()
    for bid, rid in enumerate([2, 0]):
        for step in range(topk.size(1)):
            for pos in range(topk.size(2)):
                token = int(topk[bid, step, pos])
                loc = int(output[bid, step, pos])
                if token < 0 or token >= lengths[bid, step]:
                    assert loc == -1
                else:
                    assert loc >= 0
                    src = int(args["host_cache_locs"][rid, token])
                    torch.testing.assert_close(
                        args["device_buffer"][loc].cpu(), args["host_cache"][src]
                    )
    assert (output[2] == -1).all()
    assert args["miss_count"][2] == 0


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_cross_step_dedup_row_validity_padding_and_warm_cache(length_dtype):
    args = _inputs(8, 4, 2)
    args["top_k_tokens"][0] = torch.tensor(
        [[1, 9], [9, 10], [10, 11], [11, -1]], device="cuda"
    )
    args["top_k_tokens"][1] = torch.tensor(
        [[3, 12], [12, 13], [13, 14], [14, 3]], device="cuda"
    )
    args["seq_lens"] = torch.tensor(
        [9, 11, 12, 13, 12, 14, 15, 16, 0, 0, 0, 0], dtype=length_dtype, device="cuda"
    )
    untouched = args["device_buffer_tokens"][1].clone()
    physical = args["device_buffer_locs"].clone()
    for expected in ([3, 3, 0], [0, 0, 0]):
        plan_cache_window_hash_mla(**args)
        _copy(args)
        _check_contents(args)
        assert args["miss_count"].tolist() == expected
        torch.testing.assert_close(args["device_buffer_tokens"][1], untouched)
        torch.testing.assert_close(args["device_buffer_locs"], physical)
        for bid in range(2):
            count = int(args["miss_count"][bid])
            assert args["miss_dst"][bid, :count].unique().numel() == count


def test_clock_fallback_across_multiple_commit_iterations():
    # 220 union hits protect both choices of 110 CLOCK partitions, forcing
    # fallback claims into partitions used by later (>512-miss) iterations.
    args = _inputs(1024, 4, 256)
    hits = torch.cat([torch.arange(110), torch.arange(600, 710)])
    misses = torch.arange(1024, 1624)
    union = torch.cat([hits, misses]).to(device="cuda", dtype=torch.int32)
    args["top_k_tokens"][:2].view(2, -1)[:, : union.numel()] = union
    for _ in range(4):
        args["device_buffer_tokens"][:, :1024] = torch.arange(
            1024, device="cuda", dtype=torch.int32
        )
        _initialize(args)
        plan_cache_window_hash_mla(**args)
        # This test checks the victim plan before copying, so refresh original
        # hit contents when resetting metadata between independent windows.
        for rid in (0, 2):
            src = args["host_cache_locs"][rid, :1024].cpu()
            args["device_buffer"][args["device_buffer_locs"][rid, :1024].long()] = args[
                "host_cache"
            ][src].cuda()
        _copy(args)
        _check_contents(args)
        assert args["miss_count"].tolist() == [600, 600, 0]
        for bid in range(2):
            assert args["miss_dst"][bid, :600].unique().numel() == 600


def test_graph_replay_force_reset_and_request_slot_reuse():
    args = _inputs(8, 2, 2)
    args["top_k_tokens"][:2] = torch.tensor([[9, 10], [10, 11]], device="cuda")
    # Compile before capture. Reset is explicit, matching force-H2D and admission.
    plan_cache_window_hash_mla(**args)
    _copy(args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan_cache_window_hash_mla(**args)
        _copy(args)
    output_ptr = args["top_k_device_locs"].data_ptr()
    for selected in ([[9, 10], [10, 11]], [[12, 13], [13, 14]]):
        args["top_k_tokens"][:2] = torch.tensor(selected, device="cuda")
        args["device_buffer_tokens"][[0, 2]] = -1
        _initialize(args)
        graph.replay()
        _check_contents(args)
        assert args["miss_count"].tolist() == [3, 3, 0]
        assert args["top_k_device_locs"].data_ptr() == output_ptr


def test_force_h2d_inside_graph_reloads_old_hits_and_deduplicates_window():
    args = _inputs(6, 3, 2)
    args["top_k_tokens"][:2] = torch.tensor([[1, 2], [2, 7], [7, 8]], device="cuda")
    physical = args["device_buffer_locs"].clone()

    def run():
        args["device_buffer_tokens"].index_fill_(0, args["req_pool_indices"][:2], -1)
        _initialize(args)
        plan_cache_window_hash_mla(**args)
        _copy(args)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
        _check_contents(args)
        assert args["miss_count"].tolist() == [4, 4, 0]
        torch.testing.assert_close(args["device_buffer_locs"], physical)


def test_clock_epoch_wrap_does_not_protect_ancient_references():
    args = _inputs(8, 2, 2)
    state = args["state"]
    epoch_mask = (1 << 23) - 1  # H=8 -> cursor bits=3, epoch bits=23.
    state.cache_policy[0, [0, 2]] = (epoch_mask - 1) << 3
    state.cache_policy[[1, 3]] = 1
    state.scratch_state[0, [9, 11]] = (1 << 31) - 2
    for selected in ([[9, 10], [10, 11]], [[12, 13], [13, 14]]):
        args["top_k_tokens"][:2] = torch.tensor(selected, device="cuda")
        plan_cache_window_hash_mla(**args)
        _copy(args)
        _check_contents(args)
        assert args["miss_count"].tolist() == [3, 3, 0]


def test_single_step_and_ragged_widths_share_persistent_workspace():
    args = _inputs(8, 4, 2)
    for width, first in [(1, 16), (4, 8), (2, 18), (4, 8)] * 3:
        args["top_k_tokens"] = torch.full(
            (3, width, 2), -1, device="cuda", dtype=torch.int32
        )
        args["top_k_tokens"][:2] = torch.arange(
            first, first + width * 2, device="cuda", dtype=torch.int32
        ).view(width, 2)
        args["top_k_device_locs"] = torch.full_like(args["top_k_tokens"], -1)
        args["seq_lens"] = torch.full(
            (3 * width,), 24, device="cuda", dtype=torch.int32
        )
        plan_cache_window_hash_mla(**args)
        _copy(args)
        _check_contents(args)


def test_degraded_cuckoo_hash_falls_back_to_resident_tokens():
    # Three keys with the same two buckets cannot fit into the cuckoo index.
    buckets = {}
    for token in range(512):
        primary = (token * 2654435761) & 15
        hashed = token ^ (token >> 16)
        hashed = (hashed * 0x7FEB352D) & 0xFFFFFFFF
        hashed ^= hashed >> 15
        hashed = (hashed * 0x846CA68B) & 0xFFFFFFFF
        hashed ^= hashed >> 16
        group = buckets.setdefault((primary, hashed & 15), [])
        group.append(token)
        if len(group) == 3:
            collision = group
            break
    args = _inputs(8, 2, 2)
    args["host_cache"] = (
        torch.arange(3 * 512 * 8, dtype=torch.float32).view(-1, 1, 8).pin_memory()
    )
    args["host_cache_locs"] = torch.arange(
        3 * 512, device="cuda", dtype=torch.int64
    ).view(3, 512)
    args["device_buffer_tokens"][:] = -1
    for rid in (0, 2):
        args["device_buffer_tokens"][rid, :3] = torch.tensor(collision, device="cuda")
        src = rid * 512 + torch.tensor(collision)
        dst = args["device_buffer_locs"][rid, :3].long()
        args["device_buffer"][dst] = args["host_cache"][src].cuda()
    args["top_k_tokens"][:2] = torch.tensor(
        [[collision[0], collision[1]], [collision[2], collision[0]]], device="cuda"
    )
    args["seq_lens"].fill_(512)
    _initialize(args)
    # Every request's generation has a high-bit degraded marker after overflow.
    assert (args["state"].scratch_state[0, [9, 11]] < 0).all()
    for _ in range(2):
        plan_cache_window_hash_mla(**args)
        _copy(args)
        _check_contents(args)
        assert args["miss_count"].tolist() == [0, 0, 0]


def test_dsv4_value_and_scale_copy_across_padded_pages():
    args = _inputs(8, 2, 2)
    page_size, value_bytes, scale_bytes = 64, 576, 8
    page_bytes = ((page_size * (value_bytes + scale_bytes) + 575) // 576) * 576
    host = torch.zeros((3, page_bytes), dtype=torch.uint8, pin_memory=True)
    for loc in range(144):
        page, offset = divmod(loc, page_size)
        host[page, offset * value_bytes : (offset + 1) * value_bytes] = (
            (torch.arange(value_bytes) + loc).remainder(256).byte()
        )
        scale_start = page_size * value_bytes + offset * scale_bytes
        host[page, scale_start : scale_start + scale_bytes] = (
            (torch.arange(scale_bytes) + loc + 17).remainder(256).byte()
        )
    args["host_cache"] = host
    args["device_buffer"] = torch.zeros(
        (2, page_bytes), dtype=torch.uint8, device="cuda"
    )
    args["host_cache_locs"] = torch.arange(144, dtype=torch.int64, device="cuda").view(
        3, 48
    )
    args["device_buffer_locs"] += 61
    args["device_buffer_tokens"][:] = -1
    args["top_k_tokens"][:2] = torch.tensor([[8, 9], [9, 10]], device="cuda")
    _initialize(args)
    physical = args["device_buffer_locs"].clone()
    plan_cache_window_hash_mla(**args)
    copy_cache_planned_mla(
        **{
            key: args[key]
            for key in (
                "miss_src",
                "miss_dst",
                "miss_count",
                "num_real_reqs",
                "host_cache",
                "device_buffer",
            )
        },
        item_size_bytes=value_bytes + scale_bytes,
        is_dsv4_layout=True,
    )
    torch.cuda.synchronize()
    actual = args["device_buffer"].cpu()

    def read(cache, loc):
        page, offset = divmod(loc, page_size)
        scale_start = page_size * value_bytes + offset * scale_bytes
        return torch.cat(
            [
                cache[page, offset * value_bytes : (offset + 1) * value_bytes],
                cache[page, scale_start : scale_start + scale_bytes],
            ]
        )

    for bid, rid in enumerate((2, 0)):
        for token, dst in zip(
            args["top_k_tokens"][bid].flatten().tolist(),
            args["top_k_device_locs"][bid].flatten().tolist(),
        ):
            src = int(args["host_cache_locs"][rid, token])
            torch.testing.assert_close(read(actual, dst), read(host, src))
    torch.testing.assert_close(args["device_buffer_locs"], physical)
    assert args["miss_count"].tolist() == [3, 3, 0]


def test_full_union_table_cold_replays_do_not_drop_valid_kv():
    # W6/K16 fills all 96 union slots across two gather CTAs. CAS losses must
    # not consume the allowance for probing distinct hash slots.
    args = _inputs(128, 6, 16)
    args["top_k_tokens"][:2] = torch.arange(
        128, 224, dtype=torch.int32, device="cuda"
    ).view(6, 16)
    initial_tokens = args["device_buffer_tokens"].clone()
    initial_data = args["device_buffer"].clone()

    def reset():
        args["device_buffer_tokens"].copy_(initial_tokens)
        args["device_buffer"].copy_(initial_data)
        args["top_k_device_locs"].fill_(-1)
        _initialize(args)

    def run():
        plan_cache_window_hash_mla(**args)
        _copy(args)

    run()  # Compile outside capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for trial in range(7):
        reset()
        if trial == 0:
            run()
        else:
            graph.replay()
        _check_contents(args)
        assert args["miss_count"].tolist() == [96, 96, 0]
        for b in (0, 1):
            assert args["top_k_device_locs"][b].unique().numel() == 96


def test_pipeline_timing_events_survive_multiple_captures_and_cold_replays(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from sglang.benchmark.hisparse_pr40670_pipeline import graph_sample_us

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[5] / "scripts/pd_test")
    )
    from bench_hisparse_pr40670_pipeline import capture

    buffers, graphs = [], []
    for value in (1.0, 3.0):
        data = torch.empty(1 << 18, device="cuda")
        reader = SimpleNamespace(
            reset=lambda data=data: data.zero_(),
            call=lambda data=data, value=value: data.add_(value),
        )
        buffers.append((data, value))
        graphs.append(capture(torch, reader, timed=True))
    for _ in range(6):
        for (data, value), (graph, start, end) in zip(buffers, graphs):
            graph.replay()
            assert graph_sample_us(start, end) > 0
            assert end.query()
            assert data[0].item() == value and data[-1].item() == value
