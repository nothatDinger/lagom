"""Full-KV checks for fused gather/commit, request ownership and graph replay."""

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse_spec import (
    allocate_hisparse_spec_states,
    initialize_hisparse_spec_state,
    load_cache_to_device_buffer_spec_mla,
    lookup_hisparse_spec_resident,
)
from sglang.srt.mem_cache.hisparse_spec import HiSparseSpecLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Fused HiSparse serving requires CUDA.",
)

PAGE_BYTES = ((584 * 64 + 575) // 576) * 576


@pytest.mark.parametrize("degraded", [False, True])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_lagom_resident_lookup_is_read_only_and_tracks_rotating_locations(
    degraded, graph_replay
):
    hot, slots, steps, top_k = 128, 3, 2, 16
    locs = torch.arange(slots * 192, dtype=torch.int32, device="cuda").view(
        1, slots, 192
    )
    tokens = torch.full_like(locs, -1)
    state = allocate_hisparse_spec_states(
        HiSparseSpecLayout(hot, steps, top_k),
        device_buffer_locs=locs,
        scratch_offset=131,
    )[0]
    reqs = torch.tensor([2, 0, 1], dtype=torch.int64, device="cuda")
    real = torch.tensor([2], dtype=torch.int32, device="cuda")
    selected = (
        torch.tensor(
            [128, 384, 640, 896, 5, 999, -1, 1000, 777, 128, 384, 5, 999, -1, 900, 901],
            dtype=torch.int32,
            device="cuda",
        )
        .view(1, 1, top_k)
        .expand(slots, steps, top_k)
        .contiguous()
    )
    lengths = torch.full((slots * steps,), 1000, dtype=torch.int32, device="cuda")
    output = torch.empty_like(selected)

    def run():
        lookup_hisparse_spec_resident(
            top_k_tokens=selected,
            device_buffer_tokens=tokens[0, :, :131],
            device_buffer_locs=locs[0, :, :131],
            req_pool_indices=reqs,
            seq_lens=lengths,
            state=state,
            num_real_reqs=real,
            output=output,
        )

    run()  # Compile before capture; empty cache yields only misses.
    graph = None
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    for trial in range(3):
        tokens.fill_(-1)
        # Deliberate primary-hash collisions, canonical hits, active-tail hits.
        tokens[0, :, :4] = torch.tensor([128, 384, 640, 896], device="cuda")
        tokens[0, :, 5] = 5
        tokens[0, :, 128] = 999
        locs.copy_(torch.roll(locs, shifts=1, dims=2))
        initialize_hisparse_spec_state(
            device_buffer_tokens=tokens[0], req_pool_indices=reqs, state=state
        )
        if degraded:
            state.cache_index.fill_(-1)
            state.scratch_state[0, 3 * slots : 4 * slots] = -(1 << 31)
        before = [
            x.clone()
            for x in (
                tokens,
                locs,
                state.cache_index,
                state.cache_policy,
                state.scratch_state,
            )
        ]
        run() if graph is None else graph.replay()
        torch.cuda.synchronize()
        expected = torch.full_like(output, -1)
        for bid, rid in enumerate([2, 0]):
            mapping = {
                int(t): int(l)
                for t, l in zip(tokens[0, rid, :131].cpu(), locs[0, rid, :131].cpu())
                if t >= 0
            }
            expected[bid] = torch.tensor(
                [
                    [mapping.get(int(t), -1) if 0 <= t < 1000 else -1 for t in row]
                    for row in selected[bid].cpu()
                ],
                dtype=torch.int32,
                device="cuda",
            )
        torch.testing.assert_close(output, expected)
        for actual, snapshot in zip(
            (tokens, locs, state.cache_index, state.cache_policy, state.scratch_state),
            before,
        ):
            torch.testing.assert_close(actual, snapshot)


def item(cache, loc):
    page, offset = divmod(loc, 64)
    value = cache[page, offset * 576 : (offset + 1) * 576]
    scale = cache[page, 64 * 576 + offset * 8 : 64 * 576 + (offset + 1) * 8]
    return torch.cat((value, scale))


@pytest.mark.parametrize("decode_width,prediction_steps", [(8, 1), (2, 9)])
def test_lagom_preserved_workspace_alternates_graph_verify_and_prefetch(
    decode_width, prediction_steps
):
    from types import SimpleNamespace

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    hot, top_k, slots, host_rows = 256, 16, 2, 1024
    full_width = decode_width + prediction_steps
    active = hot + 1 + (full_width + 3) // 4
    locs = torch.arange(slots * (hot + 64), dtype=torch.int32, device="cuda").view(
        1, slots, hot + 64
    )
    tokens = torch.full_like(locs, -1)
    state = allocate_hisparse_spec_states(
        HiSparseSpecLayout(hot, max(decode_width, prediction_steps), top_k),
        device_buffer_locs=locs,
        scratch_offset=active,
    )[0]
    host = torch.empty(
        (slots * host_rows // 64, PAGE_BYTES), dtype=torch.uint8, pin_memory=True
    )
    host.copy_(
        torch.arange(host.numel(), dtype=torch.int64).remainder(251).view_as(host)
    )
    device = torch.zeros(
        (slots * (hot + 64) // 64, PAGE_BYTES), dtype=torch.uint8, device="cuda"
    )
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c._lagom_prediction_steps = prediction_steps
    c._speculative_verify_width = full_width
    c._spec_states, c._pr_verify_layout = [state], {"active_slots": active}
    c.force_h2d_each_step, c.skip_io, c.top_k, c.item_size_bytes = (
        False,
        False,
        top_k,
        584,
    )
    c.req_device_buffer_tokens, c.req_device_buffer_token_locs = tokens, locs
    c.req_to_host_pool = torch.arange(
        slots * host_rows, dtype=torch.int64, device="cuda"
    ).view(slots, host_rows)
    c.mem_pool_host, c.mem_pool_device = (
        SimpleNamespace(kv_buffer=[host]),
        SimpleNamespace(kv_buffer=[device]),
    )
    c.num_real_reqs = torch.tensor([slots], dtype=torch.int32, device="cuda")
    reqs = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    lengths = torch.full((slots,), host_rows, dtype=torch.int32, device="cuda")
    real_tokens = torch.full(
        (slots * decode_width, top_k), -1, dtype=torch.int32, device="cuda"
    )
    real_output = torch.empty_like(real_tokens)

    def run(width, selected, output):
        c._run_swap_in_verify_window(
            req_pool_indices=reqs,
            compressed_seq_lens=lengths,
            top_k_result=selected,
            layer_id=0,
            verify_width=width,
            output_buffer=output,
        )

    run(decode_width, real_tokens, real_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(decode_width, real_tokens, real_output)
    for trial in range(5):
        for width in (decode_width, prediction_steps, 1):
            selected = (
                torch.arange(slots * width * top_k, dtype=torch.int32, device="cuda")
                .remainder(96)
                .add_(hot + trial * 96)
                .view(slots * width, top_k)
            )
            if width == decode_width:
                real_tokens.copy_(selected)
                graph.replay()
                output = real_output
            else:
                output = torch.empty_like(selected)
                run(width, selected, output)
            torch.cuda.synchronize()
            device_cpu, out_cpu, selected_cpu = (
                device.cpu(),
                output.cpu(),
                selected.cpu(),
            )
            for bid, rid in enumerate([1, 0]):
                for token, location in zip(
                    selected_cpu.view(slots, -1)[bid].tolist(),
                    out_cpu.view(slots, -1)[bid].tolist(),
                ):
                    assert location >= 0
                    torch.testing.assert_close(
                        item(device_cpu, location), item(host, rid * host_rows + token)
                    )


@pytest.mark.parametrize("unique_misses", [48, 96])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_forced_cold_windows_preserve_kv_and_owned_allocation(
    unique_misses, graph_replay
):
    # 96 misses overflow the 61 request-owned scratch slots: commit must perform
    # direct H2D as well as rotate scratch. 48 exercises cross-step deduplication.
    hot, width, top_k, slots, host_rows = 128, 6, 16, 4, 512
    locs = torch.arange(slots * (hot + 64), dtype=torch.int32, device="cuda").view(
        1, slots, hot + 64
    )
    initial_locs = locs.clone()
    tokens = torch.full_like(locs, -1)
    tokens[:, 1] = 71  # Request slot 1 is padding and must retain its state.
    state = allocate_hisparse_spec_states(
        HiSparseSpecLayout(hot, width, top_k),
        device_buffer_locs=locs,
        scratch_offset=131,
    )[0]
    reqs = torch.tensor([2, 0, 3, 1], dtype=torch.int64, device="cuda")
    host = torch.empty(
        (slots * host_rows // 64, PAGE_BYTES), dtype=torch.uint8, pin_memory=True
    )
    host.copy_(
        torch.arange(host.numel(), dtype=torch.int64).remainder(251).view_as(host)
    )
    device = torch.full(
        (slots * (hot + 64) // 64, PAGE_BYTES), 165, dtype=torch.uint8, device="cuda"
    )
    padded_device_before = device[3:6].clone()  # Original allocation of padded rid=1.
    host_locs = torch.arange(slots * host_rows, dtype=torch.int64, device="cuda").view(
        slots, host_rows
    )
    selected = torch.empty((slots, width, top_k), dtype=torch.int32, device="cuda")
    output = torch.full_like(selected, -7)
    lengths = torch.full((slots * width,), host_rows, dtype=torch.int32, device="cuda")
    real = torch.tensor([3], dtype=torch.int32, device="cuda")

    def run():
        tokens[:, [2, 0, 3]] = -1
        initialize_hisparse_spec_state(
            device_buffer_tokens=tokens[0], req_pool_indices=reqs[:3], state=state
        )
        load_cache_to_device_buffer_spec_mla(
            top_k_tokens=selected,
            device_buffer_tokens=tokens[0, :, :131],
            device_buffer_locs=locs[0, :, :131],
            host_cache_locs=host_locs,
            host_cache=host,
            device_buffer=device,
            top_k_device_locs=output,
            req_pool_indices=reqs,
            seq_lens=lengths,
            state=state,
            num_real_reqs=real,
            transfer_kind="dsv4_paged",
        )

    def select(trial):
        values = (
            torch.arange(width * top_k, dtype=torch.int32, device="cuda")
            % unique_misses
        )
        values += hot + (trial % 2) * 96
        selected.copy_(values.view(1, width, top_k).expand(slots, -1, -1))
        selected[0, 0, 0] = -1
        selected[1, -1, -1] = host_rows  # Out-of-range selections must not copy.

    select(0)
    run()  # Compile before capture.
    graph = None
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    output_ptr = output.data_ptr()
    for trial in range(8):
        select(trial)
        run() if graph is None else graph.replay()
        torch.cuda.synchronize()
        selected_cpu, output_cpu, device_cpu = (
            selected.cpu(),
            output.cpu(),
            device.cpu(),
        )
        for bid, rid in enumerate([2, 0, 3]):
            token_to_loc = {}
            for token, loc in zip(
                selected_cpu[bid].flatten().tolist(), output_cpu[bid].flatten().tolist()
            ):
                if token < 0 or token >= host_rows:
                    assert loc == -1
                    continue
                assert rid * (hot + 64) <= loc < (rid + 1) * (hot + 64)
                assert loc == token_to_loc.setdefault(token, loc)
                torch.testing.assert_close(
                    item(device_cpu, loc), item(host, rid * host_rows + token)
                )
        for rid in range(slots):
            assert sorted(locs[0, rid].cpu().tolist()) == sorted(
                initial_locs[0, rid].cpu().tolist()
            )
        torch.testing.assert_close(locs[:, :, hot:131], initial_locs[:, :, hot:131])
        torch.testing.assert_close(device[3:6], padded_device_before)
        assert (tokens[:, 1] == 71).all()
        assert (output_cpu[3] == 0).all()  # Pinned PR padded-row sentinel.
        assert output.data_ptr() == output_ptr
