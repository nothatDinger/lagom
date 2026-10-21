"""Full-KV checks for fused gather/commit, request ownership and graph replay."""

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse_spec import (
    allocate_hisparse_spec_states,
    initialize_hisparse_spec_state,
    load_cache_to_device_buffer_spec_mla,
)
from sglang.srt.mem_cache.hisparse_spec import HiSparseSpecLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Fused HiSparse serving requires CUDA.",
)

PAGE_BYTES = ((584 * 64 + 575) // 576) * 576


def item(cache, loc):
    page, offset = divmod(loc, 64)
    value = cache[page, offset * 576 : (offset + 1) * 576]
    scale = cache[page, 64 * 576 + offset * 8 : 64 * 576 + (offset + 1) * 8]
    return torch.cat((value, scale))


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
