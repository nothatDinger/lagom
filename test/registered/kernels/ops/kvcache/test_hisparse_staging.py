"""Packed KV bytes, oversized verification unions and N=21 cross-graph reuse."""
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse_spec import (
    allocate_hisparse_spec_states, initialize_hisparse_spec_state,
)
from sglang.kernels.ops.kvcache.hisparse_staging import HiSparseVerificationStaging
from sglang.srt.managers.lagom_graph import LagomPrefetchGraphs
from sglang.srt.mem_cache.hisparse_spec import HiSparseSpecLayout
from sglang.srt.mem_cache.hisparse_staging import HiSparseStagingLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="DeepSeek-V4 packed staging requires CUDA",
)
PAGE_BYTES = ((584 * 64 + 575) // 576) * 576


def row_bytes(buffer, locations):
    locations = locations.clamp_min(0).long()
    pages, rows = locations // 64, locations % 64
    body = pages[..., None] * PAGE_BYTES + rows[..., None] * 576
    scales = pages[..., None] * PAGE_BYTES + 64 * 576 + rows[..., None] * 8
    body = body + torch.arange(576, device=locations.device)
    scales = scales + torch.arange(8, device=locations.device)
    return torch.cat((buffer.flatten()[body], buffer.flatten()[scales]), dim=-1)


@pytest.mark.parametrize("p", [0, 1, 2, 4, 7])
@pytest.mark.parametrize("captured", [False, True])
def test_n21_forecasts_survive_a_full_verify_and_request_slot_reuse(p, captured):
    layers, requests, hot, top_k, decode, host_rows = 21, 3, 8, 16, 3, 1024
    width, stride = decode + p, hot + 64
    pool = SimpleNamespace(
        device="cuda", create_buffer=lambda *, num_pages: torch.empty(
            (num_pages, PAGE_BYTES), dtype=torch.uint8, device="cuda"),
    )
    staging = HiSparseVerificationStaging(
        layout=HiSparseStagingLayout(requests, width, top_k, layers, p, layers if p else None),
        pool=pool, hot_slots=hot,
    )
    locs = torch.arange(requests * stride, dtype=torch.int32, device="cuda").reshape(
        1, requests, stride).repeat(layers, 1, 1)
    tokens = torch.full_like(locs, -1)
    states = allocate_hisparse_spec_states(
        HiSparseSpecLayout(hot, width, top_k), device_buffer_locs=locs, scratch_offset=hot + 3,
    )
    reqs = torch.tensor([2, 0, 1], dtype=torch.int64, device="cuda")
    live = torch.tensor([2], dtype=torch.int32, device="cuda")
    caches = [pool.create_buffer(num_pages=(requests * stride + 63) // 64 + 1) for _ in range(layers)]
    hosts = []
    for layer in range(layers):
        host = torch.empty((requests * host_rows // 64, PAGE_BYTES), dtype=torch.uint8, pin_memory=True)
        host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8,
                                 generator=torch.Generator().manual_seed(42 + layer)))
        hosts.append(host)
        initialize_hisparse_spec_state(
            device_buffer_tokens=tokens[layer], req_pool_indices=reqs, state=states[layer],
        )
    host_locs = torch.arange(requests * host_rows, dtype=torch.int64, device="cuda").reshape(requests, host_rows)
    prefix = torch.full((requests,), host_rows * 4, dtype=torch.int64, device="cuda")
    lengths = torch.full((requests, width), host_rows, dtype=torch.int32, device="cuda")
    selected = torch.empty((requests, width, top_k), dtype=torch.int32, device="cuda")
    output = torch.empty_like(selected)
    writer = torch.full_like(selected, -1)
    samples = torch.empty((layers, requests, decode, top_k, 584), dtype=torch.uint8, device="cuda")
    side = LagomPrefetchGraphs(torch.cuda.Stream()) if p else None

    def inputs(current, future):
        selected[:, :decode] = (
            torch.arange(decode * top_k, device="cuda").reshape(1, decode, top_k) + current
        )
        if p:
            selected[:, decode:] = (
                torch.arange(p * top_k, device="cuda").reshape(1, p, top_k) + future
            )

    def step(capturing=False):
        for layer in range(layers):
            if side:
                side.consume_layer(layer)
            staging.resolve(
                layer_id=layer, req_pool_indices=reqs, source_prefix_lens=prefix,
                compressed_seq_lens=lengths, top_k_result=selected, window_locs=writer,
                output_buffer=output, num_real_reqs=live, decode_width=decode,
                host_locs=host_locs, host_cache=hosts[layer], device_cache=caches[layer],
                hot_tokens=tokens[layer, :, :hot + 3], hot_locs=locs[layer, :, :hot + 3],
                state=states[layer],
            )
            # Consumers read before release permits the next-round overwrite.
            samples[layer].copy_(row_bytes(staging.arena, output[:, :decode]))
            bs, transfer = staging.attention_finished(layer)
            if side:
                side.plan_ready(layer, bs, transfer, capturing=capturing)

    inputs(100, 300)
    step()
    torch.cuda.synchronize()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(capturing=True)
        if side:
            side.capture(requests)
    for trial in range(3):
        inputs(300 + 200 * trial, 500 + 200 * trial)
        if graph:
            graph.replay()
            if side:
                side.replay(requests)
        else:
            step()
        torch.cuda.synchronize()
        source = reqs[:2, None, None] * host_rows + selected[:2, :decode]
        for layer in range(layers):
            torch.testing.assert_close(samples[layer, :2].cpu(), row_bytes(hosts[layer], source.cpu()))
        assert (output[2] == -1).all()
        if p:
            assert (staging.plan.kinds[:2] == 3).any()
            assert len({staging.arena.data_ptr(), *(x.data_ptr() for x in staging.forecasts)}) == layers + 1
            assert not torch.isin(tokens[:, :, :hot], selected[:2, decode:].flatten()).any()
    if side:
        side.synchronize()
        staging.reset_request(2)
        for layer in range(layers):
            tokens[layer, 2].fill_(-1)
            initialize_hisparse_spec_state(
                device_buffer_tokens=tokens[layer], req_pool_indices=reqs[:1], state=states[layer],
            )
            hosts[layer][2 * (host_rows // 64):].zero_()
            assert (staging.forecast_indices[layer][2] == -1).all()
        inputs(900, 100)
        step()
        torch.cuda.synchronize()
        assert (samples[:, 0] == 0).all()
        side.synchronize()


def test_current_c4_writer_rows_are_exact_and_cannot_enter_hot_cache():
    hot, width, top_k = 8, 3, 16
    locs = torch.arange(hot + 64, dtype=torch.int32, device="cuda").reshape(1, 1, -1)
    tokens = torch.full_like(locs, -1)
    state = allocate_hisparse_spec_states(
        HiSparseSpecLayout(hot, width, top_k), device_buffer_locs=locs, scratch_offset=hot + 2,
    )[0]
    reqs = torch.zeros(1, dtype=torch.int64, device="cuda")
    initialize_hisparse_spec_state(device_buffer_tokens=tokens[0], req_pool_indices=reqs, state=state)
    pool = SimpleNamespace(
        device="cuda", create_buffer=lambda *, num_pages: torch.empty(
            (num_pages, PAGE_BYTES), dtype=torch.uint8, device="cuda"),
    )
    staging = HiSparseVerificationStaging(
        layout=HiSparseStagingLayout(1, width, top_k, 1), pool=pool, hot_slots=hot,
    )
    cache = pool.create_buffer(num_pages=2)
    cache.fill_(123)
    host = torch.zeros((2, PAGE_BYTES), dtype=torch.uint8, pin_memory=True)
    selected = torch.full((1, width, top_k), -1, dtype=torch.int32, device="cuda")
    selected[0, -1, 0] = 64
    writer = torch.full_like(selected, -1)
    writer[0, -1, 0] = 37
    output = torch.empty_like(selected)
    staging.resolve(
        layer_id=0, req_pool_indices=reqs,
        source_prefix_lens=torch.tensor([256], device="cuda"),
        compressed_seq_lens=torch.tensor([[64, 64, 65]], dtype=torch.int32, device="cuda"),
        top_k_result=selected, window_locs=writer, output_buffer=output,
        num_real_reqs=torch.ones(1, dtype=torch.int32, device="cuda"), decode_width=width,
        host_locs=torch.arange(128, device="cuda").reshape(1, 128),
        host_cache=host, device_cache=cache,
        hot_tokens=tokens[0, :, :hot + 2], hot_locs=locs[0, :, :hot + 2], state=state,
    )
    got = row_bytes(staging.arena, output[0, -1, :1])
    staging.attention_finished(0)
    torch.cuda.synchronize()
    assert (got == 123).all()
    assert not (tokens == 64).any()
