"""GPU byte equality and graph replay tests for Adaptive DHiSparse."""

import pytest
import torch

from sglang.kernels.ops.kvcache.dhisparse import allocate_workspace, promote, resolve
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Adaptive DHiSparse CUDA execution requires an NVIDIA GPU",
)


def rows(buffer, locations, packed):
    if not packed:
        return buffer[locations.long()]
    page, offset = locations // 64, locations % 64
    value = buffer[
        page[..., None],
        offset[..., None] * 576 + torch.arange(576, device=buffer.device),
    ]
    scale = buffer[
        page[..., None],
        64 * 576 + offset[..., None] * 8 + torch.arange(8, device=buffer.device),
    ]
    return torch.cat((value, scale), -1)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_union_overflow_payload_padding_and_graph_replay(packed, capture):
    n, hot, width, k, history = 3, 64, 8, 16, 256
    stride, capacity = hot + 64, width * k
    owned, total = n * stride, n * (stride + capacity)
    item_bytes = 584 if packed else 576
    page_bytes = 65 * 576
    shape = (n * history // 64, page_bytes) if packed else (n * history, item_bytes)
    host = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    host.copy_(
        torch.arange(host.numel(), dtype=torch.int64).remainder(251).view_as(host)
    )
    device_shape = (total // 64, page_bytes) if packed else (total, item_bytes)
    payload = torch.full(device_shape, 211, dtype=torch.uint8, device="cuda")
    tokens = torch.full((n, stride), -1, dtype=torch.int32, device="cuda")
    locations = torch.arange(n * stride, dtype=torch.int32, device="cuda").view(
        n, stride
    )
    optional = torch.full((n, history), -1, dtype=torch.int32, device="cuda")
    host_locs = torch.arange(n * history, device="cuda").view(n, history)
    requests = torch.tensor([2, 0, 1], dtype=torch.int64, device="cuda")
    lengths = torch.full((n, width), history, dtype=torch.int32, device="cuda")
    selected = (
        torch.arange(n * width * k, device="cuda", dtype=torch.int32).view(n, width, k)
        % history
    )
    selected[0, :, 0] = 19
    selected[0, 0, 1] = -1
    lengths[0, 0] = 23
    output = torch.empty_like(selected)
    saved = torch.empty((*selected.shape, item_bytes), dtype=torch.uint8, device="cuda")
    ages, epochs = torch.zeros(n, hot, dtype=torch.int64, device="cuda"), torch.zeros(
        n, dtype=torch.int64, device="cuda"
    )
    workspace = allocate_workspace(n, capacity, hot, "cuda")
    kwargs = dict(
        request_ids=requests,
        buffer_tokens=tokens,
        buffer_locs=locations,
        hot_ages=ages,
        epochs=epochs,
        device_buffer=payload,
        staging_offset=owned,
        staging_stride=capacity,
        hot_size=hot,
        workspace=workspace,
        num_real_requests=torch.tensor([2], dtype=torch.int32, device="cuda"),
        item_bytes=item_bytes,
        packed=packed,
    )

    def run():
        resolve(
            selections=selected,
            seq_lens=lengths,
            optional_locs=optional,
            host_locs=host_locs,
            host=host,
            output=output,
            **kwargs
        )
        saved.copy_(rows(payload, output.clamp_min(0), packed))
        promote(**kwargs)

    run()  # JIT compilation and temporary storage before capture.
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    for offset in (0, 31, 101):
        selected.add_(offset).remainder_(history)
        selected[0, 0, 1] = -1
        if graph is None:
            run()
        else:
            graph.replay()
        torch.cuda.synchronize()
        chosen = selected.cpu()
        valid = (chosen >= 0) & (chosen < lengths.cpu()[..., None])
        valid[2] = False
        logical = requests.cpu()[:, None, None] * history + chosen.clamp_min(0)
        expected = rows(host, logical, packed)
        torch.testing.assert_close(saved.cpu()[valid], expected[valid], rtol=0, atol=0)
        assert (output.cpu()[~valid] == -1).all()
        assert epochs[1].item() == 0
        # Promotions have stable addresses even when the shared arena is reused.
        if packed:
            payload[owned // 64 :].fill_(17)
        else:
            payload[owned:].fill_(17)
        for rid in (0, 2):
            live = tokens[rid, :hot] >= 0
            expected = rows(host, rid * history + tokens[rid, :hot][live].cpu(), packed)
            actual = rows(payload, locations[rid, :hot][live], packed).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
