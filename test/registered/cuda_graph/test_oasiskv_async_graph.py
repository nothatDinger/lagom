"""External event nodes must leave the previous step's late transfer in flight."""

import pytest
import torch
from sglang.srt.mem_cache.oasiskv_async import LayerPrefetchGraphs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_next_step_early_layer_overlaps_previous_step_late_transfer():
    select, transfer = torch.cuda.Stream(), torch.cuda.Stream()
    pipes = [LayerPrefetchGraphs(select, transfer) for _ in range(2)]
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")
    late_payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    observed = torch.zeros_like(late_payload)
    early = torch.cuda.Event(enable_timing=True, external=True)
    late = torch.cuda.Event(enable_timing=True, external=True)
    start = torch.cuda.Event(enable_timing=True)

    def late_transfer():
        torch.cuda._sleep(50_000_000)
        late_payload.add_(1)
        late.record()

    pipes[0].capture(1, lambda: None, lambda: None)
    pipes[1].capture(1, lambda: None, late_transfer)
    torch.cuda.synchronize()
    late_payload.zero_()  # capture warmup performed one transfer
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipes[0].consume()
        observed.copy_(late_payload)
        counters[0].add_(1)
        early.record()
        pipes[0].prediction_ready.record()
        pipes[0].read_done.record()
        pipes[1].consume()
        counters[1].add_(1)
        pipes[1].prediction_ready.record()
        pipes[1].read_done.record()
    start.record()
    graph.replay()
    for pipe in pipes:
        pipe.replay(1)
    # Enqueue next foreground before previous late-layer transfer has finished.
    # No second background launch: inspect the previous transfer's event epoch.
    graph.replay()
    torch.cuda.synchronize()
    assert counters.tolist() == [2, 2]
    assert late_payload.item() == 1
    assert observed.item() == 0
    assert start.elapsed_time(early) < start.elapsed_time(late)
