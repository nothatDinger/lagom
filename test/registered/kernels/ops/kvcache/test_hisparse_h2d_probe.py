import pytest
import torch

from sglang.srt.utils.hisparse_h2d_probe import HiSparseH2DProbe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("capacity", [2, 8])
def test_every_graph_replay_records_real_rows_and_detects_overflow(capacity):
    probe = HiSparseH2DProbe("cuda", capacity, 8)
    counts = torch.tensor(
        [1, 2, 3, 999, 999, 999, 999, 999], dtype=torch.int32, device="cuda"
    )
    real = torch.tensor([3], dtype=torch.int32, device="cuda")

    def record():
        probe.record(
            counts, real, layer=7, width=6, item_bytes=584, copy_blocks=4, skip_io=False
        )

    record()  # Compile outside capture; profiling is still disabled.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        record()
    probe.start()
    for _ in range(3):
        graph.replay()
    report = probe.stop()
    assert report["total_records"] == 3
    assert report["overflow"] == (capacity == 2)
    assert len(report["samples"]) == min(3, capacity)
    for sample in report["samples"]:
        assert sample["misses_per_request"] == [1, 2, 3]
        assert sample["h2d_bytes"] == 6 * 584
        assert sample["real_batch"] == 3
        assert sample["captured_batch"] == 8
        assert sample["layer"] == 7


def test_paged_copy_and_probe_use_the_same_real_miss_plan():
    from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla

    page_bytes = ((64 * 584 + 575) // 576) * 576
    host = torch.randint(1, 255, (1, page_bytes), dtype=torch.uint8, pin_memory=True)
    device = torch.zeros((1, page_bytes), dtype=torch.uint8, device="cuda")
    src = torch.arange(24, dtype=torch.int64, device="cuda").view(8, 3)
    dst = src.to(torch.int32)
    counts = torch.tensor([1, 2, 3, 3, 3, 3, 3, 3], dtype=torch.int32, device="cuda")
    real = torch.tensor([3], dtype=torch.int32, device="cuda")
    probe = HiSparseH2DProbe("cuda", 8, 8)

    def run():
        copy_cache_planned_mla(
            miss_src=src,
            miss_dst=dst,
            miss_count=counts,
            num_real_reqs=real,
            host_cache=host,
            device_buffer=device,
            item_size_bytes=584,
            is_dsv4_layout=True,
        )
        probe.record(
            counts, real, layer=0, width=6, item_bytes=584, copy_blocks=4, skip_io=False
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    probe.start()
    graph.replay()
    report = probe.stop()
    actual = device.cpu()
    for loc in (0, 3, 4, 6, 7, 8):
        assert torch.equal(
            actual[0, loc * 576 : (loc + 1) * 576], host[0, loc * 576 : (loc + 1) * 576]
        )
        scale = 64 * 576 + loc * 8
        assert torch.equal(actual[0, scale : scale + 8], host[0, scale : scale + 8])
    assert not actual[0, 9 * 576 : 10 * 576].any()  # Padded request must not copy.
    assert report["samples"][0]["h2d_bytes"] == 6 * 584
