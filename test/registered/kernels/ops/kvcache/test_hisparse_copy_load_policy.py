"""Byte-exact DSv4 load-policy checks, including dynamic CUDA graph replay."""

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("policy", ["legacy", "ca", "cg"])
def test_paged_copy_policy_preserves_bytes_padding_and_dynamic_graph(policy):
    page_bytes, value_bytes, page_tokens = 38016, 576, 64
    rng = torch.Generator().manual_seed(908)
    host = torch.empty((4, page_bytes), dtype=torch.uint8, pin_memory=True)
    host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8, generator=rng))
    output = torch.empty((5, page_bytes), dtype=torch.uint8, device="cuda")
    # Non-contiguous rows; unused and graph-padded requests contain invalid IDs.
    src_cpu = torch.full((4, 96), -1, dtype=torch.int64)[:, :80]
    dst_cpu = torch.full((4, 96), -1, dtype=torch.int32)[:, :80]
    sources = torch.randperm(256, generator=rng)[:66]
    sources[:4] = torch.tensor([0, 63, 64, 255])
    destinations = torch.randperm(320, generator=rng)
    destinations = torch.cat(
        (
            torch.tensor([63, 64]),
            destinations[(destinations != 63) & (destinations != 64)][:64],
        )
    )
    src_cpu[1, 0], src_cpu[2, :65] = sources[0], sources[1:]
    dst_cpu[1, 0], dst_cpu[2, :65] = destinations[0], destinations[1:].to(torch.int32)
    # Allocate the backing rows before slicing, preserving a 96-element stride.
    src = torch.full((4, 96), -1, dtype=torch.int64, device="cuda")[:, :80]
    dst = torch.full((4, 96), -1, dtype=torch.int32, device="cuda")[:, :80]
    src.copy_(src_cpu)
    dst.copy_(dst_cpu)
    counts = torch.tensor([0, 1, 65, 9], dtype=torch.int32, device="cuda")
    real = torch.tensor([3], dtype=torch.int32, device="cuda")

    def run():
        copy_cache_planned_mla(
            miss_src=src,
            miss_dst=dst,
            miss_count=counts,
            num_real_reqs=real,
            host_cache=host,
            device_buffer=output,
            item_size_bytes=584,
            is_dsv4_layout=True,
            num_blocks=4,
            block_size=1024,
            load_policy=policy,
        )

    def expected(active_counts, real_count):
        result = torch.full((5, page_bytes), 165, dtype=torch.uint8)
        for request in range(real_count):
            for miss in range(active_counts[request]):
                source, destination = int(src_cpu[request, miss]), int(
                    dst_cpu[request, miss]
                )
                sp, so = divmod(source, page_tokens)
                dp, do = divmod(destination, page_tokens)
                result[dp, do * value_bytes : (do + 1) * value_bytes] = host[
                    sp, so * value_bytes : (so + 1) * value_bytes
                ]
                result[dp, 64 * 576 + do * 8 : 64 * 576 + (do + 1) * 8] = host[
                    sp, 64 * 576 + so * 8 : 64 * 576 + (so + 1) * 8
                ]
        return result

    output.fill_(165)
    run()  # Compile outside capture and validate eager execution as well.
    torch.cuda.synchronize()
    assert torch.equal(output.cpu(), expected([0, 1, 65, 9], 3))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        run()
    for active_counts, real_count in [
        ([0, 1, 65, 9], 3),
        ([0, 0, 0, 9], 3),
        ([0, 1, 65, 9], 2),
        ([0, 1, 65, 9], 0),
    ]:
        # Exercise changing host bytes between replays, as well as padded/zero rows.
        host.bitwise_xor_(255)
        counts.copy_(torch.tensor(active_counts, dtype=torch.int32))
        real.fill_(real_count)
        output.fill_(165)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.cpu(), expected(active_counts, real_count))
