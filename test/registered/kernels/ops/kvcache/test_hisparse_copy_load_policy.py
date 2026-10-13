"""Byte-exact DSv4 load-policy checks, including dynamic CUDA graph replay."""

from types import SimpleNamespace

import pytest
import torch

from sglang.benchmark.hisparse_h2d_diagnostics import (
    ITEM_BYTES,
    PAGE_BYTES,
    PAGE_TOKENS,
    SCALE_BYTES,
    VALUE_BYTES,
)
from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def make_copy_case(device, *, page_bytes=PAGE_BYTES):
    """Build the same fixture on CPU for layout checks and CUDA for execution.

    page_bytes is overridden only by the CPU regression reproducing the old bug.
    The real C4 page is ceil(64 * 584 / 576) * 576 = 37440 bytes.
    """
    rng = torch.Generator().manual_seed(908)
    host = torch.empty(
        (4, page_bytes),
        dtype=torch.uint8,
        pin_memory=torch.device(device).type == "cuda",
    )
    host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8, generator=rng))
    output = torch.empty((5, page_bytes), dtype=torch.uint8, device=device)
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
    src = torch.full((4, 96), -1, dtype=torch.int64, device=device)[:, :80]
    dst = torch.full((4, 96), -1, dtype=torch.int32, device=device)[:, :80]
    src.copy_(src_cpu)
    dst.copy_(dst_cpu)
    return SimpleNamespace(
        host=host,
        output=output,
        src_cpu=src_cpu,
        dst_cpu=dst_cpu,
        src=src,
        dst=dst,
        counts=torch.tensor([0, 1, 65, 9], dtype=torch.int32, device=device),
        real=torch.tensor([3], dtype=torch.int32, device=device),
    )


def expected_copy(case, active_counts, real_count):
    result = torch.full(case.output.shape, 165, dtype=torch.uint8)
    scale_base = PAGE_TOKENS * VALUE_BYTES
    for request in range(real_count):
        for miss in range(active_counts[request]):
            source, destination = int(case.src_cpu[request, miss]), int(
                case.dst_cpu[request, miss]
            )
            sp, so = divmod(source, PAGE_TOKENS)
            dp, do = divmod(destination, PAGE_TOKENS)
            result[dp, do * VALUE_BYTES : (do + 1) * VALUE_BYTES] = case.host[
                sp, so * VALUE_BYTES : (so + 1) * VALUE_BYTES
            ]
            result[
                dp, scale_base + do * SCALE_BYTES : scale_base + (do + 1) * SCALE_BYTES
            ] = case.host[
                sp, scale_base + so * SCALE_BYTES : scale_base + (so + 1) * SCALE_BYTES
            ]
    return result


def assert_copy_matches(case, active_counts, real_count, *, policy, phase):
    actual = case.output.cpu()
    expected = expected_copy(case, active_counts, real_count)
    mismatch = actual != expected
    if not mismatch.any():
        return
    first = [
        dict(
            tensor_row=row,
            byte_in_row=col,
            flat_byte=row * actual.shape[1] + col,
            actual=int(actual[row, col]),
            expected=int(expected[row, col]),
        )
        for row, col in mismatch.nonzero()[:8].tolist()
    ]
    raise AssertionError(
        f"Copy mismatch policy={policy} phase={phase} counts={active_counts} "
        f"real={real_count} page_bytes={actual.shape[1]} "
        f"src_stride={case.src.stride()} dst_stride={case.dst.stride()} "
        f"mismatched_bytes_per_tensor_row={mismatch.sum(dim=1).tolist()} first={first}"
    )


@pytest.mark.parametrize("policy", ["legacy", "ca", "cg"])
def test_paged_copy_policy_preserves_bytes_padding_and_dynamic_graph(policy):
    case = make_copy_case("cuda")

    def run():
        copy_cache_planned_mla(
            miss_src=case.src,
            miss_dst=case.dst,
            miss_count=case.counts,
            num_real_reqs=case.real,
            host_cache=case.host,
            device_buffer=case.output,
            item_size_bytes=ITEM_BYTES,
            is_dsv4_layout=True,
            num_blocks=4,
            block_size=1024,
            load_policy=policy,
        )

    case.output.fill_(165)
    run()  # Compile outside capture and validate eager execution as well.
    torch.cuda.synchronize()
    assert_copy_matches(case, [0, 1, 65, 9], 3, policy=policy, phase="eager")
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
        case.host.bitwise_xor_(255)
        case.counts.copy_(torch.tensor(active_counts, dtype=torch.int32))
        case.real.fill_(real_count)
        case.output.fill_(165)
        graph.replay()
        torch.cuda.synchronize()
        assert_copy_matches(
            case, active_counts, real_count, policy=policy, phase="graph_replay"
        )
