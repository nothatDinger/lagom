"""Dequantize C4 prefill through a compact HiSparse payload, including graphs."""
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import SparsePrefillChunkCache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="DeepSeek-V4 paged CUDA dequantization is required",
)


@pytest.mark.parametrize("capture", [False, True])
def test_c4_gather_uses_live_hisparse_mapping_and_physical_bounds(capture):
    page_size, physical_pages, logical_pages = 64, 4, 12
    page_bytes = ((page_size * 584 + 575) // 576) * 576
    payload = torch.randint(
        0, 256, (physical_pages, page_bytes), dtype=torch.uint8, device="cuda"
    )
    # Unit scales avoid numerical underflow/overflow in the random FP8 bytes.
    payload[:, page_size * 576:page_size * (576 + 8)].fill_(127)
    mapping = torch.zeros(logical_pages * page_size, dtype=torch.int64, device="cuda")
    pages = torch.tensor(
        [[9, 11], [9, 11], [3, 1 << 30]], dtype=torch.int32, device="cuda"
    )
    cache = SimpleNamespace(
        c4_flat_token_ids=None, num_reqs=2, max_seq_len=260,
        seq_lens=torch.tensor([260, 4], dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda"),
        swa_offsets=torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda"),
    )
    output = torch.empty((130, 1, 512), dtype=torch.bfloat16, device="cuda")
    assert 9 * page_size >= payload.shape[0] * page_size

    def set_mapping(first_page, second_page):
        mapping[9 * page_size:10 * page_size].copy_(
            torch.arange(page_size, device="cuda") + first_page * page_size
        )
        mapping[11 * page_size] = second_page * page_size
        mapping[3 * page_size] = 3 * page_size

    def run():
        # Each forward owns a fresh chunk cache. Force recomputation during
        # warmup/capture; replay then executes the captured mapping operations.
        cache.c4_flat_token_ids = None
        SparsePrefillChunkCache.ensure_c4(
            cache, pages, page_size, logical_to_device=mapping,
            physical_num_pages=payload.shape[0],
        )
        dequantize_k_cache_paged(
            payload, cache.c4_flat_token_ids, page_size=page_size, out=output
        )

    set_mapping(1, 2)
    run()
    torch.cuda.synchronize()
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    for first_page, second_page in ((1, 2), (2, 1), (1, 2)):
        set_mapping(first_page, second_page)
        if graph is None:
            run()
        else:
            graph.replay()
        expected_ids = torch.cat((
            torch.arange(page_size, device="cuda") + first_page * page_size,
            torch.tensor([second_page * page_size, 3 * page_size], device="cuda"),
            torch.zeros(64, dtype=torch.int64, device="cuda"),
        )).to(torch.int32)
        reference = dequantize_k_cache_paged_ref(
            payload, expected_ids, page_size=page_size
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(cache.c4_flat_token_ids, expected_ids)
        torch.testing.assert_close(output, reference, atol=0, rtol=0, equal_nan=True)
