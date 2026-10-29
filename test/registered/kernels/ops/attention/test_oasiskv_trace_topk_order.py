"""Real CUDA V4 v1 Top-K launches; full-model replay is the replay-check recipe.

These tests do not assert that the native scatter happens to reorder on a given
GPU. They check repeated fixed-score membership and raw/page alignment, including
padding and equal-score selection, with the optional diagnostic ordering enabled.
"""

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.srt.speculative.oasiskv_trace_topk import canonicalize_topk_
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("width", [3, 6])
@pytest.mark.parametrize("ties", [False, True])
def test_native_topk_repeated_fixed_scores_with_logical_order(width, ties):
    n, k, page_size = 1024, 512, 64
    torch.manual_seed(42)
    scores = torch.stack(
        [torch.randperm(n, device="cuda") for _ in range(width)]
    ).float()
    if ties:
        # 512 cuts through a group of tied scores, rather than between groups.
        scores = scores.div(65, rounding_mode="floor")
    lengths = torch.full((width,), n, dtype=torch.int32, device="cuda")
    lengths[1], lengths[2] = 117, 0
    page_table = torch.stack(
        [torch.randperm(n // page_size, device="cuda") for _ in range(width)]
    ).to(torch.int32)
    raw = torch.empty((width, k), dtype=torch.int32, device="cuda")
    pages = torch.empty_like(raw)
    reference = None
    for _ in range(8):
        topk_transform_512(scores, lengths, page_table, pages, page_size, raw)
        before = raw.clone()
        canonicalize_topk_(raw, pages)
        assert torch.equal(raw.sort(-1).values, before.sort(-1).values)
        safe = raw.clamp_min(0).long()
        expected_pages = (
            page_table.gather(1, safe // page_size) * page_size + safe % page_size
        )
        expected_pages.masked_fill_(raw < 0, -1)
        assert torch.equal(pages, expected_pages)
        for row, length in enumerate(lengths.tolist()):
            selected = raw[row][raw[row] >= 0].long()
            assert len(selected) == min(k, length)
            assert len(selected.unique()) == len(selected)
            assert (selected < length).all()
            # Score multiset remains exact even if equal-score tie winners vary.
            expected_scores = (
                scores[row, :length].topk(min(k, length)).values.sort().values
            )
            assert torch.equal(scores[row, selected].sort().values, expected_scores)
        if not ties and reference is not None:
            assert torch.equal(raw, reference)
        reference = raw.clone()
