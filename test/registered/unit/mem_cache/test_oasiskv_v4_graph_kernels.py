"""Run with CUDA or TRITON_INTERPRET=1 to check V4 graph metadata kernels."""

import os

import pytest
import torch
import triton
from sglang.srt.layers.attention.triton_ops.oasiskv_v4 import (
    filter_history_candidates,
    publish_hot_metadata,
)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="CUDA or interpreter required",
)


@pytest.mark.parametrize("prefix", [0, 1, 2, 3, 4, 14, 15, 16, 18, 19, 20, 32767])
def test_filter_c4_boundaries_invalid_drafts_padding_and_strided_predictions(prefix):
    history = prefix // 4
    # Noncontiguous slices are used by native indexer outputs too.
    predictions = torch.tensor(
        [[-1, history - 1, history, history + 1, 91]] * 3,
        dtype=torch.int32,
        device=DEVICE,
    )[:, :4]
    output = torch.full((3, 4), 99, dtype=torch.int32, device=DEVICE)
    filter_history_candidates[(3,)](
        predictions,
        torch.tensor([prefix] * 3, device=DEVICE),
        torch.tensor([True, False, True], device=DEVICE),
        torch.tensor([2], dtype=torch.int32, device=DEVICE),
        output,
        predictions.stride(0),
        4,
        4,
        4,
    )
    assert output.cpu().tolist() == [
        [-1, max(-1, history - 1), -1, -1],
        [-1] * 4,
        [-1] * 4,
    ]


@pytest.mark.parametrize("real_count", [0, 1, 2])
def test_publish_never_touches_padding_newest_slot_or_other_layers(real_count):
    hot = 5
    tokens = torch.arange(2 * 4 * 8, dtype=torch.int32, device=DEVICE).view(2, 4, 8)
    lru = torch.arange(2 * 4 * hot, dtype=torch.int16, device=DEVICE).view(2, 4, hot)
    before_tokens, before_lru = tokens.clone(), lru.clone()
    planned = torch.arange(3 * hot, dtype=torch.int32, device=DEVICE).view(3, hot) + 90
    planned_lru = planned.to(torch.int16)
    publish_hot_metadata[(3,)](
        torch.tensor([2, 0, 2], device=DEVICE),  # padding aliases a live request
        torch.tensor([real_count], device=DEVICE),
        planned,
        planned_lru,
        tokens[1],
        lru[1],
        hot,
        tokens.stride(1),
        triton.next_power_of_2(hot),
    )
    for b, req in enumerate([2, 0][:real_count]):
        before_tokens[1, req, :hot] = planned[b]
        before_lru[1, req] = planned_lru[b]
    torch.testing.assert_close(tokens, before_tokens)
    torch.testing.assert_close(lru, before_lru)
