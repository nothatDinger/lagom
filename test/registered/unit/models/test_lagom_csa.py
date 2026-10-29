from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.dsv4.lagom import resolve_lagom_topk


def make_batch():
    return SimpleNamespace(
        req_pool_indices=torch.tensor([3, 8]),
        req_pool_indices_cpu=torch.tensor([3, 8]),
        lagom_decode_width=3,
        lagom_prefix_lens_cpu=torch.tensor([1024, 2048]),
        positions=(torch.tensor([1024, 2048])[:, None] + torch.arange(5)).reshape(-1),
    )


def test_lagom_csa_passes_both_windows_and_device_prefix_to_split_planner():
    batch = make_batch()
    logical_ids = torch.arange(2 * 5 * 4).reshape(10, 4)
    physical_locs = torch.full((10, 4), 500, dtype=torch.int32)
    coordinator = SimpleNamespace(
        swap_in_lagom_pages=Mock(return_value=physical_locs),
    )
    result = resolve_lagom_topk(
        coordinator=coordinator,
        forward_batch=batch,
        layer_id=7,
        raw_indices=logical_ids,
        compressed_seq_lens=torch.arange(10),
        output_buffer=physical_locs,
    )
    assert result is physical_locs
    swap = coordinator.swap_in_lagom_pages.call_args.kwargs
    assert swap["decode_width"] == 3  # anchor + W; suffix has a separate copy plan
    assert swap["top_k_result"] is logical_ids
    assert swap["source_prefix_lens"].tolist() == [1024, 2048]
    assert swap["req_pool_indices"] is batch.req_pool_indices
    assert swap["layer_id"] == 7
    assert not swap["capturing"]


@pytest.mark.parametrize("width,rows,positions", [(5, 10, 10), (3, 9, 10), (3, 10, 9)])
def test_invalid_lagom_geometry_cannot_launch_prefetch(width, rows, positions):
    batch = make_batch()
    batch.lagom_decode_width = width
    batch.positions = torch.arange(positions)
    coordinator = Mock()
    with pytest.raises(ValueError):
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=batch,
            layer_id=0,
            raw_indices=torch.zeros((rows, 4), dtype=torch.int32),
            compressed_seq_lens=torch.zeros(rows),
            output_buffer=torch.empty((rows, 4), dtype=torch.int32),
        )
    coordinator.swap_in_lagom_pages.assert_not_called()


def test_graph_capture_uses_live_device_metadata_without_postreplay_descriptors():
    batch = make_batch()
    batch.is_lagom_graph_capture = True
    batch.lagom_prefix_lens_cpu = None
    coordinator = Mock()
    for fill in (11, 22):
        batch.positions.add_(fill)
        raw = torch.full((10, 4), fill, dtype=torch.int32)
        resolve_lagom_topk(
            coordinator=coordinator,
            forward_batch=batch,
            layer_id=1,
            raw_indices=raw,
            compressed_seq_lens=torch.zeros(10),
            output_buffer=torch.empty_like(raw),
        )
    args = coordinator.swap_in_lagom_pages.call_args.kwargs
    assert args["capturing"]
    assert args["top_k_result"] is raw
    assert args["source_prefix_lens"].tolist() == [1057, 2081]
    assert not hasattr(batch, "_lagom_pending_prefetch")
