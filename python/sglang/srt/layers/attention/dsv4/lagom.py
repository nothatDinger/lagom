"""Route CSA selections for a Lagom decode window and its disposable suffix."""

from __future__ import annotations

import torch


def resolve_lagom_topk(
    *,
    coordinator,
    forward_batch,
    layer_id: int,
    raw_indices: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    output_buffer: torch.Tensor,
    window_locs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize real queries and retain disposable next-round forecasts.

    Forecast-only host misses are excluded from prediction attention. The
    dedicated forecast slot cannot be overwritten before attention finishes.
    """
    batch_size = forward_batch.req_pool_indices.numel()
    decode_width = forward_batch.lagom_decode_width
    if batch_size == 0 or raw_indices.shape[0] % batch_size:
        raise ValueError("Lagom CSA requires static request-major rows")
    full_width = raw_indices.shape[0] // batch_size
    if not 1 < decode_width < full_width:
        raise ValueError("Lagom CSA requires real candidates and a prediction suffix")
    capturing = getattr(forward_batch, "is_lagom_graph_capture", False)
    if forward_batch.positions.numel() != batch_size * full_width:
        raise ValueError("Lagom CSA requires one position per execution row")
    # The anchor position is the committed prefix. This graph input changes
    # on replay; CPU mirrors and reserved seq_lens are not valid graph inputs.
    prefix_lens = forward_batch.positions.reshape(batch_size, full_width)[:, 0]
    return coordinator.swap_in_lagom_pages(
        req_pool_indices=forward_batch.req_pool_indices,
        compressed_seq_lens=compressed_seq_lens,
        top_k_result=raw_indices,
        layer_id=layer_id,
        decode_width=decode_width,
        output_buffer=output_buffer,
        source_prefix_lens=prefix_lens,
        capturing=capturing,
        window_locs=window_locs,
    )
