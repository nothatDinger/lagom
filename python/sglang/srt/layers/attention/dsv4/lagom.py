"""Predict CSA selections from frozen first-layer inputs on a separate stream."""

from __future__ import annotations

import torch


def lagom_prediction_geometry(forward_batch, page_table):
    """Build P query rows from live device inputs, restricted to committed history."""
    batch_size, prediction_width = forward_batch.lagom_prediction_tokens.shape
    decode_width = forward_batch.lagom_decode_width
    if (
        batch_size == 0
        or prediction_width == 0
        or decode_width <= 1
        or forward_batch.positions.numel() != batch_size * decode_width
        or page_table.shape[0] != batch_size * decode_width
    ):
        raise ValueError("Lagom requires separate request-major W and P windows")
    prefix = forward_batch.positions.reshape(batch_size, decode_width)[:, :1]
    positions = prefix + torch.arange(
        decode_width,
        decode_width + prediction_width,
        device=prefix.device,
        dtype=prefix.dtype,
    )
    # Paged indexer kernels require a positive length. The planner subsequently
    # masks the dummy selection for prefixes without any complete C4 entries.
    lengths = (prefix // 4).clamp_min(1).expand(-1, prediction_width)
    pages = page_table.reshape(batch_size, decode_width, -1)[:, :1]
    pages = pages.expand(-1, prediction_width, -1).reshape(
        batch_size * prediction_width, page_table.shape[1]
    )
    return (
        positions.reshape(-1).contiguous(),
        lengths.reshape(-1).to(torch.int32).contiguous(),
        pages.contiguous(),
    )


def prepare_lagom_prediction(forward_batch, attn_backend):
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    source = attn_backend.forward_metadata.indexer_metadata
    positions, lengths, pages = lagom_prediction_geometry(
        forward_batch, source.page_table
    )
    metadata = PagedIndexerMetadata(
        page_size=source.page_size,
        page_table=pages,
        c4_seq_lens=lengths,
        force_deep_gemm_metadata=source.force_deep_gemm_metadata,
    )
    return positions, metadata


def launch_lagom_prediction(
    *,
    attention,
    hidden_states,
    positions,
    metadata,
    forward_batch,
    attn_backend,
    stream,
):
    """Use the same frozen input for each CSA layer's query and weight projections."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qkv_a = None
        if attention.fuse_wqa_wkv:
            qkv_a, _ = attention.wqkv_a(hidden_states)
        q_lora = attention._compute_q_a(hidden_states, qkv_a=qkv_a)
        indices = attn_backend.forward_c4_indexer(
            x=hidden_states,
            q_lora=q_lora,
            c4_indexer=attention.indexer,
            forward_batch=forward_batch,
            skip_compressor=True,
            prediction_metadata=metadata,
            prediction_positions=positions,
        )
        ready = stream.record_event()
    return indices, ready


def resolve_lagom_topk(
    *,
    coordinator,
    forward_batch,
    layer_id: int,
    raw_indices: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    output_buffer: torch.Tensor,
) -> torch.Tensor:
    """Plan the W/P union but return only real attention locations."""
    batch_size = forward_batch.req_pool_indices.numel()
    decode_width = forward_batch.lagom_decode_width
    prediction = forward_batch.lagom_prediction_indices
    if (
        batch_size == 0
        or decode_width <= 1
        or raw_indices.shape[0] != batch_size * decode_width
        or forward_batch.positions.numel() != batch_size * decode_width
        or prediction is None
        or prediction.ndim != 2
        or prediction.shape[0] == 0
        or prediction.shape[0] % batch_size
        or prediction.shape[1] != raw_indices.shape[1]
    ):
        raise ValueError("Lagom CSA requires separate real and prediction index rows")
    ready = forward_batch.lagom_prediction_ready
    if ready is not None:
        ready.wait(torch.cuda.current_stream())
        prediction.record_stream(torch.cuda.current_stream())
    prediction_width = prediction.shape[0] // batch_size
    top_k = raw_indices.shape[1]
    full_width = decode_width + prediction_width
    indices = torch.cat(
        (
            raw_indices.reshape(batch_size, decode_width, top_k),
            prediction.reshape(batch_size, prediction_width, top_k),
        ),
        dim=1,
    ).reshape(batch_size * full_width, top_k)
    prefix_lens = forward_batch.positions.reshape(batch_size, decode_width)[:, 0]
    lengths = compressed_seq_lens.reshape(batch_size, -1)
    if lengths.shape[1] == 1:
        lengths = lengths.expand(-1, decode_width)
    if lengths.shape[1] != decode_width:
        raise ValueError("Lagom needs one compressed length per real query or request")
    history = (prefix_lens // coordinator.compress_ratio).to(lengths.dtype)
    lengths = torch.cat(
        (lengths, history[:, None].expand(-1, prediction_width)), dim=1
    ).reshape(-1)
    locations = coordinator.swap_in_lagom_pages(
        req_pool_indices=forward_batch.req_pool_indices,
        compressed_seq_lens=lengths,
        top_k_result=indices,
        layer_id=layer_id,
        decode_width=decode_width,
        source_prefix_lens=prefix_lens,
        capturing=forward_batch.is_lagom_graph_capture,
    )
    output_buffer.copy_(
        locations.reshape(batch_size, full_width, top_k)[:, :decode_width].reshape(
            batch_size * decode_width, top_k
        )
    )
    return output_buffer
