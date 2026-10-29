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
) -> torch.Tensor:
    """Demand-load real rows and queue only suffix logical IDs as cache hints.

    The indexer has evaluated the target's per-layer hidden states for all
    W+P+1 rows. Forecast attention may use resident history and C4 transaction
    scratch, but must not trigger demand IO. Publication happens only after the
    worker commits the real prefix, when prefetch eviction is safe.
    """
    batch_size = forward_batch.req_pool_indices.numel()
    decode_width = forward_batch.lagom_decode_width
    if batch_size == 0 or raw_indices.shape[0] % batch_size:
        raise ValueError("Lagom CSA requires static request-major rows")
    full_width = raw_indices.shape[0] // batch_size
    if not 1 < decode_width < full_width:
        raise ValueError("Lagom CSA requires real candidates and a prediction suffix")
    capturing = getattr(forward_batch, "is_lagom_graph_capture", False)
    prefix_lens_cpu = forward_batch.lagom_prefix_lens_cpu
    if not capturing and (
        prefix_lens_cpu is None or len(prefix_lens_cpu) != batch_size
    ):
        raise ValueError("Lagom CSA requires the committed prefix lengths")

    result = coordinator.swap_in_lagom_pages(
        req_pool_indices=forward_batch.req_pool_indices,
        compressed_seq_lens=compressed_seq_lens,
        top_k_result=raw_indices,
        layer_id=layer_id,
        decode_width=decode_width,
        output_buffer=output_buffer,
    )
    predictions = raw_indices.reshape(batch_size, full_width, -1)[:, decode_width:]
    if capturing:
        # Keep the final capture's graph-pool addresses, not the warmup
        # allocations. Python descriptors and live identities are published
        # after replay; queueing here would submit only once during capture.
        pending = getattr(forward_batch, "_lagom_pending_prefetch", None)
        if pending is None:
            pending = forward_batch._lagom_pending_prefetch = {}
        pending[layer_id] = (coordinator, predictions)
        return result
    coordinator.queue_lagom_prefetch(
        req_pool_indices=forward_batch.req_pool_indices,
        req_pool_indices_cpu=forward_batch.req_pool_indices_cpu,
        source_committed_lens_cpu=prefix_lens_cpu,
        predicted_c4_entries=predictions,
        layer_id=layer_id,
    )
    return result
