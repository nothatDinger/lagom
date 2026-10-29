"""Opt-in offline diagnostic: fix KV visitation order, not Top-K membership."""

import torch


def canonicalize_topk_(raw_indices, page_indices):
    """Apply one logical-ID permutation to BOTH raw and translated indices.

    Must run before the indexer clones logical page indices for transaction
    scratch resolution, swap-in and attention. Sorting just the observer's CPU
    lists cannot remove attention's floating-point reduction-order variation.
    This does not stabilize score computation or choose deterministic tie winners.
    """
    if raw_indices.ndim != 2 or raw_indices.shape != page_indices.shape:
        raise ValueError("Trace Top-K ordering requires matching 2D index buffers")
    # Retain invalid padding at the end, including short/all-padding rows.
    keys = torch.where(
        raw_indices >= 0, raw_indices, torch.iinfo(raw_indices.dtype).max
    )
    permutation = torch.argsort(keys, dim=-1, stable=True)
    ordered_raw = raw_indices.gather(-1, permutation)
    ordered_pages = page_indices.gather(-1, permutation)
    raw_indices.copy_(ordered_raw)
    page_indices.copy_(ordered_pages)
