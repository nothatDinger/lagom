"""Keep Lagom's prediction suffix outside DSpark's acceptance boundary."""

from __future__ import annotations

from typing import Any

import torch


def select_decode_rows(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    execution_width: int,
    decode_width: int,
) -> torch.Tensor:
    """Select each request's prefix from flattened request-major model output."""
    if not 0 < decode_width <= execution_width:
        raise ValueError("Lagom decode width must be within the execution window")
    if tensor.shape[0] != batch_size * execution_width:
        raise ValueError(
            "Lagom tensor does not match its request-major execution width"
        )
    return (
        tensor.reshape(batch_size, execution_width, *tensor.shape[1:])[:, :decode_width]
        .reshape(batch_size * decode_width, *tensor.shape[1:])
        .contiguous()
    )


def decode_proposal(proposal: Any, decode_candidates: int) -> Any:
    """Exclude probes from token acceptance, proposal probabilities and confidence."""
    block = proposal.draft_block
    bs, execution_candidates = block.draft_tokens.shape
    if not 0 < decode_candidates <= execution_candidates:
        raise ValueError("Lagom decode candidates must be within the draft block")
    corrected_logits = block.corrected_logits
    if corrected_logits is not None:
        corrected_logits = corrected_logits[:, :decode_candidates].contiguous()
    confidence_tap = proposal.confidence_tap
    if confidence_tap is not None:
        confidence_tap = select_decode_rows(
            confidence_tap,
            batch_size=bs,
            execution_width=execution_candidates,
            decode_width=decode_candidates,
        )
    return type(proposal)(
        draft_block_ids=proposal.draft_block_ids[:, :decode_candidates].contiguous(),
        draft_block=type(block)(
            draft_tokens=block.draft_tokens[:, :decode_candidates].contiguous(),
            corrected_logits=corrected_logits,
            greedy_mask=block.greedy_mask,
            temperatures=block.temperatures,
        ),
        draft_hidden=(
            None
            if proposal.draft_hidden is None
            else proposal.draft_hidden[:, :decode_candidates].contiguous()
        ),
        confidence=(
            None
            if proposal.confidence is None
            else proposal.confidence[:, :decode_candidates].contiguous()
        ),
        confidence_tap=confidence_tap,
        # Preserve whether the draft graph produced this proposal. The worker
        # separately excludes Lagom from the W-wide target epilogue.
        folded=proposal.folded,
    )


def decode_window(window: Any, decode_width: int) -> Any:
    """Keep committed hidden rows, positions and physical cache locations aligned."""
    if not 0 < decode_width <= window.positions_2d.shape[1]:
        raise ValueError("Lagom decode width must be within the execution window")
    locations = window.verify_cache_loc_2d[:, :decode_width].contiguous()
    return type(window)(
        positions_2d=window.positions_2d[:, :decode_width].contiguous(),
        verify_cache_loc=locations.reshape(-1),
        verify_cache_loc_2d=locations,
    )
