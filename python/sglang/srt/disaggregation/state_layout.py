"""State layout helpers shared by PD allocation and transfer validation.

Keep this module independent of torch/CUDA: layout negotiation is startup/IO
control-plane work, not part of the decode forward or its CUDA graphs.
"""

from __future__ import annotations


def dsv4_state_execution_width(
    *,
    is_lagom: bool,
    max_speculative_num_draft_tokens: int | None,
    prefill_execution_width: int | None = None,
) -> int:
    """Size compressor state without changing the local speculative window.

    Ordinary DSPARK prefill can prepare the wider state layout needed by a
    Lagom decoder. Its public W, proposal width, acceptance and graph shapes
    remain unchanged. Without an override, preserve the existing sizing.
    """
    local_width = (max_speculative_num_draft_tokens or 0) if is_lagom else 0
    if prefill_execution_width is None:
        return local_width
    # Largest C4 ring: 128 slots, preceding window: 8 rows, write pad: +2.
    if (
        type(prefill_execution_width) is not int
        or not 0 < prefill_execution_width <= 122
    ):
        raise ValueError(
            "DeepSeek-V4 PD state execution width must be an integer in [1, 122]"
        )
    return max(local_width, prefill_execution_width)


def validate_state_transfer_layout(
    transfers: list[tuple[int, int, int]],
    dst_data_ptrs: list[int],
    dst_item_lens: list[int],
    *,
    state_type: str,
) -> None:
    """Validate selected source/destination entries before any state DMA.

    Transfers have already been paired/sliced for PP. Whole-pool capacities
    and base addresses can differ; the byte stride of each paired item cannot.
    The generic copy path uses one item size for both address calculations.
    """
    if len(dst_data_ptrs) != len(dst_item_lens):
        raise RuntimeError(
            f"PD {state_type} state layout metadata is incomplete: "
            f"{len(dst_data_ptrs)} destination buffers, "
            f"{len(dst_item_lens)} destination item sizes"
        )
    dst_sizes = dict(zip(dst_data_ptrs, dst_item_lens))
    for _, dst_ptr, src_item_len in transfers:
        dst_item_len = dst_sizes.get(dst_ptr)
        if src_item_len <= 0 or src_item_len != dst_item_len:
            raise RuntimeError(
                f"PD {state_type} state layout mismatch: "
                f"source item bytes={src_item_len}, "
                f"destination item bytes={dst_item_len}. "
                "State copying requires matching per-item layouts. "
                "For a DeepSeek-V4 Lagom decoder, set "
                "--disaggregation-dsv4-state-execution-width=W+P+1 "
                "on its ordinary DSPARK prefill server."
            )
