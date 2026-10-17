"""Experimental one-change swap variant, not the pinned PR or serving path."""

from __future__ import annotations

import torch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.benchmark.hisparse_pr40670_reference import (
    HiSparseSpecState,
    _GATHER_BLOCK_SIZE,
    _LINEAR_TRANSFER,
    _spec_transfer_policy,
)
from sglang.benchmark.hisparse_pr40670_retry import validate_candidate


@cache_once
def _jit_retry_module(
    block_size, num_top_k, hot_buffer_size, num_steps, record_miss_plan, transfer_policy
):
    validate_candidate()
    args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
        transfer_policy,
        is_arch_support_pdl(),
    )
    return load_jit(
        "hisparse_pr40670_retry_candidate",
        *args,
        cuda_files=["pr40670_retry_candidate/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_spec",
                f"load_cache_to_device_buffer_spec<{args}>",
            )
        ],
    )


def load_cache_to_device_buffer_spec_mla(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    transfer_kind: str = _LINEAR_TRANSFER,
    item_size_bytes: int | None = None,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the single-step HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.
    """
    _, num_steps, num_top_k = top_k_tokens.shape
    if num_steps <= 0 or num_top_k <= 0 or num_steps * num_top_k > 8192:
        raise ValueError(
            "HiSparse speculative swap requires positive steps/top-k and "
            f"steps * top-k <= 8192, got {num_steps} * {num_top_k}."
        )
    hot_buffer_size = state.cache_policy.size(1)
    active_tail_slots = device_buffer_tokens.size(1) - hot_buffer_size
    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32.")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32.")
        plan_capacity = num_steps * num_top_k
        batch_size = top_k_tokens.size(0)
        if (
            miss_src.ndim != 2
            or miss_dst.ndim != 2
            or miss_src.size(0) < batch_size
            or miss_dst.size(0) < batch_size
            or miss_src.size(1) < plan_capacity
            or miss_dst.size(1) < plan_capacity
        ):
            raise ValueError(
                "speculative miss_src/miss_dst must have shape "
                f"[batch, >= steps * top_k] (capacity {plan_capacity})."
            )
        if miss_count.ndim != 1 or miss_count.numel() < batch_size:
            raise ValueError("speculative miss_count must have shape [batch].")
        if miss_src.stride(0) != miss_dst.stride(0):
            raise ValueError("miss_src/miss_dst row strides must match.")
    else:
        if miss_dst is not None or miss_count is not None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        empty = torch.empty(0)
        miss_src = miss_dst = miss_count = empty

    if transfer_kind == _LINEAR_TRANSFER and item_size_bytes is None:
        item_size_bytes = host_cache.stride(0) * host_cache.element_size()

    module = _jit_retry_module(
        _GATHER_BLOCK_SIZE,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
        _spec_transfer_policy(transfer_kind, item_size_bytes),
    )

    module.load_cache_to_device_buffer_spec(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        state.cache_index,
        state.cache_policy,
        state.scratch_locs,
        state.scratch_state,
        num_real_reqs,
        active_tail_slots,
        miss_src,
        miss_dst,
        miss_count,
    )
