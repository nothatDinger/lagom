"""Fused multi-token HiSparse gather/commit with persistent hash/CLOCK state."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.mem_cache.hisparse_spec import HiSparseSpecLayout

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_GATHER_BLOCK_SIZE = 64
_LINEAR_TRANSFER = "linear"
_DSV4_PAGED_TRANSFER = "dsv4_paged"


class HiSparseSpecState(NamedTuple):
    """Persistent cache state and reusable miss workspace for speculative swap.

    ``cache_index`` stores the two int64 hash banks as
    ``[num_requests, 2, hash_size]``. ``cache_policy`` uses a control-plane row
    for the packed CLOCK states followed by one reference-epoch row per
    request: ``[1 + num_requests, hot_buffer_size]``.

    ``scratch_locs`` and ``scratch_state`` hold reusable miss locations,
    counters, and metadata allocated separately for each layer.
    """

    cache_index: torch.Tensor
    cache_policy: torch.Tensor
    scratch_locs: torch.Tensor
    scratch_state: torch.Tensor


class LagomCachePlan(NamedTuple):
    """One deduplicated miss plan partitioned between main and side streams.

    The two copy passes select complementary ``is_demand`` values. Their
    destinations are therefore disjoint, including a token selected by both
    windows: any real query makes its unique miss a main-stream refill.
    """

    miss_src: torch.Tensor
    miss_dst: torch.Tensor
    miss_count: torch.Tensor
    is_demand: torch.Tensor


def allocate_hisparse_spec_states(
    layout: HiSparseSpecLayout,
    *,
    device_buffer_locs: torch.Tensor,
    scratch_offset: int,
) -> list[HiSparseSpecState]:
    """Allocate metadata; KV scratch aliases each request's existing allocation."""
    num_layers, num_req_slots, padded_slots = device_buffer_locs.shape
    if not layout.hot_slots < scratch_offset < padded_slots:
        raise ValueError(
            "Fused HiSparse needs request-owned KV scratch after the active tail."
        )
    device = device_buffer_locs.device
    return [
        HiSparseSpecState(
            torch.full(
                (num_req_slots, 2, layout.hash_slots),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            torch.zeros(
                (num_req_slots + 1, layout.policy_width(num_req_slots)),
                dtype=torch.int32,
                device=device,
            )[:, : layout.hot_slots],
            device_buffer_locs[layer, :, scratch_offset:],
            torch.zeros(
                (num_req_slots + 1, layout.state_width(num_req_slots)),
                dtype=torch.int32,
                device=device,
            ),
        )
        for layer in range(num_layers)
    ]


def _spec_transfer_policy(
    transfer_kind: str, item_size_bytes: int | None = None
) -> str:
    if transfer_kind == _LINEAR_TRANSFER:
        if item_size_bytes is None:
            raise ValueError("Linear HiSparse KV transfer requires item_size_bytes.")
        return f"LinearKVTransferPolicy<{item_size_bytes}>"
    if transfer_kind == _DSV4_PAGED_TRANSFER:
        return "Dsv4PagedKVTransferPolicy"
    raise ValueError(f"Unsupported HiSparse KV transfer kind: {transfer_kind!r}.")


@cache_once
def _jit_spec_module(
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_steps: int,
    record_miss_plan: bool,
    transfer_policy: str,
    decode_width: int = 0,
    prefetch_top_k: int = 0,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
        decode_width,
        prefetch_top_k,
        transfer_policy,
        is_arch_support_pdl(),
    )
    return load_jit(
        "hisparse_spec",
        *template_args,
        cuda_files=["hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_spec",
                f"load_cache_to_device_buffer_spec<{template_args}>",
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

    module = _jit_spec_module(
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
        torch.empty(0),
    )


def plan_lagom_cache(
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
    decode_width: int,
    prefetch_top_k: int | None = None,
    plan: LagomCachePlan | None = None,
    transfer_kind: str = _DSV4_PAGED_TRANSFER,
    item_size_bytes: int | None = None,
) -> LagomCachePlan:
    """Reserve a protected current/future union without transferring KV bytes.

    The full window must fit in the hot buffer. All resident forecast entries
    remain readable; future-only misses are returned as -1, while misses shared
    with real queries are readable after the demand copy. ``prefetch_top_k``
    limits future IO without removing resident hits outside that rank budget.
    Caller-owned plans support graph-stable, per-layer storage.
    """
    batch, steps, top_k = top_k_tokens.shape
    capacity = steps * top_k
    if not 0 < decode_width < steps:
        raise ValueError("Lagom requires nonempty real and prediction windows")
    if not 0 < capacity <= min(state.cache_policy.size(1), 8192):
        raise ValueError(
            "Lagom current/future union must fit hot slots and 8192 entries"
        )
    prefetch_top_k = top_k if prefetch_top_k is None else prefetch_top_k
    if not 0 < prefetch_top_k <= top_k:
        raise ValueError("Lagom prefetch_top_k must be between 1 and top_k")
    if plan is None:
        plan = LagomCachePlan(
            torch.empty(
                (batch, capacity), dtype=torch.int64, device=top_k_tokens.device
            ),
            torch.empty(
                (batch, capacity), dtype=torch.int32, device=top_k_tokens.device
            ),
            torch.empty((batch,), dtype=torch.int32, device=top_k_tokens.device),
            torch.empty(
                (batch, capacity), dtype=torch.int32, device=top_k_tokens.device
            ),
        )
    for tensor, dtype in (
        (plan.miss_src, torch.int64),
        (plan.miss_dst, torch.int32),
        (plan.is_demand, torch.int32),
    ):
        if (
            tensor.dtype != dtype
            or tensor.device != top_k_tokens.device
            or tensor.ndim != 2
            or tensor.size(0) < batch
            or tensor.size(1) < capacity
            or tensor.stride(1) != 1
            or tensor.stride(0) != plan.miss_src.stride(0)
        ):
            raise ValueError(
                "Lagom plan arrays must share device, capacity and row stride"
            )
    if (
        plan.miss_count.dtype != torch.int32
        or plan.miss_count.device != top_k_tokens.device
        or plan.miss_count.ndim != 1
        or plan.miss_count.numel() < batch
        or plan.miss_count.stride(0) != 1
    ):
        raise ValueError("Lagom plan counts must be a contiguous int32 batch vector")
    if transfer_kind == _LINEAR_TRANSFER and item_size_bytes is None:
        item_size_bytes = host_cache.stride(0) * host_cache.element_size()
    module = _jit_spec_module(
        _GATHER_BLOCK_SIZE,
        top_k,
        state.cache_policy.size(1),
        steps,
        True,
        _spec_transfer_policy(transfer_kind, item_size_bytes),
        decode_width,
        prefetch_top_k,
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
        device_buffer_tokens.size(1) - state.cache_policy.size(1),
        *plan,
    )
    return plan


@cache_once
def _jit_lagom_copy_module(
    block_size: int, demand: bool, transfer_policy: str
) -> Module:
    args = make_cpp_args(block_size, demand, transfer_policy)
    return load_jit(
        "hisparse_lagom_copy",
        *args,
        cuda_files=["hisparse_spec.cuh"],
        cuda_wrappers=[("copy_lagom_cache_plan", f"copy_lagom_cache_plan<{args}>")],
    )


def copy_lagom_cache_plan(
    plan: LagomCachePlan,
    *,
    demand: bool,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    transfer_kind: str = _DSV4_PAGED_TRANSFER,
    item_size_bytes: int | None = None,
    num_blocks: int = 4,
) -> None:
    """Copy one disjoint plan partition without touching cache metadata."""
    if num_blocks <= 0:
        raise ValueError("Lagom copy requires a positive block count")
    if transfer_kind == _LINEAR_TRANSFER and item_size_bytes is None:
        item_size_bytes = host_cache.stride(0) * host_cache.element_size()
    module = _jit_lagom_copy_module(
        1024, demand, _spec_transfer_policy(transfer_kind, item_size_bytes)
    )
    module.copy_lagom_cache_plan(
        *plan, num_real_reqs, host_cache, device_buffer, num_blocks
    )


@cache_once
def _jit_spec_state_module(hot_buffer_size: int) -> Module:
    template_args = make_cpp_args(hot_buffer_size)
    return load_jit(
        "hisparse_spec_state",
        hot_buffer_size,
        cuda_files=["hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "initialize_hisparse_spec_state",
                f"initialize_hisparse_spec_state<{template_args}>",
            )
        ],
    )


def initialize_hisparse_spec_state(
    *,
    device_buffer_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    state: HiSparseSpecState,
) -> None:
    hot_buffer_size = state.cache_policy.size(1)
    module = _jit_spec_state_module(hot_buffer_size)
    module.initialize_hisparse_spec_state(
        device_buffer_tokens,
        req_pool_indices.to(torch.int64),
        state.cache_index,
        state.cache_policy,
        state.scratch_state,
    )


@cache_once
def _jit_spec_lookup_module(hot_buffer_size: int) -> Module:
    return load_jit(
        "hisparse_spec_lookup",
        hot_buffer_size,
        cuda_files=["hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "lookup_hisparse_spec_resident",
                f"lookup_hisparse_spec_resident<{hot_buffer_size}>",
            )
        ],
    )


def lookup_hisparse_spec_resident(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Resolve probe queries against residency without IO or CLOCK updates.

    Missing historical entries stay -1. The caller separately overlays current
    transaction C4 writer locations, which are absent from the committed hash.
    """
    if top_k_tokens.ndim != 3 or output.shape != top_k_tokens.shape:
        raise ValueError(
            "resident lookup expects matching [batch, steps, top_k] tensors"
        )
    if not top_k_tokens.is_contiguous() or not output.is_contiguous():
        raise ValueError("resident lookup inputs and output must be contiguous")
    batch, steps, _ = top_k_tokens.shape
    if seq_lens.numel() != batch * steps:
        raise ValueError("resident lookup requires one sequence length per query")
    module = _jit_spec_lookup_module(state.cache_policy.size(1))
    module.lookup_hisparse_spec_resident(
        top_k_tokens,
        device_buffer_tokens,
        device_buffer_locs,
        req_pool_indices.to(torch.int64),
        seq_lens.to(torch.int32).contiguous().reshape(-1),
        state.cache_index,
        state.scratch_state,
        num_real_reqs,
        output,
    )
    return output
