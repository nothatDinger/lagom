"""Whole-window parallel hash/CLOCK planning adapted from PR #40670.

Only metadata is updated here. The caller copies the unique miss plan with the
same copy_cache_planned_mla kernel used by the positional LRU baseline.
"""

import functools
from typing import NamedTuple

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args
from sglang.srt.mem_cache.hisparse_window_hash import HiSparseWindowHashLayout


class HiSparseWindowHashState(NamedTuple):
    cache_index: torch.Tensor
    cache_policy: torch.Tensor
    scratch_locs: torch.Tensor
    scratch_state: torch.Tensor


def allocate_window_hash_states(
    layout: HiSparseWindowHashLayout, *, num_layers: int, num_req_slots: int, device
) -> list[HiSparseWindowHashState]:
    states = []
    for _ in range(num_layers):
        states.append(
            HiSparseWindowHashState(
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
                # Preserve main's fixed physical slot mapping. No cache entry
                # rotates into DSpark's C4 writer scratch or canonical tail.
                torch.empty((num_req_slots, 0), dtype=torch.int32, device=device),
                torch.zeros(
                    (num_req_slots + 1, layout.state_width(num_req_slots)),
                    dtype=torch.int32,
                    device=device,
                ),
            )
        )
    return states


@functools.cache
def _jit_window_hash_module(hot_slots: int, verify_width: int, top_k: int):
    args = make_cpp_args(
        64, top_k, hot_slots, verify_width, True, "MetadataOnlyKVTransferPolicy", False
    )
    return load_jit(
        "hisparse_window_hash",
        *args,
        cuda_files=["hisparse_window_hash.cuh"],
        cuda_wrappers=[
            (
                "plan_window",
                f"load_cache_to_device_buffer_spec<{args}>",
            ),
        ],
    )


@functools.cache
def _jit_window_hash_init_module(hot_slots: int):
    return load_jit(
        "hisparse_window_hash_init",
        hot_slots,
        cuda_files=["hisparse_window_hash.cuh"],
        cuda_wrappers=[
            (
                "initialize",
                f"initialize_hisparse_spec_state<{hot_slots}>",
            ),
        ],
    )


def initialize_window_hash_state(
    *,
    device_buffer_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    state: HiSparseWindowHashState,
) -> None:
    if req_pool_indices.numel() == 0:
        return
    _jit_window_hash_init_module(state.cache_policy.size(1)).initialize(
        device_buffer_tokens,
        req_pool_indices.to(torch.int64),
        state.cache_index,
        state.cache_policy,
        state.scratch_state,
    )


def plan_cache_window_hash_mla(
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
    state: HiSparseWindowHashState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
) -> None:
    if top_k_tokens.ndim != 3:
        raise ValueError("Window hash Top-K must have shape [batch, steps, top_k].")
    batch, steps, top_k = top_k_tokens.shape
    layout = HiSparseWindowHashLayout(state.cache_policy.size(1), steps, top_k)
    if top_k_device_locs.shape != top_k_tokens.shape:
        raise ValueError("Window hash output must match the Top-K shape.")
    for tensor in (top_k_tokens, top_k_device_locs):
        if (
            tensor.dtype != torch.int32
            or tensor.stride(2) != 1
            or tensor.stride(1) != top_k
        ):
            raise ValueError("Window hash Top-K and output need contiguous int32 rows.")
    if req_pool_indices.shape != (batch,) or req_pool_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Window hash needs one integer request index per request.")
    if seq_lens.numel() not in (batch, batch * steps) or seq_lens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Window hash needs integer lengths per request or Top-K row.")
    if (
        device_buffer_tokens.dtype != torch.int32
        or device_buffer_locs.dtype != torch.int32
        or device_buffer_tokens.shape != device_buffer_locs.shape
        or device_buffer_tokens.stride() != device_buffer_locs.stride()
        or device_buffer_tokens.stride(1) != 1
        or device_buffer_tokens.size(1) != layout.hot_slots + 1
    ):
        raise ValueError(
            "Window hash requires matching int32 hot slots plus one canonical tail slot."
        )
    if (
        miss_src.dtype != torch.int64
        or miss_dst.dtype != torch.int32
        or miss_count.dtype != torch.int32
        or miss_src.ndim != 2
        or miss_dst.ndim != 2
        or miss_src.size(0) < batch
        or miss_dst.size(0) < batch
        or miss_src.size(1) < layout.occurrences
        or miss_dst.size(1) < layout.occurrences
        or miss_src.stride(1) != 1
        or miss_dst.stride(1) != 1
        or miss_src.stride(0) != miss_dst.stride(0)
        or miss_count.ndim != 1
        or miss_count.numel() < batch
    ):
        raise ValueError(
            "Window hash miss-plan buffers must cover batch * steps * top_k."
        )
    if state.scratch_locs.size(1) != 0:
        raise ValueError(
            "Window hash preserves main's fixed slots and requires empty cache scratch."
        )
    if batch == 0:
        return
    lengths = seq_lens.reshape(batch, -1)
    if lengths.size(1) == 1:
        lengths = lengths.expand(batch, steps)
    _jit_window_hash_module(layout.hot_slots, steps, top_k).plan_window(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices.to(torch.int64),
        lengths.to(torch.int32).contiguous().view(-1),
        *state,
        num_real_reqs,
        1,
        miss_src,
        miss_dst,
        miss_count,
    )
