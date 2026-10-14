from __future__ import annotations

import functools
import logging
import os
from typing import TYPE_CHECKING

import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _dsv4_copy_load_policy_from_env() -> str:
    policy = os.environ.get("SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY", "legacy")
    if policy not in ("legacy", "ca", "cg"):
        raise ValueError(f"Invalid SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY={policy!r}")
    logging.getLogger(__name__).info(
        "HiSparse DSv4 planned-copy load policy: %s", policy
    )
    return policy


def _resolve_copy_load_policy(policy: str | None, is_dsv4_layout: bool) -> int:
    if policy is None:
        policy = _dsv4_copy_load_policy_from_env() if is_dsv4_layout else "legacy"
    policies = {"legacy": 0, "ca": 1, "cg": 2, "ca_pair": 3, "ca_v2": 4}
    if policy not in policies:
        raise ValueError(f"Invalid planned-copy load policy: {policy!r}")
    if policy != "legacy" and (not is_dsv4_layout or torch.version.hip is not None):
        raise ValueError(
            "Explicit planned-copy cache policies require NVIDIA DSv4 layout"
        )
    return policies[policy]


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
    record_miss_plan: bool = False,
    skip_io: bool = False,
    verify_width: int = 1,
) -> Module:
    # record_miss_plan / skip_io are compile-time kernel flags; the
    # (False, False) production instantiation stays byte-identical.
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        record_miss_plan,
        skip_io,
        verify_width,
    )
    cache_args = make_cpp_args(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        record_miss_plan,
        skip_io,
        verify_width,
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_copy_planned_module(
    block_size: int,
    is_mla: bool,
    is_dsv4_layout: bool,
    skip_io: bool,
    load_policy: int = 0,
) -> Module:
    template_args = make_cpp_args(
        block_size, is_mla, is_dsv4_layout, skip_io, load_policy
    )
    return load_jit(
        "sparse_copy_planned",
        block_size,
        is_mla,
        is_dsv4_layout,
        skip_io,
        load_policy,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned",
                f"copy_cache_planned<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_dsv4_transfer_module(block_size: int) -> Module:
    template_args = make_cpp_args(block_size)
    return load_jit(
        "sparse_cache_dsv4_transfer",
        block_size,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "transfer_cache_dsv4_mla",
                f"transfer_cache_dsv4_mla<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_copy_iterator_module(block_size: int, depth: int) -> Module:
    template_args = make_cpp_args(block_size, depth)
    return load_jit(
        "sparse_copy_iterator",
        block_size,
        depth,
        cuda_files=["hisparse_copy_iterator.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned_iterator",
                f"copy_cache_planned_iterator<{template_args}>",
            )
        ],
    )


def copy_cache_planned_mla_iterator(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    depth: int,
    num_blocks: int = 4,
    block_size: int = 1024,
) -> None:
    """Diagnostic CUDA DSv4 C4 copy with one/two KV slots per warp.

    Separate JIT entry point; the service does not select it via environment.
    Both depths use legacy scalar loads and identical traversal. Inspect actual
    codegen for staging and spills before interpreting performance.
    """
    if depth not in (1, 2) or torch.version.hip is not None:
        raise ValueError("Iterator copy requires NVIDIA CUDA and depth 1 or 2")
    if num_blocks != 4 or block_size != 1024:
        raise ValueError("Iterator diagnostic fixes the launch at CTA4/1024")
    if (
        miss_src.dtype != torch.int64
        or miss_dst.dtype != torch.int32
        or miss_count.dtype != torch.int32
        or num_real_reqs.dtype != torch.int32
        or miss_src.ndim != 2
        or miss_dst.shape != miss_src.shape
        or miss_src.stride(1) != 1
        or miss_dst.stride(1) != 1
        or miss_src.stride(0) != miss_dst.stride(0)
        or miss_count.shape != (miss_src.shape[0],)
        or not miss_count.is_contiguous()
        or num_real_reqs.numel() != 1
    ):
        raise ValueError("Invalid iterator miss plan/counts shape, dtype or stride")
    if any(
        tensor.dtype != torch.uint8
        or tensor.ndim != 2
        or tensor.shape[1] != 37440
        or not tensor.is_contiguous()
        for tensor in (host_cache, device_buffer)
    ):
        raise ValueError(
            "Iterator copy requires contiguous uint8 C4 rows of 37440 bytes"
        )
    if miss_src.device.type != "cuda" or any(
        tensor.device != miss_src.device
        for tensor in (miss_dst, miss_count, num_real_reqs, device_buffer)
    ):
        raise ValueError(
            "Iterator plans, counts and destination must share one CUDA device"
        )
    if host_cache.device.type != "cpu" and host_cache.device != miss_src.device:
        raise ValueError(
            "Iterator source must be pinned CPU memory or the same CUDA device"
        )
    module = _jit_copy_iterator_module(block_size, depth)
    module.copy_cache_planned_iterator(
        miss_src,
        miss_dst,
        miss_count,
        num_real_reqs,
        host_cache,
        device_buffer,
        num_blocks,
    )


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers."""
    module = _jit_dsv4_transfer_module(block_size)
    module.transfer_cache_dsv4_mla(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
    )


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
    miss_src: torch.Tensor | None,
    miss_dst: torch.Tensor | None,
    miss_count: torch.Tensor | None,
    skip_io: bool,
    verify_width: int = 1,
) -> None:
    if verify_width <= 0:
        raise ValueError(f"verify_width must be positive, got {verify_width}")
    required_window_capacity = verify_width * num_top_k
    if hot_buffer_size < required_window_capacity:
        raise ValueError(
            f"hot_buffer_size ({hot_buffer_size}) must be >= verify_width * "
            f"num_top_k ({verify_width} * {num_top_k} = "
            f"{required_window_capacity})"
        )

    if top_k_tokens.size(0) % verify_width != 0:
        raise ValueError(
            f"Top-K rows ({top_k_tokens.size(0)}) must be divisible by "
            f"verify_width ({verify_width})"
        )
    num_reqs = top_k_tokens.size(0) // verify_width
    if seq_lens.numel() not in (num_reqs, top_k_tokens.size(0)):
        raise ValueError(
            "Sequence lengths must contain one value per request or Top-K row: "
            f"lengths={seq_lens.numel()}, requests={num_reqs}, "
            f"rows={top_k_tokens.size(0)}"
        )

    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together"
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32")
        if miss_src.dim() != 2 or miss_dst.dim() != 2:
            raise ValueError("miss_src and miss_dst must be two-dimensional")
        if miss_src.size(0) < num_reqs or miss_dst.size(0) < num_reqs:
            raise ValueError(
                "Miss-plan buffers must contain one row per request: "
                f"requests={num_reqs}, src_rows={miss_src.size(0)}, "
                f"dst_rows={miss_dst.size(0)}"
            )
        if (
            miss_src.size(1) < required_window_capacity
            or miss_dst.size(1) < required_window_capacity
        ):
            raise ValueError(
                "Each miss-plan row must hold verify_width * num_top_k entries: "
                f"required={required_window_capacity}, "
                f"src_capacity={miss_src.size(1)}, "
                f"dst_capacity={miss_dst.size(1)}"
            )
        if miss_count.numel() < num_reqs:
            raise ValueError(
                "miss_count must contain one value per request: "
                f"required={num_reqs}, capacity={miss_count.numel()}"
            )
        if miss_src.stride(1) != 1 or miss_dst.stride(1) != 1:
            raise ValueError("Miss-plan entries must be contiguous within each row")
        if (
            miss_src.stride(0) < required_window_capacity
            or miss_dst.stride(0) < required_window_capacity
            or miss_src.stride(0) != miss_dst.stride(0)
        ):
            raise ValueError(
                "Miss-plan row strides must match and cover the verify window: "
                f"required={required_window_capacity}, "
                f"src_stride={miss_src.stride(0)}, "
                f"dst_stride={miss_dst.stride(0)}"
            )

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
        record_miss_plan=record_miss_plan,
        skip_io=skip_io,
        verify_width=verify_width,
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0) // verify_width],
            dtype=torch.int32,
            device=top_k_tokens.device,
        )

    if record_miss_plan:
        assert miss_dst is not None and miss_count is not None
    else:
        # Unused sentinels; the RecordMissPlan=false instantiation never reads them.
        miss_src = miss_dst = miss_count = empty

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        miss_src,
        miss_dst,
        miss_count,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
    verify_width: int = 1,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    Optional miss_src/miss_dst/miss_count record the miss plan for replay by
    copy_cache_planned_mla; skip_io elides only the KV bytes (timing probe).
    """
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
        verify_width=verify_width,
    )


def copy_cache_planned_mla(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_blocks: int = 4,
    block_size: int = 1024,
    is_dsv4_layout: bool = False,
    skip_io: bool = False,
    load_policy: str | None = None,
) -> None:
    """Replay a miss plan for a verify window or prefetched skip layer.

    IO-only, no planning; the caller chooses the stream and owns the slot table.
    DSv4 ``load_policy=None`` uses SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY, resolved
    once per process (default: legacy). Explicit legacy/ca/cg overrides that value.
    Restart the process and recapture CUDA graphs when changing the environment.
    ``ca_pair``/``ca_v2`` are explicit diagnostic-only arguments, unavailable
    through the environment. The latter requires a 16-byte aligned source base.
    """
    assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
    assert miss_count.dtype == torch.int32
    policy = _resolve_copy_load_policy(load_policy, is_dsv4_layout)
    if policy == 4 and host_cache.data_ptr() % 16:
        raise ValueError("ca_v2 requires a 16-byte aligned source base")
    module = _jit_copy_planned_module(block_size, True, is_dsv4_layout, skip_io, policy)
    empty = torch.empty(0)
    module.copy_cache_planned(
        miss_src,
        miss_dst,
        miss_count,
        num_real_reqs,
        host_cache,
        empty,
        device_buffer,
        empty,
        num_blocks,
        item_size_bytes,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
    verify_width: int = 1,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
        verify_width=verify_width,
    )
