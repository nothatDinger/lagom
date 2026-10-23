"""Exact union resolution and post-attention promotion for Adaptive DHiSparse."""

from __future__ import annotations

from functools import cache

import torch


@cache
def _module(item_bytes: int, packed: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(item_bytes, packed)
    return load_jit(
        "dhisparse",
        *args,
        cuda_files=["dhisparse.cuh"],
        cuda_wrappers=[
            ("resolve", f"dhisparse_resolve<{args}>"),
            ("promote", f"dhisparse_promote<{args}>"),
        ],
    )


def allocate_workspace(requests: int, occurrences: int, hot: int, device):
    hash_size = 1 << (2 * (hot + occurrences) - 1).bit_length()
    return dict(
        keys=torch.full((requests, hash_size), -1, dtype=torch.int32, device=device),
        locations=torch.empty((requests, hash_size), dtype=torch.int32, device=device),
        hot_indices=torch.empty(
            (requests, hash_size), dtype=torch.int32, device=device
        ),
        miss_src=torch.full(
            (requests, occurrences), -1, dtype=torch.int64, device=device
        ),
        miss_tokens=torch.full(
            (requests, occurrences), -1, dtype=torch.int32, device=device
        ),
    )


def resolve(
    *,
    selections,
    request_ids,
    seq_lens,
    optional_locs,
    host_locs,
    buffer_tokens,
    buffer_locs,
    hot_ages,
    epochs,
    host,
    device_buffer,
    staging_offset: int,
    staging_stride: int,
    hot_size: int,
    output,
    workspace,
    num_real_requests,
    item_bytes: int,
    packed: bool = False,
):
    """Resolve [batch, queries, k], copying each distinct missing row once.

    Tail identities are explicit, including tentative rows; query sequence lengths
    determine visibility. Optional locations contain only completed copies. Scratch
    is indexed by request slot, not batch order. No hot row is overwritten here.
    """
    if selections.ndim != 3 or seq_lens.shape != selections.shape[:2]:
        raise ValueError("Selection/causal-bound geometry mismatch")
    if selections.shape[1] * selections.shape[2] > staging_stride:
        raise ValueError("Verification window exceeds reserved staging")
    if selections.numel() == 0:
        return output
    if selections.device.type == "cpu":
        return _resolve_cpu(
            selections=selections,
            request_ids=request_ids,
            seq_lens=seq_lens,
            optional_locs=optional_locs,
            host_locs=host_locs,
            buffer_tokens=buffer_tokens,
            buffer_locs=buffer_locs,
            hot_ages=hot_ages,
            epochs=epochs,
            host=host,
            device_buffer=device_buffer,
            staging_offset=staging_offset,
            staging_stride=staging_stride,
            hot_size=hot_size,
            output=output,
            workspace=workspace,
            num_real_requests=num_real_requests,
            packed=packed,
        )
    _module(item_bytes, packed).resolve(
        selections,
        request_ids.to(torch.int64),
        seq_lens.to(torch.int32),
        optional_locs,
        host_locs,
        buffer_tokens,
        buffer_locs,
        hot_ages,
        epochs,
        host,
        device_buffer,
        output,
        workspace["keys"],
        workspace["locations"],
        workspace["hot_indices"],
        workspace["miss_src"],
        workspace["miss_tokens"],
        num_real_requests,
        staging_offset,
        staging_stride,
        hot_size,
    )
    return output


def promote(
    *,
    request_ids,
    buffer_tokens,
    buffer_locs,
    hot_ages,
    epochs,
    device_buffer,
    staging_offset,
    staging_stride,
    hot_size,
    workspace,
    num_real_requests,
    item_bytes,
    packed=False,
):
    """Copy staged admissions to stable hot slots after all attention readers.

    Arenas must not be reused before this operation completes. Sorting is over the
    bounded hot set, independent of optional history or full context length.
    """
    if request_ids.numel() == 0:
        return
    victims = torch.argsort(hot_ages, dim=-1, stable=True).to(torch.int32)
    ranks = (
        torch.cumsum(
            (workspace["miss_src"] >= 0).to(torch.int32), dim=-1, dtype=torch.int32
        )
        - 1
    )
    if device_buffer.device.type == "cpu":
        for request in request_ids[: int(num_real_requests[0])].tolist():
            for occurrence in range(workspace["miss_src"].shape[1]):
                if workspace["miss_src"][request, occurrence] < 0:
                    continue
                rank = int(ranks[request, occurrence])
                if rank >= hot_size:
                    continue
                victim = int(victims[request, rank])
                src = staging_offset + request * staging_stride + occurrence
                dst = int(buffer_locs[request, victim])
                _copy_cpu(device_buffer, device_buffer, src, dst, packed)
                buffer_tokens[request, victim] = workspace["miss_tokens"][
                    request, occurrence
                ]
                hot_ages[request, victim] = epochs[request]
        return
    _module(item_bytes, packed).promote(
        request_ids.to(torch.int64),
        buffer_tokens,
        buffer_locs,
        hot_ages,
        epochs,
        device_buffer,
        workspace["miss_src"],
        workspace["miss_tokens"],
        victims,
        ranks,
        num_real_requests,
        staging_offset,
        staging_stride,
        hot_size,
    )


def _copy_cpu(src, dst, src_loc, dst_loc, packed):
    if packed:
        # Each page stores 64 contiguous 576-byte values, then 64 scales.
        # RoPE bytes are part of the value; never reinterpret quantized data.
        for offset, width in ((0, 576), (64 * 576, 8)):
            si = offset + src_loc % 64 * width
            di = offset + dst_loc % 64 * width
            dst[dst_loc // 64, di : di + width].copy_(
                src[src_loc // 64, si : si + width].clone()
            )
    else:
        dst[dst_loc].copy_(src[src_loc].clone())


def _resolve_cpu(**kw):
    selected, out = kw["selections"], kw["output"]
    workspace = kw["workspace"]
    out.fill_(-1)
    real = int(kw["num_real_requests"][0])
    for batch, request in enumerate(kw["request_ids"][:real].tolist()):
        workspace["miss_src"][request].fill_(-1)
        workspace["miss_tokens"][request].fill_(-1)
        kw["epochs"][request] += 1
        union = {}
        hot = {
            int(token): slot
            for slot, token in enumerate(
                kw["buffer_tokens"][request, : kw["hot_size"]].tolist()
            )
            if token >= 0
        }
        tail = {
            int(token): int(kw["buffer_locs"][request, slot])
            for slot, token in enumerate(kw["buffer_tokens"][request].tolist())
            if slot >= kw["hot_size"] and token >= 0
        }
        for query in range(selected.shape[1]):
            for col in range(selected.shape[2]):
                token = int(selected[batch, query, col])
                if token < 0 or token >= int(kw["seq_lens"][batch, query]):
                    continue
                if token >= kw["optional_locs"].shape[1]:
                    raise ValueError("Selected row exceeds logical history capacity")
                location = tail.get(token, -1)
                if location < 0:
                    location = int(kw["optional_locs"][request, token])
                if location < 0 and token in hot:
                    slot = hot[token]
                    location = int(kw["buffer_locs"][request, slot])
                    kw["hot_ages"][request, slot] = kw["epochs"][request]
                if location < 0:
                    if token not in union:
                        src = int(kw["host_locs"][request, token])
                        if src < 0:
                            raise RuntimeError(
                                "Selected row has neither ready GPU nor host backing"
                            )
                        occurrence = query * selected.shape[2] + col
                        location = (
                            kw["staging_offset"]
                            + request * kw["staging_stride"]
                            + occurrence
                        )
                        union[token] = location
                        workspace["miss_src"][request, occurrence] = src
                        workspace["miss_tokens"][request, occurrence] = token
                        _copy_cpu(
                            kw["host"], kw["device_buffer"], src, location, kw["packed"]
                        )
                    location = union[token]
                out[batch, query, col] = location
    return out
