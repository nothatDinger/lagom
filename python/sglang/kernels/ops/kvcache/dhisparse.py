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
            ("reread", f"dhisparse_reread<{args}>"),
            ("promote", f"dhisparse_promote<{args}>"),
        ],
    )


def allocate_workspace(
    requests: int, occurrences: int, hot: int, device, *, track_hits=False
):
    hash_size = 1 << (2 * (hot + occurrences) - 1).bit_length()
    return dict(
        hit_src=torch.empty(
            (requests, occurrences) if track_hits else (0,),
            dtype=torch.int64,
            device=device,
        ),
        miss_order=torch.empty(
            (requests, occurrences), dtype=torch.int32, device=device
        ),
        victims=torch.empty((requests, hot), dtype=torch.int32, device=device),
        activity=torch.empty(requests, dtype=torch.int32, device=device),
        empty_counts=torch.empty(0, dtype=torch.int32, device=device),
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
    counts=None,
    packed: bool = False,
):
    """Resolve [batch, queries, k], copying each distinct missing row once.

    Tail identities are explicit, including tentative rows; query sequence lengths
    determine visibility. Optional locations contain only completed copies. Scratch
    uses compact batch rows; persistent identities use request slots. No hot row is overwritten here.
    """
    if selections.ndim != 3 or seq_lens.shape != selections.shape[:2]:
        raise ValueError("Selection/causal-bound geometry mismatch")
    occurrences = selections.shape[1] * selections.shape[2]
    if (
        occurrences > min(staging_stride, workspace["miss_src"].shape[1])
        or selections.size(0) > workspace["miss_src"].size(0)
        or request_ids.numel() != selections.size(0)
    ):
        raise ValueError("Verification batch/window exceeds reserved staging")
    if counts is None:
        counts = workspace["empty_counts"]
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
            counts=counts,
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
        workspace["hit_src"],
        workspace["miss_order"],
        workspace["activity"],
        counts,
        num_real_requests,
        staging_offset,
        staging_stride,
        hot_size,
    )
    return output


def reread(
    *,
    request_ids,
    host,
    device_buffer,
    workspace,
    num_real_requests,
    staging_offset,
    staging_stride,
    item_bytes,
    packed=False,
):
    """Repeat host-backed hot hits into unused arena rows; keep the normal selection and LRU."""
    hits = workspace["hit_src"]
    if not hits.numel():
        raise ValueError("Hit tracking must be enabled for a forced reread")
    if not request_ids.numel():
        return
    if device_buffer.device.type == "cpu":
        for batch in range(int(num_real_requests[0])):
            for i in torch.where(hits[batch] >= 0)[0].tolist():
                _copy_cpu(
                    host,
                    device_buffer,
                    int(hits[batch, i]),
                    staging_offset + batch * staging_stride + i,
                    packed,
                )
        return
    _module(item_bytes, packed).reread(
        request_ids.to(torch.int64),
        hits,
        host,
        device_buffer,
        num_real_requests,
        staging_offset,
        staging_stride,
    )


def promote(
    *,
    request_ids,
    buffer_tokens,
    buffer_locs,
    hot_ages,
    lru_slots,
    epochs,
    device_buffer,
    staging_offset,
    staging_stride,
    hot_size,
    workspace,
    num_real_requests,
    item_bytes,
    counts=None,
    packed=False,
):
    """Copy staged admissions to stable hot slots after all attention readers.

    Arenas must not be reused before this operation completes. The bounded LRU is scanned
    only for requests with unresolved selections.
    """
    if request_ids.numel() == 0:
        return
    if counts is None:
        counts = workspace["empty_counts"]
    if device_buffer.device.type == "cpu":
        for batch, request in enumerate(
            request_ids[: int(num_real_requests[0])].tolist()
        ):
            if workspace["activity"][batch] < 0:
                continue
            lru = lru_slots[request].tolist()
            cold = [slot for slot in lru if hot_ages[request, slot] != epochs[request]]
            warm = [slot for slot in lru if hot_ages[request, slot] == epochs[request]]
            occurrences = torch.where(workspace["miss_src"][batch] >= 0)[0].tolist()
            admitted = min(len(cold), len(occurrences))
            lru_slots[request] = torch.tensor(
                cold[admitted:] + warm + cold[:admitted], dtype=torch.int32
            )
            for victim, occurrence in zip(cold, occurrences):
                if counts.numel():
                    counts[3] += 1
                src = staging_offset + batch * staging_stride + occurrence
                dst = int(buffer_locs[request, victim])
                _copy_cpu(device_buffer, device_buffer, src, dst, packed)
                buffer_tokens[request, victim] = workspace["miss_tokens"][
                    batch, occurrence
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
        workspace["victims"],
        workspace["miss_order"],
        workspace["activity"],
        lru_slots,
        counts,
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
        workspace["activity"][batch] = -1
        workspace["miss_src"][batch].fill_(-1)
        workspace["miss_tokens"][batch].fill_(-1)
        hits = workspace["hit_src"]
        if hits.numel():
            hits[batch].fill_(-1)
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
                if location < 0:
                    workspace["activity"][batch] = 0
                if location < 0 and token in hot:
                    slot = hot[token]
                    location = int(kw["buffer_locs"][request, slot])
                    if (
                        hits.numel()
                        and kw["hot_ages"][request, slot] != kw["epochs"][request]
                        and kw["seq_lens"][batch, 0] > kw["hot_size"]
                    ):
                        src = int(kw["host_locs"][request, token])
                        if src >= 0:
                            hits[batch, query * selected.shape[2] + col] = src
                            if kw["counts"].numel():
                                kw["counts"][1] += 1
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
                            + batch * kw["staging_stride"]
                            + occurrence
                        )
                        union[token] = location
                        if kw["counts"].numel():
                            kw["counts"][0] += 1
                        workspace["miss_src"][batch, occurrence] = src
                        workspace["miss_tokens"][batch, occurrence] = token
                        _copy_cpu(
                            kw["host"], kw["device_buffer"], src, location, kw["packed"]
                        )
                    location = union[token]
                out[batch, query, col] = location
    return out
