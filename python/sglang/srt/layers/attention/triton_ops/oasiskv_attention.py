"""Physical-page GQA decode: shared KV tiles and tensor-core QK/PV.

The descriptor buffer holds addresses/lengths only; KV stays in its resident
pages. Temporary shared memory used by dot/pipelining is not HBM KV staging.
"""

import triton
import triton.language as tl


@triton.jit
def _tile(
    q,
    ptr,
    valid,
    acc,
    m,
    z,
    KD: tl.constexpr,
    VD: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    SCALE: tl.constexpr,
    CAP: tl.constexpr,
):
    dk, dv = tl.arange(0, DK), tl.arange(0, DV)
    # Resident records are 16-byte aligned, including padded odd dimensions.
    # Preserve that fact after loading an absolute address from the descriptor.
    ptr = tl.multiple_of(ptr, 16)
    kp = tl.max_contiguous(tl.multiple_of(ptr[:, None] + dk[None, :], (1, 16)), (1, 8))
    k = tl.load(kp, valid[:, None] & (dk[None, :] < KD), 0)
    logits = tl.dot(q, tl.trans(k)) * SCALE
    if CAP > 0:
        logits = CAP * (2 / (1 + tl.exp(-2 * logits / CAP)) - 1)
    logits = tl.where(valid[None, :], logits, -float("inf"))
    new_m = tl.maximum(m, tl.max(logits, 1))
    safe_m = tl.where(new_m == -float("inf"), 0, new_m)
    alpha = tl.exp(m - safe_m)
    p = tl.exp(logits - safe_m[:, None])
    # Do not keep expanded FP32 K and V alive together. Matrix operands remain
    # in the cache dtype, with FP32 scores, normalization and accumulation.
    VA: tl.constexpr = min(16, 2 * (KD & -KD))
    vp = tl.max_contiguous(
        tl.multiple_of(ptr[:, None] + KD + dv[None, :], (1, VA)), (1, 8)
    )
    v = tl.load(vp, valid[:, None] & (dv[None, :] < VD), 0)
    acc = tl.dot(p.to(q.dtype), v, acc * alpha[:, None])
    z = z * alpha + tl.sum(p, 1)
    return acc, new_m, z


@triton.jit
def attention_grouped(
    Pages,
    Query,
    Key,
    Value,
    Partial,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    STORAGE: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    CAP: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    M: tl.constexpr,
    TILE: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
):
    b, hg, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    chunks: tl.constexpr = triton.cdiv(GROUP, HEAD_TILE)
    h, chunk = hg // chunks, hg % chunks
    rows = tl.arange(0, M)
    gh, lane = chunk * HEAD_TILE + rows // WIDTH, rows % WIDTH
    row_valid = (rows < HEAD_TILE * WIDTH) & (gh < GROUP)
    qh = h * GROUP + gh
    descriptors = Pages + (b * HEADS + h) * (SLOTS + 1) * 2
    active_length = tl.load(descriptors + SLOTS * 2 + 1)
    dk, dv = tl.arange(0, DK), tl.arange(0, DV)
    acc = tl.full((M, DV), 0, tl.float32)
    m, z = tl.full((M,), -float("inf"), tl.float32), tl.full((M,), 0, tl.float32)
    # Skip graph padding before loading request data or performing matrix work.
    if active_length > 0:
        q = tl.load(
            Query
            + ((b * WIDTH + lane[:, None]) * HEADS * GROUP + qh[:, None]) * KD
            + dk[None, :],
            row_valid[:, None] & (active_length > 0) & (dk[None, :] < KD),
            0,
        )
        dtype: tl.constexpr = Key.dtype.element_ty
        # Whole-page tiles amortize each descriptor load across BLOCK tokens.
        pages_per_tile: tl.constexpr = TILE // BLOCK
        ps = tl.arange(0, pages_per_tile)
        within = tl.arange(0, BLOCK)
        for t in tl.range(TILES_PER_SPLIT, num_stages=1):
            slots = (split * TILES_PER_SPLIT + t) * pages_per_tile + ps
            addresses = tl.load(descriptors + slots * 2, slots < SLOTS, 0)
            addresses = tl.multiple_of(addresses, 16)
            lengths = tl.load(descriptors + slots * 2 + 1, slots < SLOTS, 0)
            ptr = tl.reshape(
                addresses.to(tl.pointer_type(dtype))[:, None]
                + within[None, :] * STORAGE,
                (TILE,),
            )
            valid = tl.reshape(within[None, :] < lengths[:, None], (TILE,))
            acc, m, z = _tile(q, ptr, valid, acc, m, z, KD, VD, DK, DV, SCALE, CAP)

        # Active-block and probe visibility are not branches/masks in the hot loop.
        if split == 0:
            address = tl.load(descriptors + SLOTS * 2)
            length = active_length
            active_n = tl.arange(0, max(16, BLOCK))
            active_ptr = address.to(tl.pointer_type(dtype)) + active_n * STORAGE
            acc, m, z = _tile(
                q, active_ptr, active_n < length, acc, m, z, KD, VD, DK, DV, SCALE, CAP
            )
            if WIDTH == 2:
                draft_k = tl.load(
                    Key + ((b * WIDTH + 1) * HEADS + h) * KD + dk,
                    (length > 0) & (dk < KD),
                    0,
                ).to(tl.float32)
                logits = tl.sum(q.to(tl.float32) * draft_k[None, :], 1) * SCALE
                if CAP > 0:
                    logits = CAP * (2 / (1 + tl.exp(-2 * logits / CAP)) - 1)
                logits = tl.where((lane == 1) & (length > 0), logits, -float("inf"))
                new_m = tl.maximum(m, logits)
                safe_m = tl.where(new_m == -float("inf"), 0, new_m)
                alpha, p = tl.exp(m - safe_m), tl.exp(logits - safe_m)
                draft_v = tl.load(
                    Value + ((b * WIDTH + 1) * HEADS + h) * VD + dv,
                    (length > 0) & (dv < VD),
                    0,
                ).to(tl.float32)
                acc = acc * alpha[:, None] + p[:, None] * draft_v[None, :]
                z, m = z * alpha + p, new_m
    offset = (((b * WIDTH + lane) * HEADS * GROUP + qh) * SPLITS + split) * (VD + 2)
    tl.store(
        Partial + offset[:, None] + dv[None, :],
        acc,
        row_valid[:, None] & (dv[None, :] < VD),
    )
    tl.store(Partial + offset + VD, m, row_valid)
    tl.store(Partial + offset + VD + 1, z, row_valid)
