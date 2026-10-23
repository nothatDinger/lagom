"""Fixed-shape OasisKV kernels, including indirect pinned-host UVA access.

Only the request table's address is captured. Its pointer/length/capacity values
are refreshed before replay, so requests may be reordered, recycled or grown.
An inactive row has prefix=-1; it must never dereference a request pointer.
"""

import triton
import triton.language as tl

# Request table columns. Keep this layout in sync with OasisKVGraphState.
HOST = 0
DATA = 1
LOW = 2
HIGH = 3
RESIDENT = 4
AGES = 5
CAPACITY = 6
PREFIX = 7
STEP = 8
FIELDS = 9


@triton.jit
def append_token(
    Table,
    Key,
    Value,
    HEADS: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    STORAGE: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    NATIVE: tl.constexpr,
    D: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    row = Table + b * 9
    pos = tl.load(row + 7)
    if pos >= 0:
        dtype = Key.dtype.element_ty
        host = tl.load(row + 0).to(tl.pointer_type(dtype))
        data = tl.load(row + 1).to(tl.pointer_type(dtype))
        capacity = tl.load(row + 6)
        d = tl.arange(0, D)
        k = tl.load(Key + (b * WIDTH * HEADS + h) * KD + d, d < KD, 0)
        v = tl.load(
            Value + (b * WIDTH * HEADS + h) * VD + d - KD, (d >= KD) & (d < KD + VD), 0
        )
        packed = tl.where(d < KD, k, v)
        offset = pos % BLOCK
        tl.store(host + (h * capacity * BLOCK + pos) * STORAGE + d, packed, d < STORAGE)
        tl.store(
            data + ((h * (SLOTS + 1) + SLOTS) * BLOCK + offset) * STORAGE + d,
            packed,
            d < STORAGE,
        )
        if not NATIVE:
            lo = tl.load(row + 2).to(tl.pointer_type(tl.float32))
            hi = tl.load(row + 3).to(tl.pointer_type(tl.float32))
            index = (h * capacity + pos // BLOCK) * KD + d
            old_lo = tl.load(lo + index, (d < KD) & (offset > 0), float("inf"))
            old_hi = tl.load(hi + index, (d < KD) & (offset > 0), -float("inf"))
            tl.store(lo + index, tl.minimum(old_lo, k.to(tl.float32)), d < KD)
            tl.store(hi + index, tl.maximum(old_hi, k.to(tl.float32)), d < KD)


@triton.jit
def score_blocks(
    Table,
    Query,
    Scores,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    KD: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
    D: tl.constexpr,
    TILE: tl.constexpr,
):
    bh, lane, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = bh // HEADS, bh % HEADS
    row = Table + b * 9
    pos = tl.load(row + 7)
    blocks = tile * TILE + tl.arange(0, TILE)
    scores = tl.full((TILE,), -float("inf"), tl.float32)
    if pos >= 0:
        capacity = tl.load(row + 6)
        lo = tl.load(row + 2).to(tl.pointer_type(tl.float32))
        hi = tl.load(row + 3).to(tl.pointer_type(tl.float32))
        # Real query excludes the active block; lookahead may predict a block
        # that the committed root just completed.
        complete = (pos + lane) // BLOCK
        d = tl.arange(0, D)
        mask = (blocks[:, None] < complete) & (d[None, :] < KD)
        offsets = (h * capacity + blocks[:, None]) * KD + d[None, :]
        low = tl.load(lo + offsets, mask, 0)
        high = tl.load(hi + offsets, mask, 0)
        for g in range(GROUP):
            q = tl.load(
                Query + ((b * WIDTH + lane) * HEADS * GROUP + h * GROUP + g) * KD + d,
                d < KD,
                0,
            ).to(tl.float32)
            bound = tl.sum(tl.maximum(low * q[None, :], high * q[None, :]), 1)
            scores = tl.maximum(scores, bound)
        scores = tl.where(blocks < complete, scores, -float("inf"))
    # Scores always reserve two lanes, also for ordinary width-one decode.
    tl.store(
        Scores + ((b * 2 + lane) * HEADS + h) * MAX_BLOCKS + blocks,
        scores,
        blocks < MAX_BLOCKS,
    )


@triton.jit
def native_candidates(
    Table,
    TopK,
    Real,
    Draft,
    HEADS: tl.constexpr,
    SLOTS: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    pos = tl.load(Table + b * 9 + 7)
    i = tl.arange(0, N)
    real = tl.load(TopK + b * WIDTH * TOPK + i, i < TOPK, -1)
    real = tl.where((real >= 0) & (real < pos) & (pos >= 0), real, -1)
    tl.store(Real + (b * HEADS + h) * SLOTS + i, real, i < SLOTS)
    if WIDTH == 2:
        draft = tl.load(TopK + (b * WIDTH + 1) * TOPK + i, i < TOPK, -1)
        draft = tl.where((draft >= 0) & (draft < pos + 1) & (pos >= 0), draft, -1)
        tl.store(Draft + (b * HEADS + h) * SLOTS + i, draft, i < SLOTS)


@triton.jit
def _contains(sorted_ids, values, N: tl.constexpr, STEPS: tl.constexpr):
    low = tl.full((N,), 0, tl.int32)
    high = tl.full((N,), N, tl.int32)
    for _ in range(STEPS):
        mid = (low + high) // 2
        value = tl.gather(sorted_ids, tl.minimum(mid, N - 1), 0)
        smaller = value < values
        low = tl.where(smaller, mid + 1, low)
        high = tl.where(smaller, high, mid)
    found = tl.gather(sorted_ids, tl.minimum(low, N - 1), 0)
    return (values >= 0) & (low < N) & (found == values)


@triton.jit
def plan_delta(
    Table,
    Candidates,
    Ids,
    Ages,
    Sources,
    Destinations,
    Counts,
    Selected,
    HEADS: tl.constexpr,
    SLOTS: tl.constexpr,
    CAP: tl.constexpr,
    BACKGROUND: tl.constexpr,
    N: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    row = Table + b * 9
    pos = tl.load(row + 7)
    i = tl.arange(0, N)
    out = (b * HEADS + h) * SLOTS
    if pos >= 0:
        resident = tl.load(row + 4).to(tl.pointer_type(tl.int64))
        ages_ptr = tl.load(row + 5).to(tl.pointer_type(tl.int64))
        ids = tl.load(resident + h * SLOTS + i, i < SLOTS, -1)
        ages = tl.load(ages_ptr + h * SLOTS + i, i < SLOTS, -1)
        candidates = tl.load(Candidates + out + i, i < SLOTS, -1).to(tl.int64)
        sorted_ids = tl.sort(
            tl.where((i < SLOTS) & (ids >= 0), ids, 2147483647), descending=False
        )
        sorted_candidates = tl.sort(
            tl.where(candidates >= 0, candidates, 2147483647), descending=False
        )
        hit = _contains(sorted_candidates, ids, N, SEARCH_STEPS)
        missing = (candidates >= 0) & ~_contains(
            sorted_ids, candidates, N, SEARCH_STEPS
        )
        count = tl.minimum(tl.sum(missing.to(tl.int32), 0), CAP)
        # Encode the slot in the low bits for deterministic, stable LRU ties.
        priority = tl.where(ids < 0, -1, ages)
        priority = tl.where(hit | (i >= SLOTS), 2147483647, priority)
        targets = (tl.sort((priority + 1) * N + i, descending=False) % N).to(tl.int32)
        compact = (tl.sort(tl.where(missing, i, i + N), descending=False) % N).to(
            tl.int32
        )
        sources = tl.gather(candidates, compact, 0)
        admitted = i < count
        step = tl.load(row + 8) + BACKGROUND
        updated_ages = tl.where(hit, step, ages)
        new_ids = tl.where(admitted, sources, tl.gather(ids, targets, 0))
        new_ages = tl.where(admitted, step, tl.gather(updated_ages, targets, 0))
        tl.store(Ids + out + targets, new_ids, targets < SLOTS)
        tl.store(Ages + out + targets, new_ages, targets < SLOTS)
        tl.store(
            Selected + out + targets,
            admitted | tl.gather(hit, targets, 0),
            targets < SLOTS,
        )
        tl.store(Sources + out + i, sources, i < SLOTS)
        tl.store(Destinations + out + i, targets, i < SLOTS)
        tl.store(Counts + b * HEADS + h, count)
    else:
        tl.store(Counts + b * HEADS + h, 0)
        tl.store(Selected + out + i, False, i < SLOTS)


@triton.jit
def copy_misses(
    Table,
    Sources,
    Destinations,
    Counts,
    HEADS: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    STORAGE: tl.constexpr,
    DTYPE: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
):
    b, h, worker = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = Table + b * 9
    if tl.load(row + 7) >= 0:
        host = tl.load(row + 0).to(tl.pointer_type(DTYPE))
        data = tl.load(row + 1).to(tl.pointer_type(DTYPE))
        capacity = tl.load(row + 6)
        count = tl.load(Counts + b * HEADS + h)
        i = tl.arange(0, TILE)
        for miss in range(worker, count, WORKERS):
            src = tl.load(Sources + (b * HEADS + h) * SLOTS + miss)
            dst = tl.load(Destinations + (b * HEADS + h) * SLOTS + miss)
            for offset in range(triton.cdiv(BLOCK * STORAGE, TILE)):
                d = offset * TILE + i
                value = tl.load(
                    host + (h * capacity + src) * BLOCK * STORAGE + d,
                    d < BLOCK * STORAGE,
                    0,
                )
                tl.store(
                    data + (h * (SLOTS + 1) + dst) * BLOCK * STORAGE + d,
                    value,
                    d < BLOCK * STORAGE,
                )


@triton.jit
def publish_plan(
    Table, Ids, Ages, HEADS: tl.constexpr, SLOTS: tl.constexpr, N: tl.constexpr
):
    b, h = tl.program_id(0), tl.program_id(1)
    row = Table + b * 9
    if tl.load(row + 7) >= 0:
        resident = tl.load(row + 4).to(tl.pointer_type(tl.int64))
        ages = tl.load(row + 5).to(tl.pointer_type(tl.int64))
        i = tl.arange(0, N)
        ids = tl.load(Ids + (b * HEADS + h) * SLOTS + i, i < SLOTS, -1)
        times = tl.load(Ages + (b * HEADS + h) * SLOTS + i, i < SLOTS, -1)
        tl.store(resident + h * SLOTS + i, ids, i < SLOTS)
        tl.store(ages + h * SLOTS + i, times, i < SLOTS)


@triton.jit
def attention_partial(
    Table,
    Query,
    Key,
    Value,
    Selected,
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
    PER_SPLIT: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    TILE: tl.constexpr,
):
    bq, qh, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, lane = bq // WIDTH, bq % WIDTH
    h = qh // GROUP
    row = Table + b * 9
    pos = tl.load(row + 7)
    dk, dv = tl.arange(0, DK), tl.arange(0, DV)
    acc = tl.full((DV,), 0, tl.float32)
    m, z = tl.full((), -float("inf"), tl.float32), tl.full((), 0, tl.float32)
    if pos >= 0:
        dtype = Key.dtype.element_ty
        data = tl.load(row + 1).to(tl.pointer_type(dtype))
        resident = tl.load(row + 4).to(tl.pointer_type(tl.int64))
        q = tl.load(Query + (bq * HEADS * GROUP + qh) * KD + dk, dk < KD, 0).to(
            tl.float32
        )
        for tile in range(triton.cdiv(PER_SPLIT, TILE)):
            token = split * PER_SPLIT + tile * TILE + tl.arange(0, TILE)
            slot, within = token // BLOCK, token % BLOCK
            history = token < (SLOTS + 1) * BLOCK
            selected = tl.load(
                Selected + (b * HEADS + h) * SLOTS + slot, slot < SLOTS, False
            )
            block_id = tl.load(resident + h * SLOTS + slot, slot < SLOTS, -1)
            block_id = tl.where(slot == SLOTS, pos // BLOCK, block_id)
            valid = (
                history
                & (block_id >= 0)
                & (block_id * BLOCK + within <= pos)
                & (selected | (slot == SLOTS))
            )
            draft = (WIDTH == 2) & (lane == 1) & (token == (SLOTS + 1) * BLOCK)
            offset = (h * (SLOTS + 1) * BLOCK + token) * STORAGE
            k = tl.load(
                data + offset[:, None] + dk[None, :],
                valid[:, None] & (dk[None, :] < KD),
                0,
            ).to(tl.float32)
            v = tl.load(
                data + offset[:, None] + KD + dv[None, :],
                valid[:, None] & (dv[None, :] < VD),
                0,
            ).to(tl.float32)
            if WIDTH == 2:
                draft_k = tl.load(
                    Key + ((b * WIDTH + 1) * HEADS + h) * KD + dk, dk < KD, 0
                ).to(tl.float32)
                draft_v = tl.load(
                    Value + ((b * WIDTH + 1) * HEADS + h) * VD + dv, dv < VD, 0
                ).to(tl.float32)
                k = tl.where(draft[:, None], draft_k[None, :], k)
                v = tl.where(draft[:, None], draft_v[None, :], v)
            logits = tl.sum(k * q[None, :], 1) * SCALE
            if CAP > 0:
                logits = CAP * (2 / (1 + tl.exp(2 * (-logits / CAP))) - 1)
            logits = tl.where(valid | draft, logits, -float("inf"))
            new_m = tl.maximum(m, tl.max(logits, 0))
            safe_m = tl.where(new_m == -float("inf"), 0, new_m)
            alpha = tl.exp(m - safe_m)
            p = tl.exp(logits - safe_m)
            acc = acc * alpha + tl.sum(p[:, None] * v, 0)
            z = z * alpha + tl.sum(p, 0)
            m = new_m
    out = ((bq * HEADS * GROUP + qh) * SPLITS + split) * (VD + 2)
    tl.store(Partial + out + dv, acc, dv < VD)
    tl.store(Partial + out + VD, m)
    tl.store(Partial + out + VD + 1, z)


@triton.jit
def attention_reduce(
    Partial,
    Output,
    QHEADS: tl.constexpr,
    VD: tl.constexpr,
    SPLITS: tl.constexpr,
    NS: tl.constexpr,
    DV: tl.constexpr,
):
    bq, h = tl.program_id(0), tl.program_id(1)
    s, d = tl.arange(0, NS), tl.arange(0, DV)
    offset = ((bq * QHEADS + h) * SPLITS + s) * (VD + 2)
    m = tl.load(Partial + offset + VD, s < SPLITS, -float("inf"))
    z = tl.load(Partial + offset + VD + 1, s < SPLITS, 0)
    maximum = tl.max(m, 0)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    weights = tl.exp(m - maximum)
    acc = tl.load(
        Partial + offset[:, None] + d[None, :],
        (s[:, None] < SPLITS) & (d[None, :] < VD),
        0,
    )
    total = tl.sum(z * weights, 0)
    out = tl.sum(acc * weights[:, None], 0) / tl.where(total > 0, total, 1)
    tl.store(Output + (bq * QHEADS + h) * VD + d, out, d < VD)
