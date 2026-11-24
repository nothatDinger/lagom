"""Device timing and transfer counters for the opt-in Lagom trace."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["region"])
def stamp(buffer, slot, steps: tl.constexpr, region, edge: tl.constexpr):
    index = tl.load(slot)
    if (index >= 0) & (index < steps):
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            constraints="=l",
            args=[],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        tl.store(buffer + (region * steps + index) * 2 + edge, now)


@triton.jit(do_not_specialize=["refill_region", "stage_region"])
def count_verify_copies(
    buffer,
    slot,
    kinds,
    demand,
    counts,
    live,
    steps: tl.constexpr,
    refill_region,
    stage_region,
    capacity: tl.constexpr,
    block: tl.constexpr,
):
    step = tl.load(slot)
    bid = tl.program_id(0)
    if (step >= 0) & (step < steps) & (bid < tl.load(live)):
        column = tl.arange(0, block)
        valid = column < tl.load(counts + bid)
        offset = bid * capacity + column
        source = tl.load(kinds + offset, valid, other=0)
        needed = tl.load(demand + offset, valid, other=0)
        # The staging copy kernels use Source::{Host,Pool,Forecast} = {1,2,3}.
        h2d = tl.sum(((source == 1) & (needed != 0)).to(tl.int32), 0)
        d2d = tl.sum(((source == 2) | (source == 3)).to(tl.int32), 0)
        tl.atomic_add(buffer + refill_region * steps + step, h2d, sem="relaxed")
        tl.atomic_add(buffer + stage_region * steps + step, d2d, sem="relaxed")


@triton.jit(do_not_specialize=["region"])
def count_forecast_copies(
    buffer,
    slot,
    counts,
    reqs,
    live,
    steps: tl.constexpr,
    region,
    batch_size: tl.constexpr,
    block: tl.constexpr,
):
    step = tl.load(slot)
    if (step >= 0) & (step < steps):
        bid = tl.arange(0, block)
        valid = (bid < batch_size) & (bid < tl.load(live))
        req = tl.load(reqs + bid, valid, other=0)
        rows = tl.load(counts + req, valid, other=0)
        tl.atomic_add(
            buffer + region * steps + step, tl.sum(rows, 0), sem="relaxed"
        )


@triton.jit(do_not_specialize=["gather", "reread", "commit"])
def store_swap_in_counts(
    entries,
    cache_work,
    slot,
    counts,
    steps: tl.constexpr,
    gather,
    reread,
    commit,
):
    step = tl.load(slot)
    if (step >= 0) & (step < steps):
        tl.store(entries + gather * steps + step, tl.load(counts))
        tl.store(entries + reread * steps + step, tl.load(counts + 4))
        tl.store(entries + commit * steps + step, tl.load(counts + 2))
        tl.store(cache_work + reread * steps + step, tl.load(counts + 1))
        tl.store(cache_work + commit * steps + step, tl.load(counts + 3))
