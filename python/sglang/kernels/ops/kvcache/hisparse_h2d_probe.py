"""One opt-in GPU record after a HiSparse copy, including on graph replay."""

import triton
import triton.language as tl


@triton.jit
def hisparse_h2d_record_kernel(
    counts,
    num_real_reqs,
    control,
    records,
    CAPACITY: tl.constexpr,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    LAYER: tl.constexpr,
    WIDTH: tl.constexpr,
    ITEM_BYTES: tl.constexpr,
    COPY_BLOCKS: tl.constexpr,
    SKIP_IO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(control) != 0:
        index = tl.atomic_add(control + 1, 1)
        if index < CAPACITY:
            real = tl.minimum(tl.load(num_real_reqs), BATCH)
            offsets = tl.arange(0, BLOCK)
            misses = tl.load(counts + offsets, offsets < real, other=0)
            row = records + index * STRIDE
            tl.store(row + 0, index)
            tl.store(row + 1, tl.sum(misses.to(tl.int64)))
            tl.store(row + 2, real)
            tl.store(row + 3, BATCH)
            tl.store(row + 4, LAYER)
            tl.store(row + 5, WIDTH)
            tl.store(row + 6, ITEM_BYTES)
            tl.store(row + 7, COPY_BLOCKS)
            tl.store(row + 8, SKIP_IO)
            # Used for ordering checks only; bandwidth comes from GPU trace time.
            stamp = tl.inline_asm_elementwise(
                "mov.u64 $0, %globaltimer;",
                constraints="=l",
                args=[],
                dtype=tl.int64,
                is_pure=False,
                pack=1,
            )
            tl.store(row + 9, stamp)
            tl.store(row + 10 + offsets, misses, offsets < BATCH)
