"""Device-clock markers for the opt-in Lagom performance trace."""

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
