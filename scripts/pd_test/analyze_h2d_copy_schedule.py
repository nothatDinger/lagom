"""CPU-only check of the planned-copy warp traversal; no timing prediction.

Compare the source kernel's request-major traversal with an independent
flattened-plan oracle. This checks scheduling arithmetic, not CUDA correctness
or how many memory requests hardware can keep in flight.
"""

import hashlib
import itertools
import random
from bisect import bisect_right
from pathlib import Path


def kernel_schedule(counts, warps, real=None):
    """Model copy_cache_planned_kernel, including empty and padded requests."""
    if real is None:
        real = len(counts)
    assert 0 <= real <= len(counts) and warps > 0
    assert all(count >= 0 for count in counts[:real])
    output = [[] for _ in range(warps)]
    for warp in range(warps):
        start = 0
        for base in range(0, real, 32):
            for row in range(base, min(base + 32, real)):
                count = counts[row]
                if count == 0:
                    continue
                # Equivalent to C++ remainder followed by m0 += total_warps
                # when negative. Python already returns a nonnegative value.
                first = (warp - start) % warps
                for miss in range(first, count, warps):
                    output[warp].append((row, miss, start + miss))
                start += count
    return output


def flat_oracle(counts, warps, real):
    """Assign flat item k to k % warps; locate its row by cumulative ends."""
    ends = list(itertools.accumulate(counts[:real]))
    total = ends[-1] if ends else 0
    output = [[] for _ in range(warps)]
    for index in range(total):
        row = bisect_right(ends, index)
        start = ends[row - 1] if row else 0
        output[index % warps].append((row, index - start, index))
    return output


def iterator_schedule(counts, warps, real, depth):
    """Model the candidate's persistent cursor and guarded one/two-item groups."""
    assert depth in (1, 2)
    output = [[] for _ in range(warps)]
    for warp in range(warps):
        flat, row, start, end = warp, -1, 0, 0
        while True:
            pending = []
            for _ in range(depth):
                if row >= real:
                    continue
                while flat >= end:
                    row += 1
                    if row >= real:
                        break
                    start, end = end, end + counts[row]
                if row < real:
                    pending.append((row, flat - start, flat))
                    flat += warps
            if not pending:
                break
            output[warp].extend(pending)
    return output


def verify(counts, warps, real=None):
    real = len(counts) if real is None else real
    actual = kernel_schedule(counts, warps, real)
    oracle = flat_oracle(counts, warps, real)
    assert actual == oracle, (counts, warps, real)
    for depth in (1, 2):
        assert iterator_schedule(counts, warps, real, depth) == oracle, (
            counts,
            warps,
            real,
            depth,
        )
    return actual


def summarize(name, counts, warps=128):
    schedule = verify(counts, warps)
    lengths = [len(items) for items in schedule]
    cross_row = sum(
        a[0] != b[0] for items in schedule for a, b in zip(items, items[1:])
    )
    transitions = sum(max(0, size - 1) for size in lengths)
    runs = [
        len(list(group))
        for items in schedule
        for _, group in itertools.groupby(items, key=lambda item: item[0])
    ]
    print(
        f"SCHEDULE {name} items={sum(lengths)} active_warps={sum(n > 0 for n in lengths)} "
        f"items_per_warp={min(lengths)}..{max(lengths)} max_inner_trip={max(runs, default=0)} "
        f"cross_request_transitions={cross_row}/{transitions}"
    )
    print(f"WARP0 {name} (request, miss, flat)={schedule[0][:6]}")


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "python/sglang/kernels/jit/csrc/hisparse.cuh"
    print(f"SOURCE hisparse.cuh sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    checks = 0
    # Exhaustive small shapes exercise empty rows and modulo ownership.
    for counts in itertools.product(range(4), repeat=5):
        for warps in (1, 2, 4, 8):
            verify(counts, warps)
            checks += 1
    # Real launch width, row-window boundaries, irregular counts, graph padding.
    rng = random.Random(0)
    for real in (0, 1, 2, 31, 32, 33, 63, 64, 65):
        for _ in range(12):
            counts = [
                rng.choice((0, 1, 2, 65, 100, 127, 128, 129, 1272)) for _ in range(real)
            ]
            # Negative padding makes an accidental traversal of inactive rows fail.
            verify(counts + [-999] * 3, 128, real)
            checks += 1
    print(
        f"CPU_TRAVERSAL_CHECK passed={checks} variants=legacy,D1,D2; no CUDA execution or bandwidth model"
    )
    summarize("B32-miss100", [100] * 32)
    summarize("B32-miss1272", [1272] * 32)
    summarize("B32-miss2", [2] * 32)


if __name__ == "__main__":
    main()
