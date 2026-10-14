"""Correctness fixtures for cross-request copy staging; no timing code."""

import math
from types import SimpleNamespace

VARIANTS = ("legacy", "iterator_d1", "iterator_d2")


def make_iterator_case(torch, device):
    rng = torch.Generator().manual_seed(12013)
    capacities = [129] * 65
    for row in (0, 31, 32, 64):
        capacities[row] = 1272
    entries = sum(capacities)
    pages = math.ceil(entries / 64) + 2
    host = torch.empty(
        (pages, 37440),
        dtype=torch.uint8,
        pin_memory=torch.device(device).type == "cuda",
    )
    host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8, generator=rng))
    src_cpu = torch.full((66, 1280), -1, dtype=torch.int64)[:, :1272]
    dst_cpu = torch.full((66, 1280), -1, dtype=torch.int32)[:, :1272]
    # Force C4 boundary tokens into the valid plan, then use unique random slots.
    tail = torch.randperm(pages * 64, generator=rng)
    sources = torch.cat((torch.tensor([63, 64]), tail[(tail != 63) & (tail != 64)]))
    tail = torch.randperm(pages * 64, generator=rng)
    destinations = torch.cat(
        (torch.tensor([64, 63]), tail[(tail != 63) & (tail != 64)])
    )
    offset = 0
    for row, count in enumerate(capacities):
        src_cpu[row, :count] = sources[offset : offset + count]
        dst_cpu[row, :count] = destinations[offset : offset + count].to(torch.int32)
        offset += count
    src = torch.full((66, 1280), -1, dtype=torch.int64, device=device)[:, :1272]
    dst = torch.full((66, 1280), -1, dtype=torch.int32, device=device)[:, :1272]
    src.copy_(src_cpu)
    dst.copy_(dst_cpu)
    return SimpleNamespace(
        host=host,
        output=torch.empty_like(host, device=device, pin_memory=False),
        src_cpu=src_cpu,
        dst_cpu=dst_cpu,
        src=src,
        dst=dst,
        counts=torch.tensor(capacities + [999], dtype=torch.int32, device=device),
        real=torch.tensor([65], dtype=torch.int32, device=device),
        capacities=capacities,
    )


def graph_states(case):
    mixed = case.capacities + [999]  # Last request is always padding with invalid IDs.
    boundary = ([0, 1, 65, 100, 127, 128, 129] * 10)[:65] + [999]
    return [
        ("real0", mixed, 0),
        ("zero_misses", [0] * 65 + [999], 65),
        ("B32_miss100", [100] * 32 + mixed[32:], 32),
        ("boundary33", boundary, 33),
        ("mixed65", mixed, 65),
        ("changed_plan", [129] + mixed[1:], 1),
    ]


def iterator_expected(torch, case, counts, real):
    """Independent byte oracle: C4 page/value/scale positions, all padding kept."""
    result = torch.full(case.output.shape, 165, dtype=torch.uint8)
    sources = (
        torch.cat([case.src_cpu[r, : counts[r]] for r in range(real)])
        if real
        else torch.empty(0, dtype=torch.int64)
    )
    destinations = (
        torch.cat([case.dst_cpu[r, : counts[r]] for r in range(real)])
        if real
        else torch.empty(0, dtype=torch.int32)
    )
    src, dst = case.host.view(-1), result.view(-1)
    assert destinations.unique().numel() == destinations.numel()
    if sources.numel():
        assert int(sources.min()) >= 0 and int(sources.max()) < case.host.shape[0] * 64
        assert (
            int(destinations.min()) >= 0
            and int(destinations.max()) < result.shape[0] * 64
        )
    for begin in range(0, sources.numel(), 512):
        s = sources[begin : begin + 512].to(torch.int64)
        d = destinations[begin : begin + 512].to(torch.int64)
        for base, width in ((0, 576), (36864, 8)):
            source_bytes = (s // 64 * 37440 + base + s % 64 * width)[
                :, None
            ] + torch.arange(width)
            destination_bytes = (d // 64 * 37440 + base + d % 64 * width)[
                :, None
            ] + torch.arange(width)
            dst[destination_bytes] = src[source_bytes]
    return result


def check_iterator_bytes(torch, case, counts, real, variant, phase):
    expected = iterator_expected(torch, case, counts, real)
    actual = case.output.cpu()
    if not torch.equal(actual, expected):
        mismatches = (actual != expected).nonzero()
        first = [
            dict(page=p, byte=b, actual=int(actual[p, b]), expected=int(expected[p, b]))
            for p, b in mismatches[:6].tolist()
        ]
        raise ValueError(
            f"{variant} {phase}: real={real} mismatch_bytes={len(mismatches)} first={first}"
        )


def run_iterator_case(torch, case, variant, source_kind):
    from sglang.kernels.ops.kvcache.hisparse import (
        copy_cache_planned_mla,
        copy_cache_planned_mla_iterator,
    )

    if variant not in VARIANTS or source_kind not in ("host", "hbm"):
        raise ValueError("Unknown iterator correctness variant/source")
    device = case.output.device
    source = case.host if source_kind == "host" else case.host.to(device)
    common = dict(
        miss_src=case.src,
        miss_dst=case.dst,
        miss_count=case.counts,
        num_real_reqs=case.real,
        host_cache=source,
        device_buffer=case.output,
    )

    def run():
        if variant == "legacy":
            copy_cache_planned_mla(
                **common, item_size_bytes=584, is_dsv4_layout=True, load_policy="legacy"
            )
        else:
            copy_cache_planned_mla_iterator(**common, depth=int(variant[-1]))

    counts = case.capacities + [999]
    case.counts.copy_(torch.tensor(counts, dtype=torch.int32))
    case.real.fill_(65)
    case.output.fill_(165)
    run()  # JIT/eager outside capture.
    torch.cuda.synchronize(device)
    check_iterator_bytes(torch, case, counts, 65, variant, "eager")
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    phases = []
    for phase, counts, real in graph_states(case):
        case.host.bitwise_xor_(255)
        if phase == "changed_plan":
            case.src_cpu[0, :2] = case.src_cpu[0, :2].flip(0)
            case.src.copy_(case.src_cpu)
        if source_kind == "hbm":
            source.copy_(case.host)
        case.counts.copy_(torch.tensor(counts, dtype=torch.int32))
        case.real.fill_(real)
        case.output.fill_(165)
        graph.replay()
        torch.cuda.synchronize(device)
        check_iterator_bytes(torch, case, counts, real, variant, phase)
        phases.append(phase)
    return dict(
        variant=variant,
        source=source_kind,
        device=device.index,
        eager=True,
        graph_phases=phases,
        all_bytes_and_padding=True,
        capture_stream_device=stream.device.index,
        source_is_pinned=case.host.is_pinned(),
        stride=list(case.src.stride()),
    )
