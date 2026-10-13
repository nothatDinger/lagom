"""Read-only Linux NUMA evidence; VMA counts are never attributed to subranges."""

import math
import os
import statistics
from collections import Counter
from pathlib import Path

PAGE_TOKENS = 64
VALUE_BYTES = 576
SCALE_BYTES = 8
ITEM_BYTES = VALUE_BYTES + SCALE_BYTES
PAGE_BYTES = math.ceil(PAGE_TOKENS * ITEM_BYTES / VALUE_BYTES) * VALUE_BYTES


def parse_maps(text):
    result = []
    for line in text.splitlines():
        fields = line.split()
        lo, hi = fields[0].split("-")
        result.append(dict(start=int(lo, 16), end=int(hi, 16), permissions=fields[1]))
    return result


def parse_numa_maps(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        row = dict(policy=fields[1], pages_by_node={})
        for field in fields[2:]:
            key, sep, value = field.partition("=")
            if sep and key.startswith("N") and key[1:].isdigit():
                row["pages_by_node"][key[1:]] = int(value)
            elif sep and key in ("kernelpagesize_kB", "anon", "dirty", "mapped"):
                row[key] = int(value)
        result[int(fields[0], 16)] = row
    return result


def describe_buffer(buffer, maps, numa_maps):
    start = int(buffer["address"])
    end = start + int(buffer["nbytes"])
    regions = []
    covered = 0
    for vma in maps:
        lo, hi = max(start, vma["start"]), min(end, vma["end"])
        if lo >= hi:
            continue
        covered += hi - lo
        regions.append(
            dict(
                **vma,
                overlap_bytes=hi - lo,
                numa=numa_maps.get(vma["start"]),
                counts_scope="whole_vma",
                buffer_covers_whole_vma=(lo == vma["start"] and hi == vma["end"]),
            )
        )
    return dict(**buffer, mapped_bytes=covered, regions=regions)


def process_snapshot(pid=None, buffers=(), proc_root="/proc"):
    pid = os.getpid() if pid is None else int(pid)
    root = Path(proc_root) / str(pid)
    result = dict(pid=pid, errors=[])

    def read(relative):
        try:
            return (root / relative).read_text()
        except OSError as exc:
            result["errors"].append(dict(path=str(relative), error=str(exc)))
            return None

    def start_ticks():
        stat = read("stat")
        # comm may contain spaces and parentheses. Fields after ')' start at #3.
        return int(stat.rsplit(")", 1)[1].split()[19]) if stat else None

    before = start_ticks()
    status = read("status")
    allowed = {
        "Name",
        "Pid",
        "PPid",
        "Threads",
        "VmRSS",
        "VmLck",
        "Cpus_allowed_list",
        "Mems_allowed_list",
    }
    result["status"] = (
        {
            k: v.strip()
            for k, v in (line.split(":", 1) for line in status.splitlines())
            if k in allowed
        }
        if status
        else {}
    )
    maps_text, numa_text = read("maps"), read("numa_maps")
    maps = parse_maps(maps_text) if maps_text is not None else []
    numa = parse_numa_maps(numa_text) if numa_text is not None else {}
    # /proc page counts have the base-page unit (hugetlb VMAs are an exception).
    # Keep kernel page sizes and raw counts; do not silently convert to bytes.
    counts = Counter()
    for row in numa.values():
        counts.update(row["pages_by_node"])
    result["process_pages_by_node"] = dict(counts)
    result["vmas"] = [dict(**vma, numa=numa.get(vma["start"])) for vma in maps]
    result["buffers"] = [describe_buffer(b, maps, numa) for b in buffers]
    affinity = Counter()
    try:
        tasks = sorted((root / "task").iterdir())
        for task in tasks:
            text = read(task.relative_to(root) / "status")
            if text:
                fields = dict(line.split(":", 1) for line in text.splitlines())
                affinity[fields.get("Cpus_allowed_list", "").strip()] += 1
    except OSError as exc:
        result["errors"].append(dict(path="task", error=str(exc)))
    result["threads_by_cpu_affinity"] = dict(affinity)
    result["start_ticks"] = before
    result["pid_stable_during_snapshot"] = (
        before is not None and before == start_ticks()
    )
    return result


def copy_geometry(batch, misses_per_request, source_bytes, max_banks):
    if min(batch, misses_per_request, source_bytes, max_banks) <= 0:
        raise ValueError("Batch, misses, source size and banks must be positive")
    entries = batch * misses_per_request
    pages = source_bytes // PAGE_BYTES
    banks = min(max_banks, pages * PAGE_TOKENS // entries)
    if banks < 2:
        raise ValueError("Host source must hold at least two disjoint miss-plan banks")
    return dict(
        entries=entries,
        host_pages=pages,
        host_bytes=pages * PAGE_BYTES,
        host_tokens=pages * PAGE_TOKENS,
        banks=banks,
        payload_bytes=entries * ITEM_BYTES,
        rotating_payload_bytes=banks * entries * ITEM_BYTES,
        destination_pages=math.ceil(entries / PAGE_TOKENS),
    )


def timing_summary(samples_us, nbytes):
    if not samples_us or any(t <= 0 or not math.isfinite(t) for t in samples_us):
        raise ValueError("Timings must be finite, positive and nonempty")
    ordered = sorted(samples_us)
    mean = statistics.fmean(samples_us)
    return dict(
        samples=len(samples_us),
        mean_us=mean,
        median_us=statistics.median(samples_us),
        p95_us=ordered[math.ceil(0.95 * len(ordered)) - 1],
        GBps=nbytes / mean / 1000,
        bytes_per_copy=nbytes,
    )
