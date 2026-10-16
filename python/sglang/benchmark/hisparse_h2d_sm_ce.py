"""Activity coverage for mixed SM gather / H2D copy-engine diagnostics."""

from sglang.benchmark.hisparse_h2d_pair import launch_shape
from sglang.benchmark.hisparse_h2d_streams import duration, merge


def mixed_overlap(trace, banks, copies, copy_bytes, parallel):
    groups = {"SM": [], "CE": []}
    streams = {"SM": set(), "CE": set()}
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat", "").lower(), e.get("name", "").lower()
        if cat == "kernel" and "copy_cache_planned_kernel" in name:
            kind = "SM"
        elif "gpu_memcpy" in cat and "htod" in name:
            kind = "CE"
        else:
            continue
        a = e.get("args", {})
        if a.get("device") != 0 or "stream" not in a:
            raise ValueError("Missing/wrong activity device or stream")
        if kind == "SM" and (
            launch_shape(a.get("grid")) != (4, 1, 1)
            or launch_shape(a.get("block")) != (1024, 1, 1)
        ):
            raise ValueError("Wrong SM launch shape")
        if kind == "CE" and a.get("bytes") != copy_bytes:
            raise ValueError(f"Wrong/missing CE activity bytes: {a}")
        streams[kind].add(str(a["stream"]))
        groups[kind].append((e["ts"], e["ts"] + e["dur"]))
    if len(groups["SM"]) != banks or len(groups["CE"]) != copies:
        raise ValueError(
            f"Incomplete SM/CE coverage: { {k: len(v) for k, v in groups.items()} }"
        )
    if any(len(s) != 1 for s in streams.values()):
        raise ValueError("Expected one stream per workload")
    if (streams["SM"] != streams["CE"]) != parallel:
        raise ValueError("Wrong serial/parallel stream binding")
    unions = {k: merge(v) for k, v in groups.items()}
    active = {k: duration(v) for k, v in unions.items()}
    united = merge(groups["SM"] + groups["CE"])
    intersection = sum(active.values()) - duration(united)
    if not parallel and intersection > 1e-3:
        raise ValueError("Serial reference overlaps")
    return dict(
        valid=True,
        counts={k: len(v) for k, v in groups.items()},
        streams={k: sorted(v) for k, v in streams.items()},
        active_us=active,
        intersection_us=intersection,
        overlap_fraction={k: intersection / t for k, t in active.items()},
        union_us=duration(united),
    )


class CEReader:
    """Pinned contiguous H2D graph, with source and destination alive throughout."""

    def __init__(self, torch, source, copies):
        self.stream = torch.cuda.Stream(device=0)
        self.source = source
        self.banks = copies
        self.payload_bytes = copies * source.numel()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.storage = torch.full(
            (source.numel() + 256,), 165, dtype=torch.uint8, device="cuda:0"
        )
        self.output = self.storage[128:-128]

        def check(_bank):
            if not torch.equal(self.output.cpu(), source):
                raise ValueError("CE payload mismatch")
            if not bool((self.storage[:128] == 165).all()) or not bool(
                (self.storage[-128:] == 165).all()
            ):
                raise ValueError("CE guard mismatch")

        self.check = check
        self.output.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        check(0)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.start.record()
            for _ in range(copies):
                self.output.copy_(source, non_blocking=True)
            self.end.record()
        self.output.zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        torch.cuda.synchronize()
        check(0)
