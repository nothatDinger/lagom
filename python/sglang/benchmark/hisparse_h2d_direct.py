"""Adjacent C4 direct-index diagnostic. All CUDA entry points are opt-in."""

import functools
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_diagnostics import PAGE_BYTES

DIRECT_LIBRARY = "sgl_kernel_jit_hisparse_c4_direct_probe_1024.so"
DIRECT_METHODS = ("gather_adjacent4", "direct4", "linear4")
EDGE_COUNTS = (0, 1, 65, 3200)
DIRECT_BINDING_KEYS = {
    "direct_same_source_output",
    "direct_same_indices",
    "direct_same_work",
}


@functools.cache
def _module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "hisparse_c4_direct_probe",
        1024,
        cuda_files=["hisparse_c4_direct_probe.cuh"],
        cuda_wrappers=[("hisparse_c4_direct", "hisparse_c4_direct")],
    )


def copy_direct(source, output, source_start, destination_start, count):
    import torch

    if torch.version.hip is not None:
        raise ValueError("Direct C4 diagnostic requires NVIDIA CUDA")
    if any(
        t.dtype != torch.uint8
        or t.ndim != 2
        or t.shape[1] != PAGE_BYTES
        or not t.is_contiguous()
        for t in (source, output)
    ):
        raise ValueError("Require contiguous uint8 C4 pages")
    if output.device.type != "cuda" or (source.data_ptr() | output.data_ptr()) & 7:
        raise ValueError("Require aligned allocations and a CUDA output")
    if source.device.type == "cpu":
        if not source.is_pinned():
            raise ValueError("Host source must be registered/pinned")
    elif source.device != output.device:
        raise ValueError("Source must be CPU memory or on the output GPU")
    if (
        any(
            type(x) is not int or x < 0
            for x in (source_start, destination_start, count)
        )
        or count > 2**31 - 1 - 128
        or source_start + count > min(source.shape[0] * 64, 2**31 - 1)
        or destination_start + count > min(output.shape[0] * 64, 2**31 - 1)
    ):
        raise ValueError("Direct C4 indices leave the allocations/int32 range")
    _module().hisparse_c4_direct(source, output, source_start, destination_start, count)


def schedule_check(batch=32, misses=100):
    """Compare ORIGINAL request-loop order against direct flat order per warp."""
    for warp in range(128):
        legacy, start = [], 0
        for _ in range(batch):
            m0 = (warp - start) % 128
            legacy.extend(start + m for m in range(m0, misses, 128))
            start += misses
        if legacy != list(range(warp, batch * misses, 128)):
            raise ValueError(f"Direct traversal differs at warp {warp}")
    return dict(valid=True, warps=128, items=batch * misses, ordered_items_match=True)


def check_edge(torch, host, device, count, source_kind, validate):
    with torch.cuda.device(device):
        cpu_source = host[:52]
        oracle = cpu_source.clone()
        source = cpu_source if source_kind == "host" else cpu_source.to("cuda")
        output = torch.zeros((53, PAGE_BYTES), dtype=torch.uint8, device="cuda")
        src, dst = torch.arange(63, 63 + count), torch.arange(9, 9 + count)

        def check():
            validate(torch, oracle, output, src, dst, check_untouched=True)
            if not torch.equal(source.cpu(), oracle):
                raise ValueError("Direct C4 changed source bytes")

        copy_direct(source, output, 63, 9, count)
        torch.cuda.synchronize()
        check()
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if (
                torch.cuda.current_device() != device
                or torch.cuda.current_stream().device.index != device
            ):
                raise RuntimeError("Direct edge capture device mismatch")
            copy_direct(source, output, 63, 9, count)
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check()
    return dict(
        device=device,
        count=count,
        source=source_kind,
        eager=True,
        graph=True,
        all_bytes=True,
        capture_device=device,
    )


class DirectCase:
    def __init__(self, torch, adjacent, adjacent_plan):
        self.device, self.method, self.probe_method = adjacent.device, "sm", "direct4"
        self.nbytes, self.banks, self.num_blocks = adjacent.nbytes, adjacent.banks, 4
        self.source, self.rx_output = adjacent.source, adjacent.rx_output
        # Reuse the ORIGINAL-source oracle and full untouched-byte checks.
        self.rx_validate = adjacent.rx_validate
        self.starts = adjacent_plan["active_src"].reshape(self.banks, -1)[:, 0].tolist()
        self.count, self.destination_start = 3200, 0
        self.functions = [
            lambda bank=b: copy_direct(
                self.source, self.rx_output, self.starts[bank], 0, self.count
            )
            for b in range(self.banks)
        ]
        with torch.cuda.device(self.device):
            for bank in (0, self.banks - 1):
                self.rx_output.zero_()
                self.functions[bank]()
                torch.cuda.synchronize()
                self.rx_validate(bank)


def binding_checks(torch, adjacent, direct, plan):
    active = plan["active_src"].reshape(64, -1)
    expected = torch.tensor(direct.starts)[:, None] + torch.arange(3200)
    return dict(
        direct_same_source_output=direct.source.data_ptr() == adjacent.source.data_ptr()
        and direct.rx_output.data_ptr() == adjacent.rx_output.data_ptr(),
        direct_same_indices=torch.equal(active, expected)
        and plan["dst"][:, :100].reshape(-1).tolist() == list(range(3200))
        and schedule_check()["valid"],
        direct_same_work=(
            direct.nbytes,
            direct.banks,
            direct.count,
            direct.destination_start,
            direct.num_blocks,
        )
        == (1868800, 64, 3200, 0, 4),
    )


def loaded_direct_artifact(maps_text=None):
    text = Path("/proc/self/maps").read_text() if maps_text is None else maps_text
    paths = {
        f[5]
        for line in text.splitlines()
        if len(f := line.split(None, 5)) == 6
        and Path(f[5].removesuffix(" (deleted)")).name == DIRECT_LIBRARY
    }
    if len(paths) != 1:
        raise ValueError(f"Expected one direct JIT artifact: {sorted(paths)}")
    path = next(iter(paths))
    return dict(variant="direct", path=path, sha256=sha256(path))


def codegen_shape(record):
    """Require the same scalar legacy read shape; counts are NOT TLP counts."""
    kernels = record.get("functions", [])
    return (
        bool(kernels)
        and all(
            k["opcodes"].get("LDG.E.64.CONSTANT") == 3
            and k["opcodes"].get("STG.E.64") == 3
            and sum(
                v
                for op, v in k["opcodes"].items()
                if op.startswith("LDG") and ".64" in op
            )
            == 3
            and not any(
                ".128" in op or op.startswith(("LDL", "STL")) for op in k["opcodes"]
            )
            for k in kernels
        )
        and bool(record.get("resource_lines"))
        and all(
            "STACK:0" in line and "LOCAL:0" in line for line in record["resource_lines"]
        )
    )


def evidence_valid(evidence, artifacts):
    rows = evidence.get("kernels", [])
    return (
        evidence.get("success") is True
        and len(rows) == 2
        and {r["variant"] for r in rows} == {"legacy", "direct"}
        and all(codegen_shape(r) for r in rows)
        and {r["variant"]: r["sha256"] for r in rows}
        == {
            a["variant"]: a["sha256"]
            for a in artifacts
            if a["variant"] in ("legacy", "direct")
        }
    )
