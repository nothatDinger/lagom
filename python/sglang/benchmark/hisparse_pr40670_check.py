"""CPU-side provenance/configuration validation for the pinned PR reference."""

import hashlib
import json
from pathlib import Path

from sglang.benchmark.hisparse_h2d_pair import launch_shape

COMMIT = "70bc9e9a643faa998ce529a38260feee83d3c0c6"
ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "python/sglang/kernels/jit/csrc/pr40670_reference"


def source_manifest(directory=REFERENCE):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["commit"] != COMMIT or len(manifest["files"]) != 3:
        raise ValueError("Wrong pinned reference")
    for row in manifest["files"]:
        text = (directory / Path(row["vendored"]).name).read_text()
        normalized = text.rstrip() + "\n"
        if hashlib.sha256(normalized.encode()).hexdigest() != row["vendored_sha256"]:
            raise ValueError("Reference source changed: " + row["vendored"])
        if row["adaptation"] != "none":
            if Path(row["vendored"]).name != "hisparse_transfer.cuh":
                raise ValueError("Unknown source adaptation")
            normalized = normalized.replace(
                '#include "kvcacheio.cuh"',
                "#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>",
            )
        if hashlib.sha256(normalized.encode()).hexdigest() != row["source_sha256"]:
            raise ValueError("Unrecorded algorithm change")
    return manifest


def launch_config(batch, width, top_k):
    if min(batch, width, top_k) <= 0 or width * top_k > 8192:
        raise ValueError("Invalid reference geometry")
    return dict(
        gather=dict(grid=[batch, (width * top_k + 63) // 64, 1], block=[64, 1, 1]),
        commit=dict(grid=[batch, 1, 1], block=[512, 1, 1]),
    )


def validate_profile(trace, batch, width, top_k):
    expected = launch_config(batch, width, top_k)
    kernels = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]
    if len(kernels) != 2:
        raise ValueError(f"Expected exactly two reference kernels, got {len(kernels)}")
    results = []
    for kind in ("gather", "commit"):
        matches = [
            e
            for e in kernels
            if f"load_cache_to_device_buffer_spec_{kind}_kernel" in e.get("name", "")
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate {kind} activity")
        e = matches[0]
        args = e.get("args", {})
        if (
            list(launch_shape(args.get("grid")) or ()) != expected[kind]["grid"]
            or list(launch_shape(args.get("block")) or ()) != expected[kind]["block"]
        ):
            raise ValueError(f"{kind} grid/block mismatch: {args}")
        if args.get("device") != 0 or e.get("dur", 0) <= 0:
            raise ValueError("Wrong device/nonpositive activity")
        results.append(
            dict(
                kind=kind,
                grid=expected[kind]["grid"],
                block=expected[kind]["block"],
                name=e["name"],
                device=args["device"],
                stream=args.get("stream"),
            )
        )
    return dict(valid=True, kernels=results, independent_copy_kernel=False)
