"""Byte-exact copy tests for mapped host payload and changing graph inputs.

CPU: TRITON_INTERPRET=1 PYTHONPATH=python SGLANG_USE_CPU_ENGINE=0 pytest -q <file>
CUDA: same command without TRITON_INTERPRET; also exercises pinned host UVA.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import triton.language as tl
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="CUDA or Triton interpreter required",
)
DEVICE = "cpu" if INTERPRET else "cuda"


@pytest.mark.parametrize(
    "block,storage,dtype,tile",
    [
        (16, 256, torch.float16, 1024),
        (16, 960, torch.float16, 1024),
        pytest.param(
            16,
            256,
            torch.bfloat16,
            1024,
            marks=pytest.mark.skipif(INTERPRET, reason="BF16 validated on CUDA"),
        ),
        (1, 1088, torch.float32, 512),
        (3, 24, torch.float16, 64),
    ],
)
@pytest.mark.parametrize("workers", [1, 4, 17])
@pytest.mark.parametrize("group_ctas", [2, 6])
def test_copy_payload_tails_zero_misses_padding_and_request_reuse(
    block, storage, dtype, tile, workers, group_ctas
):
    batch, heads, slots, capacity = 3, 2, 3, 7
    host = torch.empty(
        (batch, heads, capacity, block, storage),
        dtype=dtype,
        pin_memory=not INTERPRET,
    )
    host.random_(0, 1024)
    data = torch.full(
        (batch, heads, slots + 1, block, storage), -5, dtype=dtype, device=DEVICE
    )
    sources = torch.tensor(
        [[[6, 1, 3], [2, 5, 0]]] * batch, device=DEVICE, dtype=torch.int64
    )
    destinations = torch.tensor(
        [[[2, 0, 1], [0, 2, 1]]] * batch, device=DEVICE, dtype=torch.int64
    )
    counts = torch.tensor([[3, 1], [0, 2], [3, 3]], device=DEVICE, dtype=torch.int32)
    table = torch.zeros((batch, kernels.FIELDS), device=DEVICE, dtype=torch.int64)
    owners = []

    def bind(order):
        rows = []
        for source_request in order:
            row = [0] * kernels.FIELDS
            if source_request is None:
                row[kernels.PREFIX] = -1
            else:
                row[kernels.PREFIX] = 16
                row[kernels.HOST] = host[source_request].data_ptr()
                row[kernels.DATA] = data[source_request].data_ptr()
                row[kernels.CAPACITY] = capacity
            rows.append(row)
        table.copy_(torch.tensor(rows, device=DEVICE))

    def run(kernel):
        legacy = kernel is kernels.copy_misses_legacy
        grid = (batch, heads, workers) if legacy else (group_ctas, workers)
        kernel[grid](
            table,
            sources,
            destinations,
            counts,
            HEADS=heads,
            SLOTS=slots,
            BLOCK=block,
            STORAGE=storage,
            DTYPE={
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }[dtype],
            TILE=tile,
            WORKERS=workers,
            num_warps=4,
            **({} if legacy else {"GROUPS": batch * heads}),
        )

    bind([0, 1, None])
    run(kernels.copy_misses)
    graph = None
    if not INTERPRET:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(kernels.copy_misses)
        torch.cuda.synchronize()

    for step, order in enumerate(([0, 1, None], [1, None, 0], [2, 0, None])):
        if step:
            # Captured pointers stay fixed; update their contents, ownership,
            # counts and host payload. Keep old owners to detect stale writes.
            owners.append(host)
            host = torch.empty_like(host, pin_memory=not INTERPRET)
            host.random_(0, 1024)
            counts.copy_(torch.tensor([[1, 0], [2, 3], [3, 2]], device=DEVICE))
        bind(order)
        expected = torch.full_like(data, -5, device="cpu")
        for b, source_request in enumerate(order):
            if source_request is None:
                continue
            for h in range(heads):
                for miss in range(int(counts[b, h])):
                    expected[source_request, h, int(destinations[b, h, miss])] = host[
                        source_request, h, int(sources[b, h, miss])
                    ]
        data.fill_(-5)
        if graph is not None:
            graph.replay()
            torch.cuda.synchronize()
        else:
            run(kernels.copy_misses)
        assert torch.equal(data.cpu().view(torch.uint8), expected.view(torch.uint8))
        # The legacy kernel provides an independent reference for tiling.
        data.fill_(-5)
        run(kernels.copy_misses_legacy)
        if not INTERPRET:
            torch.cuda.synchronize()
        assert torch.equal(data.cpu().view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(
    INTERPRET, reason="Benchmark timing requires CUDA graph event nodes"
)
def test_copy_benchmark_cuda_graph_timing_and_concurrent_compute(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmark/oasiskv/bench_copy.py"
    spec = importlib.util.spec_from_file_location("oasiskv_copy_bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    output = tmp_path / "copy.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--batch",
            "2",
            "--live-batch",
            "1",
            "--heads",
            "2",
            "--storage",
            "256",
            "--top-k",
            "32",
            "--context-length",
            "64",
            "--misses-per-head",
            "0,1",
            "--samples",
            "2",
            "--warmup",
            "1",
            "--flush-mib",
            "1",
            "--matmul-size",
            "32",
            "--output",
            str(output),
        ],
    )
    bench.main()
    report = json.loads(output.read_text())
    assert len(report["results"]) == 6
    assert {row["total_misses"] for row in report["results"]} == {0, 2}
    for row in report["results"]:
        assert row["payload_bytes"] == row["total_misses"] * 8192
        assert row["copy"]["p50_ms"] > 0
        assert row["joint"]["p50_ms"] > 0
        assert row["compute_slowdown"] > 0
