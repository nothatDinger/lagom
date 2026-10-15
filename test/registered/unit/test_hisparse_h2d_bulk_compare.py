import copy
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang.benchmark.hisparse_h2d_bulk_compare import (
    BINDINGS,
    CONFIG,
    COPY_BYTES,
    METHODS,
    ORDER,
    REFERENCE_SHA256,
    compact_report,
    report_status,
)
from sglang.benchmark.hisparse_h2d_device_order import HOST_BYTES
from sglang.benchmark.hisparse_h2d_linear import LINEAR_LIBRARY, PAYLOAD
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture():
    return dict(
        success=True,
        config=copy.deepcopy(CONFIG),
        visible_devices="4",
        devices={0: dict(physical_index=4, uuid="GPU-4", pci_bus="00000000:98:00.0")},
        registration=dict(device=0, physical=4, flags=1, rc=0),
        host_snapshot=dict(
            errors=[],
            buffers=[
                dict(
                    nbytes=HOST_BYTES,
                    mapped_bytes=HOST_BYTES,
                    regions=[
                        dict(
                            overlap_bytes=HOST_BYTES,
                            buffer_covers_whole_vma=True,
                            numa=dict(
                                policy="bind:1",
                                pages_by_node={"1": 131072},
                                kernelpagesize_kB=4,
                            ),
                        )
                    ],
                )
            ],
        ),
        correctness=[
            dict(method=m, bank=b, all_bytes=True, guards=True)
            for m in METHODS
            for b in range(7)
        ],
        chunk_graphs=[
            dict(
                method=m,
                device=0,
                capture_device=0,
                correctness=True,
                nbytes=COPY_BYTES,
                banks=7,
                waves=12,
                copy_nodes=84,
                per_copy_timing_events=0,
                launches_per_chunk=1,
                num_blocks=4 if m == "sm64" else None,
                block_size=1024 if m == "sm64" else None,
            )
            for m in METHODS
        ],
        binding=dict.fromkeys(BINDINGS, True),
        jit_artifact=dict(
            variant="linear", path="/cache/" + LINEAR_LIBRARY, sha256=REFERENCE_SHA256
        ),
        windows=[
            dict(
                device=0,
                method=m,
                trial=t,
                sampler_enabled=False,
                bytes_per_graph=COPY_BYTES * 7,
                states=[],
                metrics=dict(
                    valid=True,
                    logical_wall_GBps=50 if m == "sm64" else 55,
                    logical_gpu_GBps=50.01 if m == "sm64" else 55.01,
                    max_enqueue_fraction=0.001,
                    chunk_gpu_cv=0.001,
                    gpu_wall_ratio=0.999,
                    reason=None,
                ),
            )
            for t, m in enumerate(ORDER)
        ],
    )


def test_equal_bulk_work_produces_unambiguous_time_and_rate_ratios():
    report = json.loads(json.dumps(fixture()))
    assert report_status(report)["ready"]
    text = compact_report(report)
    assert "results_valid=True" in text
    assert "correctness=14/14 graphs=2/2" in text
    assert "SM/CE_time=[1.1, 1.1] SM/CE_GBps=[0.909091, 0.909091]" in text


@pytest.mark.parametrize(
    "fault",
    [
        "device",
        "registration",
        "payload",
        "numa",
        "byte_check",
        "guard",
        "capture",
        "nodes",
        "events",
        "blocks",
        "target",
        "source",
        "chunk",
        "artifact",
        "missing",
        "duplicate",
        "order",
        "quality",
        "drift",
        "slow_ce",
        "failure",
    ],
)
def test_invalid_baselines_do_not_produce_performance_conclusions(fault):
    r = fixture()
    if fault == "device":
        r["devices"][0]["physical_index"] = 5
    elif fault == "registration":
        r["registration"]["physical"] = 5
    elif fault == "payload":
        r["chunk_graphs"][0]["nbytes"] = PAYLOAD
    elif fault == "numa":
        r["host_snapshot"]["buffers"][0]["regions"][0]["numa"]["pages_by_node"] = {
            "0": 131072
        }
    elif fault == "byte_check":
        r["correctness"].pop()
    elif fault == "guard":
        r["correctness"][0]["guards"] = False
    elif fault == "capture":
        r["chunk_graphs"][0]["capture_device"] = 1
    elif fault == "nodes":
        r["chunk_graphs"][0]["copy_nodes"] += 1
    elif fault == "events":
        r["chunk_graphs"][0]["per_copy_timing_events"] = 84
    elif fault == "blocks":
        r["chunk_graphs"][0]["num_blocks"] = 8
    elif fault in ("target", "source", "chunk"):
        r["binding"][
            {
                "target": "same_target",
                "source": "same_sources",
                "chunk": "same_chunk_work",
            }[fault]
        ] = False
    elif fault == "artifact":
        r["jit_artifact"]["sha256"] = "a" * 64
    elif fault == "missing":
        r["windows"].pop()
    elif fault == "duplicate":
        r["windows"][-1] = copy.deepcopy(r["windows"][0])
    elif fault == "order":
        r["windows"][0], r["windows"][1] = r["windows"][1], r["windows"][0]
    elif fault == "quality":
        r["windows"][0]["metrics"]["valid"] = False
    elif fault == "drift":
        r["windows"][0]["metrics"]["logical_wall_GBps"] *= 1.2
    elif fault == "slow_ce":
        for row in r["windows"]:
            if row["method"] == "ce64":
                row["metrics"]["logical_wall_GBps"] = 30
    else:
        r["success"] = False
    assert not report_status(r)["ready"]
    assert "RATIOS " not in compact_report(r)


@pytest.mark.parametrize("size", [None, 264])
def test_parameterized_case_preserves_defaults_guards_and_shared_output(
    monkeypatch, size
):
    # Exercise construction/byte checking on CPU; this is not a CUDA test.
    import torch

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts/pd_test"))
    import bench_h2d_linear as script

    fake = SimpleNamespace(
        uint8=torch.uint8,
        cuda=SimpleNamespace(device=lambda d: nullcontext(), synchronize=lambda: None),
        empty=lambda n, dtype, device: torch.empty(n, dtype=dtype),
    )

    def cpu_copy(source, output, blocks):
        assert blocks == 4
        output.copy_(source)

    monkeypatch.setattr(script, "copy_linear", cpu_copy)
    nbytes = PAYLOAD if size is None else size
    inputs = [torch.full((nbytes,), v, dtype=torch.uint8) for v in (17, 29)]
    kwargs = {} if size is None else dict(nbytes=size)
    sm = script.ContiguousCase(fake, 0, "linear4", inputs, **kwargs)
    ce = script.ContiguousCase(fake, 0, "ce_same", inputs, shared=sm, **kwargs)
    assert sm.nbytes == ce.nbytes == nbytes
    assert sm.output.data_ptr() == ce.output.data_ptr()
    assert sm.rx_output.numel() == nbytes + 256
    for case in (sm, ce):
        case.rx_reset()
        case.functions[0]()
        case.rx_validate(0)
        case.rx_output[0] = 0
        with pytest.raises(ValueError, match="guard"):
            case.rx_validate(0)
    with pytest.raises(ValueError, match="Shared output"):
        script.ContiguousCase(fake, 1, "ce_same", inputs, shared=sm, **kwargs)
    with pytest.raises(ValueError, match="input sizes"):
        script.ContiguousCase(fake, 0, "linear4", [inputs[0][:-1]], **kwargs)


def test_cli_preserves_report_on_unavailable_cuda(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root / "python"))
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/pd_test/bench_h2d_bulk_compare.py"),
            "--output-dir",
            str(tmp_path / "run"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    report = json.loads((tmp_path / "run/bulk.json").read_text())
    assert report["error"] == "RuntimeError: Expose only physical GPU4 as CUDA0"
    assert "results_valid=False" in (tmp_path / "run/bulk-compact.txt").read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
