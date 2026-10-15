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
    RX_CONFIG,
    RX_ENABLED,
    RX_ORDER,
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


@pytest.mark.parametrize("rx", [False, True])
def test_cli_preserves_report_on_unavailable_cuda(tmp_path, rx):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root / "python"))
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/pd_test/bench_h2d_bulk_compare.py"),
            "--output-dir",
            str(tmp_path / "run"),
            *(["--rx-sampler"] if rx else []),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    report = json.loads((tmp_path / "run/bulk.json").read_text())
    assert report["error"] == "RuntimeError: Expose only physical GPU4 as CUDA0"
    assert "results_valid=False" in (tmp_path / "run/bulk-compact.txt").read_text()


def rx_fixture():
    r = fixture()
    templates = {w["method"]: w for w in r["windows"]}
    r["config"] = copy.deepcopy(RX_CONFIG)
    r["windows"] = []
    for trial, (method, enabled) in enumerate(zip(RX_ORDER, RX_ENABLED)):
        w = copy.deepcopy(templates[method])
        w.update(trial=trial, sampler_enabled=enabled, samples=[])
        if enabled:
            logical = w["metrics"]["logical_wall_GBps"]
            nominal = logical * 1.1
            w["metrics"].update(
                rx_GBps_1024=nominal,
                idle_over_active=0.0,
                idle=dict(valid=True, rx_kbps=0.0),
                rates=dict(
                    valid=True,
                    rx_kbps=nominal * 1e9 / 1024,
                    samples=38,
                    coverage=1.0,
                    rx_cv=0.005,
                ),
            )
            w["samples"] = [
                dict(t_ns=i * 50_000_000, call_ns=20_000_000) for i in range(40)
            ]
        r["windows"].append(w)
    return r


def test_rx_same_interval_ratios_and_sampler_controls():
    r = json.loads(json.dumps(rx_fixture()))
    s = report_status(r)
    assert s["ready"] and s["rx_samples"] and s["ce_calibration"]
    assert s["controls"]["sm64"]["on_over_off"] == [1.0, 1.0]
    text = compact_report(r)
    assert "windows=8/8" in text
    assert "SM/CE_nominal_Rx=[0.909091, 0.909091]" in text
    assert "relative_Rx_per_logical=[1.0, 1.0]" in text
    assert "query_ms_min_med_max=[20.0, 20.0, 20.0] period_ms_median=50.0" in text
    # Keep throughput unchanged while increasing received traffic per payload.
    # SM Rx/logical is not forced through the CE calibration bounds.
    for w in r["windows"]:
        if w["method"] == "sm64" and w["sampler_enabled"]:
            w["metrics"]["rx_GBps_1024"] *= 1.1
            w["metrics"]["rates"]["rx_kbps"] *= 1.1
    assert report_status(r)["ready"]
    assert "SM/CE_nominal_Rx=[1.0, 1.0]" in compact_report(r)
    assert "relative_Rx_per_logical=[1.1, 1.1]" in compact_report(r)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "order",
        "off_label",
        "sampler_effect",
        "drift",
        "ce_mismatch",
        "sm_no_rx",
        "sm_zero_rx",
        "units",
        "rx_quality",
        "coverage",
        "few_samples",
        "background",
        "window_quality",
        "config",
    ],
)
def test_rx_invalid_controls_suppress_ratios(fault):
    r = rx_fixture()
    sm = r["windows"][3]
    if fault == "missing":
        r["windows"].pop()
    elif fault == "order":
        r["windows"][2], r["windows"][3] = r["windows"][3], r["windows"][2]
    elif fault == "off_label":
        r["windows"][0]["sampler_enabled"] = True
    elif fault == "sampler_effect":
        sm["metrics"]["logical_wall_GBps"] *= 0.95
    elif fault == "drift":
        r["windows"][-1]["metrics"]["logical_wall_GBps"] *= 0.95
    elif fault == "ce_mismatch":
        for w in r["windows"]:
            if w["method"] == "ce64" and w["sampler_enabled"]:
                w["metrics"]["rx_GBps_1024"] *= 1.1
                w["metrics"]["rates"]["rx_kbps"] *= 1.1
    elif fault == "sm_no_rx":
        sm["metrics"].pop("rates")
    elif fault == "sm_zero_rx":
        sm["metrics"]["rx_GBps_1024"] = 0
        sm["metrics"]["rates"]["rx_kbps"] = 0
    elif fault == "units":
        sm["metrics"]["rx_GBps_1024"] *= 1000 / 1024
    elif fault == "rx_quality":
        sm["metrics"]["rates"]["rx_cv"] = 0.2
    elif fault == "coverage":
        sm["metrics"]["rates"]["coverage"] = 0.5
    elif fault == "few_samples":
        sm["metrics"]["rates"]["samples"] = 4
    elif fault == "background":
        sm["metrics"]["idle"]["rx_kbps"] = sm["metrics"]["rates"]["rx_kbps"] * 0.2
    elif fault == "window_quality":
        sm["metrics"]["valid"] = False
    else:
        r["config"]["sample_ms"] = 20
    assert not report_status(r)["ready"]
    assert "RATIOS" not in compact_report(r)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
