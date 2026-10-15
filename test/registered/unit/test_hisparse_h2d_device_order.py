import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.benchmark.hisparse_h2d_device_order import (
    BINDINGS,
    HOST_BYTES,
    METHODS,
    compact_report,
    compare_reports,
    configuration,
    pair_compact,
    report_status,
    sequence,
)
from sglang.benchmark.hisparse_h2d_linear import EDGE_BYTES, LINEAR_LIBRARY, PAYLOAD
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture(order=(4, 5), *, fast_logical=False):
    config = configuration(order)
    r = dict(
        success=True,
        config=config,
        pid=400 if order[0] == 4 else 500,
        actual_visible_devices=config["visible_devices"],
        cuda_was_initialized=False,
        allocation_order=[0, 1],
        registration=dict(device=config["register_device"], physical=5, flags=1, rc=0),
        devices={
            d: dict(
                physical_index=p,
                uuid=f"GPU-{p}",
                pci_bus="00000000:" + ("98:00.0" if p == 4 else "B8:00.0"),
            )
            for d, p in enumerate(order)
        },
        runtime=dict(python=sys.executable, torch="test", cuda="13", device_order=None),
        code=dict(sha256={"probe.py": "b" * 64}),
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
        edge_checks=[
            dict(
                device=d,
                blocks=4,
                size=n,
                source=s,
                eager=True,
                graph=True,
                guards=True,
            )
            for d in (0, 1)
            for n in EDGE_BYTES
            for s in ("host", "hbm")
        ],
        chunk_graphs=[
            dict(
                device=d,
                method=m,
                capture_device=d,
                correctness=True,
                per_copy_timing_events=0,
                launches_per_chunk=1,
                nbytes=PAYLOAD if m == "linear4" else config["bulk_bytes"],
                banks=64 if m == "linear4" else 7,
                num_blocks=4 if m == "linear4" else None,
                block_size=1024 if m == "linear4" else None,
                waves=(
                    config["linear_waves"] if m == "linear4" else config["bulk_waves"]
                ),
                copy_nodes=(
                    64 * config["linear_waves"]
                    if m == "linear4"
                    else 7 * config["bulk_waves"]
                ),
            )
            for d in (0, 1)
            for m in METHODS
        ],
        bindings=[dict(device=d, checks=dict.fromkeys(BINDINGS, True)) for d in (0, 1)],
        jit_artifact=dict(
            variant="linear", path="/cache/" + LINEAR_LIBRARY, sha256="a" * 64
        ),
        windows=[],
    )
    for t, d, m, position in sequence():
        fast = d == 0 if fast_logical else order[d] == 4
        rate = (46 if fast else 33) if m == "linear4" else 55
        r["windows"].append(
            dict(
                trial=t,
                device=d,
                method=m,
                position=position,
                sampler_enabled=False,
                states=[],
                metrics=dict(
                    valid=True,
                    logical_wall_GBps=rate,
                    max_enqueue_fraction=0.001,
                    chunk_gpu_cv=0.001,
                    gpu_wall_ratio=0.999,
                    reason=None,
                ),
            )
        )
    return r


@pytest.mark.parametrize("logical_fast", [False, True])
def test_physical_identity_controls_comparison_after_ordinal_reversal(logical_fast):
    normal = fixture(fast_logical=logical_fast)
    reverse = json.loads(json.dumps(fixture((5, 4), fast_logical=logical_fast)))
    assert report_status(normal)["ready"] and report_status(reverse)["ready"]
    result = compare_reports(normal, reverse)
    assert result["results_valid"]
    a, b = result["comparisons"]
    assert a["GPU5_over_GPU4_time"]["linear4"] == pytest.approx([46 / 33] * 2)
    expected = 33 / 46 if logical_fast else 46 / 33
    assert b["GPU5_over_GPU4_time"]["linear4"] == pytest.approx([expected] * 2)
    assert a["GPU5_over_GPU4_time"]["ce_bulk"] == [1, 1]
    text = pair_compact(normal, reverse, result)
    assert "DEVICE_ORDER results_valid=True" in text
    assert "IDENTITY d0 physical=5" in text
    assert "RESULT GPU5 d0 linear4" in text


@pytest.mark.parametrize(
    "fault",
    [
        "registration",
        "visible",
        "preinitialized",
        "initialization",
        "mapping",
        "bdf",
        "edge",
        "capture",
        "waves",
        "event_nodes",
        "binding",
        "numa",
        "numa_missing",
        "partial_vma",
        "missing_window",
        "duplicate_window",
        "method_order",
        "quality",
        "negative",
        "nan",
        "drift",
        "slow_bulk",
        "failure",
    ],
)
def test_invalid_or_confounded_runs_do_not_produce_cross_device_ratios(fault):
    r = fixture((5, 4))
    if fault == "registration":
        r["registration"]["device"] = 1  # This would register physical GPU4.
    elif fault == "visible":
        r["actual_visible_devices"] = "4,5"
    elif fault == "preinitialized":
        r["cuda_was_initialized"] = True
    elif fault == "initialization":
        r["allocation_order"] = [1, 0]
    elif fault == "mapping":
        r["devices"][0]["physical_index"] = 4
    elif fault == "bdf":
        r["devices"][0]["pci_bus"] = "0000:98:00.0"
    elif fault == "edge":
        r["edge_checks"][1]["guards"] = False
    elif fault == "capture":
        r["chunk_graphs"][2]["capture_device"] = 0
    elif fault == "waves":
        r["chunk_graphs"][0]["waves"] += 1
    elif fault == "event_nodes":
        r["chunk_graphs"][0]["per_copy_timing_events"] = 64
    elif fault == "binding":
        r["bindings"][0]["checks"]["shared_host"] = False
    elif fault in ("numa", "numa_missing", "partial_vma"):
        region = r["host_snapshot"]["buffers"][0]["regions"][0]
        if fault == "numa":
            region["numa"]["pages_by_node"] = {"0": 131072}
        elif fault == "numa_missing":
            region["numa"] = None
        else:
            region["buffer_covers_whole_vma"] = False
    elif fault == "missing_window":
        r["windows"].pop()
    elif fault == "duplicate_window":
        r["windows"][-1] = copy.deepcopy(r["windows"][0])
    elif fault == "method_order":
        r["windows"][0], r["windows"][1] = r["windows"][1], r["windows"][0]
    elif fault == "quality":
        r["windows"][0]["metrics"]["valid"] = False
    elif fault in ("negative", "nan", "drift"):
        r["windows"][0]["metrics"]["logical_wall_GBps"] = {
            "negative": -1,
            "nan": float("nan"),
            "drift": 10,
        }[fault]
    elif fault == "slow_bulk":
        for w in r["windows"]:
            if w["method"] == "ce_bulk":
                w["metrics"]["logical_wall_GBps"] = 30
    else:
        r["success"] = False
    assert not report_status(r)["ready"]
    result = compare_reports(fixture(), r)
    assert not result["results_valid"] and result["comparisons"] == []
    assert "CROSS physical_order" not in pair_compact(fixture(), r, result)


@pytest.mark.parametrize(
    "fault", ["uuid", "jit", "code", "runtime", "same_process", "same_order"]
)
def test_pair_requires_matching_provenance_and_two_opposite_fresh_processes(fault):
    normal, reverse = fixture(), fixture((5, 4))
    if fault == "uuid":
        reverse["devices"][0]["uuid"] = "another-GPU"
    elif fault == "jit":
        reverse["jit_artifact"]["sha256"] = "c" * 64
    elif fault == "code":
        reverse["code"]["sha256"]["probe.py"] = "c" * 64
    elif fault == "runtime":
        reverse["runtime"]["cuda"] = "different"
    elif fault == "same_process":
        reverse["pid"] = normal["pid"]
    else:
        reverse = copy.deepcopy(normal)
        reverse["pid"] += 1
    assert report_status(normal)["ready"] and report_status(reverse)["ready"]
    result = compare_reports(normal, reverse)
    assert not result["results_valid"] and result["comparisons"] == []


def test_offline_cli_round_trip_and_partial_acquisition_report(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/pd_test/bench_h2d_device_order.py"
    env = dict(os.environ, PYTHONPATH=str(root / "python"))
    paths = []
    for name, r in (("normal", fixture()), ("reversed", fixture((5, 4)))):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(r))
        paths.append(p)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "summarize",
            "--normal",
            str(paths[0]),
            "--reversed",
            str(paths[1]),
            "--output-dir",
            str(tmp_path / "summary"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DEVICE_ORDER results_valid=True" in result.stdout
    assert json.loads((tmp_path / "summary/order.json").read_text())["results_valid"]
    partial = dict(success=False, config=configuration([4, 5]), error="No CUDA")
    assert not report_status(partial)["ready"]
    assert "No CUDA" in compact_report(partial)


@pytest.mark.parametrize("fail_order", ["", "4,5", "5,4"])
def test_wrapper_reverses_visibility_in_separate_processes_and_stops_on_failure(
    tmp_path, fail_order
):
    root = Path(__file__).resolve().parents[3]
    for order in ((4, 5), (5, 4)):
        (tmp_path / ("%s,%s.json" % order)).write_text(json.dumps(fixture(order)))
    interpreter = tmp_path / "probe-python"
    interpreter.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
from pathlib import Path
if sys.argv[2] != "run":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
order = sys.argv[sys.argv.index("--physical-order") + 1]
assert os.environ["CUDA_VISIBLE_DEVICES"] == order
if os.environ["H2D_TEST_FAIL"] == order:
    print("simulated GPU worker failure")
    raise SystemExit(7)
from sglang.benchmark.hisparse_h2d_device_order import compact_report
r = json.loads((Path(os.environ["H2D_TEST_FIXTURES"]) / (order + ".json")).read_text())
r["pid"] = os.getpid()
out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
(out / "order.json").write_text(json.dumps(r))
(out / "order-compact.txt").write_text(compact_report(r))
""")
    interpreter.chmod(0o755)
    numa = tmp_path / "numactl"
    numa.write_text(
        '#!/bin/sh\n[ "$1" = "--cpunodebind=1" ] || exit 9\n[ "$2" = "--membind=1" ] || exit 9\nshift 2\nexec "$@"\n'
    )
    numa.chmod(0o755)
    out = tmp_path / "out"
    env = dict(
        os.environ,
        LAGOM_PY=str(interpreter),
        H2D_TEST_FIXTURES=str(tmp_path),
        H2D_TEST_FAIL=fail_order,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
    )
    proc = subprocess.run(
        ["bash", "scripts/pd_test/run_h2d_device_order.sh", str(out)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    if fail_order:
        assert proc.returncode == 7
        assert "simulated GPU worker failure" in proc.stderr
        assert not (out / "order.json").exists()
        if fail_order == "4,5":
            assert not (out / "reversed").exists()
    else:
        assert proc.returncode == 0, proc.stderr
        assert "DEVICE_ORDER results_valid=True" in proc.stdout
        assert json.loads((out / "order.json").read_text())["checks"][
            "independent_processes"
        ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
