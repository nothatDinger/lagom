import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.benchmark.hisparse_h2d_diagnostics import copy_geometry
from sglang.benchmark.hisparse_h2d_pair import (
    abba_order,
    activity_check,
    compact_report,
    correctness_summary,
    expected_case_keys,
    pci_path,
    validate_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_pci_path_normalizes_nvml_domain_and_retains_missing_fields(tmp_path):
    root = tmp_path / "devices" / "0000:80:04.0"
    endpoint = root / "0000:98:00.0"
    endpoint.mkdir(parents=True)
    (endpoint / "current_link_speed").write_text("32.0 GT/s PCIe\n")
    (endpoint / "current_link_width").write_text("16\n")
    links = tmp_path / "links"
    links.mkdir()
    (links / "0000:98:00.0").symlink_to(endpoint)
    result = pci_path("00000000:98:00.0", links)
    assert [p["bdf"] for p in result["chain"]] == ["0000:98:00.0", "0000:80:04.0"]
    assert result["chain"][0]["current_link_width"] == "16"
    assert "numa_node" in result["chain"][0]["errors"]
    assert pci_path("invalid", links)["error"]


def fixture():
    wave = dict(
        scope="copy-scope",
        device=0,
        banks=2,
        method="sm",
        payload_bytes=1000,
        copy_us=[6, 7],
    )
    events = [dict(ph="X", name="copy-scope", ts=100, dur=100, cat="user_annotation")]
    for device, start, duration, name, cat in [
        (0, 110, 5, "copy_cache_planned_kernel<1024>", "kernel"),
        (0, 130, 6, "copy_cache_planned_kernel<1024>", "kernel"),
        (1, 150, 20, "copy_cache_planned_kernel<1024>", "kernel"),
        (0, 210, 10, "copy_cache_planned_kernel<1024>", "kernel"),
        (0, 120, 1, "cudaGraphLaunch", "cuda_runtime"),
    ]:
        events.append(
            dict(
                ph="X",
                name=name,
                cat=cat,
                ts=start,
                dur=duration,
                args=dict(device=device),
            )
        )
    return events, wave


def test_abba_keeps_both_devices_balanced_and_can_reverse_start():
    assert abba_order([0, 1], 2) == [0, 1, 1, 0, 0, 1, 1, 0]
    assert abba_order([0, 1], 1, True) == [1, 0, 0, 1]
    with pytest.raises(ValueError):
        abba_order([0, 0], 2)


def test_activity_time_is_distinct_from_graph_event_interval():
    events, wave = fixture()
    check = activity_check(events, wave)
    assert check["valid"]
    assert check["activity_gpu_us"] == 11
    assert check["event_gpu_us"] == 13
    assert check["event_over_activity"] == pytest.approx(13 / 11)
    assert check["activity_GBps"] == pytest.approx(2000 / 11 / 1000)


def test_gpu_projection_with_same_name_does_not_duplicate_the_cpu_scope():
    events, wave = fixture()
    # The GPU range may be shorter; it is not the synchronized CPU boundary.
    events.insert(
        0,
        dict(
            ph="X",
            name=wave["scope"],
            cat="gpu_user_annotation",
            ts=112,
            dur=20,
            pid=0,
            tid=7,
            args=dict(device=0),
        ),
    )
    check = activity_check(events, wave)
    assert check["valid"] and check["activity_gpu_us"] == 11
    assert check["scope_evidence"]["named_events"] == 2
    assert check["scope_evidence"]["cpu_scopes"] == 1
    assert check["scope_evidence"]["categories"] == {
        "gpu_user_annotation": 1,
        "user_annotation": 1,
    }
    # A real duplicate CPU annotation is still ambiguous, even if identical.
    events.append(copy.deepcopy(events[1]))
    check = activity_check(events, wave)
    assert not check["valid"] and "activity_GBps" not in check
    assert check["scope_evidence"]["cpu_scopes"] == 2


@pytest.mark.parametrize("category", ["gpu_user_annotation", "cpu_op", "unknown", ""])
def test_gpu_or_unknown_scope_cannot_replace_a_missing_cpu_annotation(category):
    events, wave = fixture()
    events[0]["cat"] = category
    check = activity_check(events, wave)
    assert not check["valid"] and "activity_GBps" not in check
    assert check["scope_evidence"]["cpu_scopes"] == 0


@pytest.mark.parametrize(
    "corrupt", ["missing_node", "missing_scope", "duplicate_scope", "bad_duration"]
)
def test_incomplete_activity_coverage_never_emits_bandwidth(corrupt):
    events, wave = fixture()
    if corrupt == "missing_node":
        events.pop(1)
    elif corrupt == "missing_scope":
        events.pop(0)
    elif corrupt == "duplicate_scope":
        events.append(copy.deepcopy(events[0]))
    else:
        events[1]["dur"] = 0
    check = activity_check(events, wave)
    assert not check["valid"]
    assert "activity_GBps" not in check


def test_ce_requires_htod_activities_and_checks_recorded_byte_count():
    events, wave = fixture()
    wave["method"] = "ce"
    for event in events[1:3]:
        event.update(name="Memcpy HtoD (Pinned -> Device)", cat="gpu_memcpy")
        event["args"]["bytes"] = 1000
    assert activity_check(events, wave)["valid"]
    events[1]["args"]["bytes"] = 999
    assert not activity_check(events, wave)["valid"]


def test_conflicting_activity_and_event_clocks_do_not_emit_bandwidth():
    events, wave = fixture()
    wave["copy_us"] = [1, 1]
    check = activity_check(events, wave)
    assert not check["valid"]
    assert "activity_GBps" not in check


def test_compact_report_weights_timings_and_keeps_missing_telemetry_unknown():
    report = dict(
        success=True,
        host_register_flags=1,
        config={},
        devices={},
        waves=[],
        profile_waves=[],
    )
    for device, times in [(0, [10, 90]), (1, [20, 180])]:
        report["waves"].append(
            dict(
                device=device,
                case="small",
                method="ce",
                phase="copy",
                payload_bytes=100000,
                copy_us=times,
                before=dict(errors={"pstate": "unavailable"}),
                after=dict(errors={}),
            )
        )
    text = compact_report(report)
    assert "d0 GB/s=2.00" in text and "d1 GB/s=1.00" in text
    assert "first/second rate=2.000" in text
    assert "Pstate=unknown" in text
    assert "errors=1" in text


@pytest.fixture
def pair_cli(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/pd_test/bench_h2d_pair.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("bench_h2d_pair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_plan_uses_one_frozen_source_and_valid_request_destinations(pair_cli):
    args = SimpleNamespace(seed=0, batch=2, plan_stride=128, buffer_slots=128)
    geometry = copy_geometry(2, 70, 4 * 2**20, 4)
    plan = pair_cli.make_plan(torch, args, geometry, 0, 70)
    assert plan["src"].shape == (4, 2, 128)
    assert torch.equal(plan["src"][:, :, :70], plan["active_src"])
    assert plan["active_src"].unique().numel() == 4 * 2 * 70
    assert int(plan["active_src"].max()) < geometry["host_tokens"]
    assert int(plan["dst"][0, :70].max()) < 128
    assert int(plan["dst"][1, :70].min()) >= 192
    assert plan["destination_pages"] == 6


def test_nvml_matches_uuid_rather_than_assuming_cuda_is_physical_index(
    pair_cli, monkeypatch
):
    nv = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 3,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: ["GPU-Z", "GPU-A", "GPU-B"][handle],
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(
            busId=f"BUS{handle}".encode()
        ),
        NVML_CLOCK_SM=1,
        NVML_CLOCK_MEM=2,
        NVML_TEMPERATURE_GPU=0,
        nvmlDeviceGetClockInfo=lambda handle, clock: 1800 if clock == 1 else 2619,
        nvmlDeviceGetPerformanceState=lambda handle: 0,
        nvmlDeviceGetCurrPcieLinkGeneration=lambda handle: 5,
        nvmlDeviceGetCurrPcieLinkWidth=lambda handle: 16,
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda handle: 0,
        nvmlDeviceGetPowerUsage=lambda handle: 100000,
        nvmlDeviceGetPowerManagementLimit=lambda handle: 700000,
        nvmlDeviceGetTemperature=lambda handle, sensor: 42,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda index: SimpleNamespace(
                uuid=["a", "GPU-B"][index]
            )
        )
    )
    observer = pair_cli.Observer(fake_torch, [0, 1])
    assert observer.metadata[0]["physical_index"] == 1
    assert observer.metadata[1]["physical_index"] == 2
    sample = observer.sample(1)
    assert sample["sm_MHz"] == 1800 and sample["power_W"] == 100
    assert not sample["errors"]


def captured_pair_fixture(device_source_control=False):
    report = dict(
        success=True,
        profile_success=False,
        host_register_flags=1,
        config=dict(
            batch=2,
            misses=[2, 70],
            devices=[0, 1],
            bulk_ce_mib=1,
            device_source_control=device_source_control,
        ),
        devices={},
        waves=[],
        profile_waves=[],
        correctness=[],
    )
    trace = []
    for index, (case, method, device) in enumerate(sorted(expected_case_keys(report))):
        events, wave = fixture()
        wave.update(
            case=case,
            method=method,
            device=device,
            scope=f"h2d_pair:{case}:{method}:device{device}",
        )
        events = events[:3]  # CPU range and two actual copy nodes.
        events[0].update(name=wave["scope"], pid=123, tid=456)
        for event in events[1:]:
            event["args"]["device"] = device
            if method == "ce":
                event.update(cat="gpu_memcpy", name="Memcpy HtoD (Pinned -> Device)")
                event["args"]["bytes"] = wave["payload_bytes"]
        events.append(
            dict(
                ph="X",
                name=wave["scope"],
                cat="gpu_user_annotation",
                ts=110,
                dur=26,
                pid=device,
                tid=7,
                args=dict(device=device),
            )
        )
        for event in events:
            event["ts"] += index * 1000
        trace.extend(events)
        wave["activity_check"] = dict(valid=False, reason="Expected one scope, found 2")
        report["profile_waves"].append(wave)
        report["correctness"].append(
            dict(
                case=case,
                method=method,
                device=device,
                all_payload_bytes_checked=True,
                graph_replay_checked=True,
            )
        )
    return report, trace


def test_local_gpu_source_keeps_sm_activity_and_is_not_reported_as_h2d():
    report, events = captured_pair_fixture(device_source_control=True)
    assert len(expected_case_keys(report)) == 14
    assert validate_profile(report, events)
    assert correctness_summary(report)["complete"]
    hbm = [w for w in report["profile_waves"] if w["method"] == "sm_hbm"]
    assert len(hbm) == 4
    assert all(w["activity_check"]["activities"] == 2 for w in hbm)
    text = compact_report(report)
    assert "NOT H2D" in text
    assert "CORRECTNESS records=14/14" in text
    assert text.count("source=device h2d_bytes_per_copy=0") == 4
    report["profile_waves"] = [
        w for w in report["profile_waves"] if w["method"] != "sm_hbm"
    ]
    assert not validate_profile(report, events)


def test_correctness_and_profile_coverage_are_independent():
    report, events = captured_pair_fixture()
    assert correctness_summary(report)["complete"]
    assert report["profile_success"] is False
    assert validate_profile(report, events)
    assert report["profile_validation"]["validated"] == 10
    report["profile_waves"].pop()
    assert not validate_profile(report, events)
    assert correctness_summary(report)["complete"]
    report["profile_waves"] = []
    assert not validate_profile(report, events)
    report["correctness"].pop()
    assert not correctness_summary(report)["complete"]


@pytest.mark.parametrize(
    "failure,device_source_control",
    [
        (None, False),
        ("cpu_duplicate", False),
        ("graph_check", False),
        ("capture", False),
        (None, True),
        ("missing_hbm", True),
    ],
)
def test_offline_cli_needs_no_site_packages_and_preserves_originals(
    tmp_path, failure, device_source_control
):
    report, events = captured_pair_fixture(device_source_control)
    if failure == "cpu_duplicate":
        events.append(copy.deepcopy(events[0]))
    elif failure == "graph_check":
        report["correctness"][0]["graph_replay_checked"] = False
    elif failure == "capture":
        report["success"] = False
    elif failure == "missing_hbm":
        report["profile_waves"] = [
            w for w in report["profile_waves"] if w["method"] != "sm_hbm"
        ]
    source = tmp_path / "pair.json"
    trace = tmp_path / "pair.trace.json"
    source.write_text(json.dumps(report))
    trace.write_text(json.dumps(dict(traceEvents=events)))
    originals = source.read_bytes(), trace.read_bytes()
    script = (
        Path(__file__).resolve().parents[3] / "scripts/pd_test/reanalyze_h2d_pair.py"
    )
    output = tmp_path / "reanalyzed"
    # -S excludes site-packages, including torch; this is strictly an offline parser.
    command = [
        sys.executable,
        "-S",
        str(script),
        "--report",
        str(source),
        "--trace",
        str(trace),
        "--output-dir",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, text=True)
    assert (run.returncode == 0) == (failure is None), run.stderr
    assert (source.read_bytes(), trace.read_bytes()) == originals
    updated = json.loads((output / "pair-reanalyzed.json").read_text())
    assert updated["success"] == report["success"]
    assert updated["reanalysis"]["ready_for_main"] == (failure is None)
    expected = 14 if device_source_control else 10
    assert f"CORRECTNESS records={expected}/{expected}" in run.stdout
    assert "gpu_user_annotation" in run.stdout
    saved = (output / "pair-reanalyzed.json").read_bytes()
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert (output / "pair-reanalyzed.json").read_bytes() == saved
