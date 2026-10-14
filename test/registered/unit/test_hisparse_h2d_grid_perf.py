import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from sglang.benchmark.hisparse_h2d_grid_perf import (
    LEGACY_LIBRARY,
    METHOD_PAIRS,
    grid_bindings,
    grid_compact_report,
    grid_method_order,
    grid_schedule,
    grid_timing_coverage,
    loaded_grid_artifact,
)
from sglang.benchmark.hisparse_h2d_pair import (
    SM_GRID_BLOCKS,
    abba_order,
    activity_check,
    copy_methods,
    expected_case_keys,
    grid_call_options,
    source_kind,
    validate_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("reverse", [False, True])
def test_per_device_order_balances_both_grid_and_source_order(reverse):
    local, orders = Counter(), {0: [], 1: []}
    for device in abba_order([0, 1], 3):
        orders[device].append(grid_method_order(local[device], reverse))
        local[device] += 1
    for device_orders in orders.values():
        for _, pair in METHOD_PAIRS:
            seen = Counter(tuple(m for m in row if m in pair) for row in device_orders)
            assert seen == {pair: 3, tuple(reversed(pair)): 3}
        assert sum(row[0] in METHOD_PAIRS[0][1] for row in device_orders) == 3
        assert all(row[-1] == "ce" for row in device_orders)


def test_grid_changes_only_runtime_launch_and_rejects_other_sweeps():
    config = dict(copy_grid_control=True, device_source_control=True)
    assert copy_methods(config) == ("sm", "sm_cta8", "sm_hbm", "sm_hbm_cta8", "ce")
    for flag in (
        "copy_iterator_control",
        "copy_vector_control",
        "copy_load_policy_control",
        "source_order_control",
        "source_layout_control",
    ):
        with pytest.raises(ValueError):
            copy_methods(dict(config, **{flag: True}))
    with pytest.raises(ValueError):
        copy_methods(dict(config, device_source_control=False))
    source = object()
    kwargs = dict(
        host_cache=source, num_blocks=4, load_policy="legacy", block_size=1024
    )
    assert grid_call_options(kwargs, "sm_cta8") == dict(kwargs, num_blocks=8)
    assert (
        kwargs["num_blocks"] == 4
        and grid_call_options(kwargs, "sm_hbm_cta8")["host_cache"] is source
    )


@pytest.mark.parametrize(
    "blocks,expected", [(4, [800] * 4), (8, [416] * 4 + [384] * 4)]
)
def test_every_cta_copies_and_total_work_is_preserved(blocks, expected):
    actual = grid_schedule(32, 100, blocks)
    assert actual["kv_per_cta"] == expected
    assert sum(expected) == actual["total_kv"] == 3200
    assert actual["copying_warps"] == blocks * 32
    assert actual["warp_request_visits"] == 32 * blocks * 32
    # Independently walk the source's request-major modulo loop.
    visits = [[] for _ in range(blocks * 32)]
    start = 0
    for row in range(32):
        for warp in range(blocks * 32):
            for miss in range((warp - start) % (blocks * 32), 100, blocks * 32):
                visits[warp].append((row, miss))
        start += 100
    assert sorted(x for warp in visits for x in warp) == [
        (r, m) for r in range(32) for m in range(100)
    ]
    assert [
        sum(len(w) for w in visits[i : i + 32]) for i in range(0, len(visits), 32)
    ] == expected


def test_bindings_reject_changed_input_or_nonlegacy_policy():
    source, hbm, plan = torch.zeros(8), torch.zeros(8), torch.zeros((2, 3))
    shared = {
        k: torch.zeros(2)
        for k in ("miss_dst", "miss_count", "num_real_reqs", "device_buffer")
    }
    cases = {
        m: SimpleNamespace(
            device=0,
            label="test",
            banks=2,
            nbytes=100,
            source=hbm if source_kind(m) == "device" else source,
            plan_src=plan,
            sm_kwargs=dict(
                shared,
                num_blocks=b,
                block_size=1024,
                load_policy="legacy",
                is_dsv4_layout=True,
                item_size_bytes=584,
            ),
        )
        for m, b in SM_GRID_BLOCKS.items()
    }
    assert all(grid_bindings(cases).values())
    cases["sm_cta8"].plan_src = plan.clone()
    assert not grid_bindings(cases)["same_source_plan"]
    cases["sm_cta8"].sm_kwargs["load_policy"] = "ca"
    assert not grid_bindings(cases)["only_grid_changes"]


def test_one_loaded_legacy_library_is_required(tmp_path):
    path = tmp_path / LEGACY_LIBRARY
    path.write_bytes(b"fixture, not CUDA")
    maps = f"0001-0002 r-xp 0000 08:01 42 {path}\n" * 2
    artifact = loaded_grid_artifact(maps)
    assert artifact["variant"] == "legacy" and len(artifact["sha256"]) == 64
    with pytest.raises(ValueError):
        loaded_grid_artifact("")
    other = tmp_path / "other" / LEGACY_LIBRARY
    other.parent.mkdir()
    other.write_bytes(b"different")
    with pytest.raises(ValueError):
        loaded_grid_artifact(maps + f"0001-0002 r-xp 0000 08:01 42 {other}\n")


def test_api_uses_identical_jit_specialization_for_both_runtime_grids(monkeypatch):
    from sglang.kernels.ops.kvcache import hisparse

    jit_calls, launches = [], []
    module = SimpleNamespace(copy_cache_planned=lambda *a: launches.append(a))

    def load(*args):
        jit_calls.append(args)
        return module

    monkeypatch.setattr(hisparse, "_jit_copy_planned_module", load)
    common = dict(
        miss_src=torch.zeros((1, 1), dtype=torch.int64),
        miss_dst=torch.zeros((1, 1), dtype=torch.int32),
        miss_count=torch.ones(1, dtype=torch.int32),
        num_real_reqs=torch.ones(1, dtype=torch.int32),
        host_cache=torch.zeros((1, 37440), dtype=torch.uint8),
        device_buffer=torch.zeros((1, 37440), dtype=torch.uint8),
        item_size_bytes=584,
        is_dsv4_layout=True,
        load_policy="legacy",
        block_size=1024,
    )
    for method in ("sm", "sm_cta8"):
        hisparse.copy_cache_planned_mla(**grid_call_options(common, method))
    assert jit_calls == [(1024, True, True, False, 0)] * 2
    assert [call[8] for call in launches] == [4, 8]
    assert all(launches[0][i] is launches[1][i] for i in (0, 1, 2, 3, 4, 6))


def captured_report():
    """Synthetic parser fixture; no hardware measurements."""
    config = dict(
        copy_grid_control=True,
        device_source_control=True,
        batch=32,
        misses=[100],
        bulk_ce_mib=64,
        devices=[0, 1],
        phases=["compute"],
        order=abba_order([0, 1], 3),
        profile=True,
    )
    report = dict(
        config=config,
        success=True,
        host_register_flags=1,
        correctness=[],
        waves=[],
        profile_waves=[],
        jit_artifacts=[dict(variant="legacy", path=LEGACY_LIBRARY, sha256="0" * 64)],
        grid_bindings=[
            dict(case="B32-miss100", device=d, valid=True, checks=dict(shared=True))
            for d in (0, 1)
        ],
    )
    times = dict(sm=100, sm_cta8=80, sm_hbm=30, sm_hbm_cta8=28, ce=40)
    events = []
    for index, (case, method, device) in enumerate(sorted(expected_case_keys(report))):
        payload = 1868800
        common = dict(
            case=case,
            method=method,
            device=device,
            grid_control=True,
            banks=2,
            payload_bytes=payload,
            h2d_bytes_per_copy=0 if source_kind(method) == "device" else payload,
            num_blocks=SM_GRID_BLOCKS.get(method),
            block_size=1024 if method != "ce" else None,
            load_policy="legacy" if method != "ce" else "CE",
            copy_us=[times[method] + 4] * 2,
        )
        report["correctness"].append(
            dict(
                case=case,
                method=method,
                device=device,
                all_payload_bytes_checked=True,
                graph_replay_checked=True,
                untouched_bytes_checked=method != "ce",
            )
        )
        for trial, d in enumerate(config["order"]):
            if d != device:
                continue
            wave = dict(common, phase="compute", trial=trial)
            if not case.startswith("bulk-ce-"):
                local = config["order"][:trial].count(device)
                wave.update(
                    local_trial=local,
                    method_position=grid_method_order(local).index(method),
                )
            report["waves"].append(wave)
        scope = f"h2d_pair:{case}:{method}:device{device}"
        base = index * 1000
        events.append(dict(ph="X", cat="user_annotation", name=scope, ts=base, dur=400))
        for node in range(2):
            args = dict(device=device)
            if method == "ce":
                name, cat = "Memcpy HtoD", "gpu_memcpy"
                args["bytes"] = payload
            else:
                name, cat = (
                    "copy_cache_planned_kernel<1024, true, true, false, 0>",
                    "kernel",
                )
                args.update(grid=[SM_GRID_BLOCKS[method], 1, 1], block=[1024, 1, 1])
            events.append(
                dict(
                    ph="X",
                    name=name,
                    cat=cat,
                    ts=base + 10 + node * (times[method] + 4),
                    dur=times[method],
                    args=args,
                )
            )
        report["profile_waves"].append(
            dict(common, scope=scope, phase="profile_compute", trial=0)
        )
    assert validate_profile(report, events)
    return report, events


def test_complete_capture_reports_grid_proof_and_paired_times():
    report, _ = captured_report()
    assert grid_timing_coverage(report) == dict(valid=True, recorded=72, expected=72)
    text = grid_compact_report(report)
    assert "correctness=12/12 graph=12/12 untouched=8/8 results_valid=True" in text
    assert "GRID_ACTIVITY valid=True verified=8/8" in text
    assert "8/4=0.8077" in text  # Event intervals 84 / 104, not activity 80 / 100.
    assert "CTA4/8_activity_us=100.000/80.000 8/4=0.8000" in text
    assert "H2D_bytes/copy=0" in text


@pytest.mark.parametrize(
    "fault", ["missing_grid", "wrong_grid", "wrong_block", "wrong_metadata"]
)
def test_profile_rejects_missing_or_wrong_actual_launch(fault):
    report, events = captured_report()
    wave = next(w for w in report["profile_waves"] if w["method"] == "sm_cta8")
    selected = [
        e
        for e in events
        if e["cat"] == "kernel"
        and e["args"]["device"] == wave["device"]
        and e["args"]["grid"] == [8, 1, 1]
    ]
    for event in selected:
        if fault == "missing_grid":
            del event["args"]["grid"]
        elif fault == "wrong_grid":
            event["args"]["grid"] = [4, 1, 1]
        elif fault == "wrong_block":
            event["args"]["block"] = [512, 1, 1]
    if fault == "wrong_metadata":
        wave["num_blocks"] = 4
    assert not activity_check(events, wave)["valid"]
    assert not validate_profile(report, events)
    assert "results_valid=False" in grid_compact_report(report)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "wrong_order",
        "wrong_grid",
        "wrong_banks",
        "nan",
        "wrong_bytes",
        "binding",
    ],
)
def test_bad_acquisition_never_passes(fault):
    report, _ = captured_report()
    wave = next(w for w in report["waves"] if w["method"] == "sm_cta8")
    if fault == "missing":
        report["waves"].remove(wave)
    elif fault == "duplicate":
        report["waves"].append(copy.deepcopy(wave))
    elif fault == "wrong_order":
        wave["method_position"] = 4
    elif fault == "wrong_grid":
        wave["num_blocks"] = 4
    elif fault == "wrong_banks":
        wave["banks"] = 3
    elif fault == "nan":
        wave["copy_us"][0] = float("nan")
    elif fault == "wrong_bytes":
        wave["h2d_bytes_per_copy"] = 0
    else:
        report["grid_bindings"][0]["checks"]["shared"] = False
    assert "results_valid=False" in grid_compact_report(report)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
