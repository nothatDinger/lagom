import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.benchmark.hisparse_h2d_diagnostics import PAGE_BYTES
from sglang.benchmark.hisparse_h2d_layout import (
    layout_compact_report,
    layout_geometries,
    layout_method_pairs,
    prepare_adjacent_source,
    source_footprint,
)
from sglang.benchmark.hisparse_h2d_pair import (
    abba_order,
    copy_methods,
    expected_case_keys,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def pair_cli(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/pd_test/bench_h2d_pair.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("layout_pair_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_copy(host, source, destination, output_pages):
    """Independent Python slicing oracle: no production address helper/gather."""
    output = torch.zeros((output_pages, 37440), dtype=torch.uint8)
    flat_host, flat_output = host.flatten(), output.flatten()
    for src, dst in zip(source.flatten().tolist(), destination.flatten().tolist()):
        sp, so = divmod(src, 64)
        dp, do = divmod(dst, 64)
        for length, sa, da in [
            (576, sp * 37440 + so * 576, dp * 37440 + do * 576),
            (8, sp * 37440 + 36864 + so * 8, dp * 37440 + 36864 + do * 8),
        ]:
            flat_output[da : da + length] = flat_host[sa : sa + length]
    return output


def test_geometry_reserves_all_cases_without_silent_bank_reuse():
    geometries = layout_geometries(2, [65, 9], 16 * PAGE_BYTES, 4)
    assert [g["banks"] for g in geometries] == [2, 2]
    assert [g["dense_token_start"] for g in geometries] == [0, 384]
    assert all(g["random_token_start"] == 512 for g in geometries)
    assert all(g["random_tokens"] == 512 for g in geometries)
    main = layout_geometries(32, [100, 1272], 512 * 2**20, 8)
    assert [g["banks"] for g in main] == [8, 8]
    assert [g["payload_bytes"] for g in main] == [1868800, 23771136]
    assert main[0]["random_tokens"] >= 8 * main[1]["entries"]


@pytest.mark.parametrize(
    "misses,source_bytes,max_banks",
    [
        ([], 4 * PAGE_BYTES, 4),
        ([2, 2], 4 * PAGE_BYTES, 4),
        ([0], 4 * PAGE_BYTES, 4),
        ([65], 2 * PAGE_BYTES, 4),
        ([2], 4 * PAGE_BYTES, 1),
    ],
)
def test_bad_geometry_cannot_produce_an_aliasing_experiment(
    misses, source_bytes, max_banks
):
    with pytest.raises(ValueError):
        layout_geometries(2, misses, source_bytes, max_banks)


@pytest.mark.parametrize(
    "misses,pages,stride", [([65, 9], 16, 96), ([513, 9], 128, 640)]
)
def test_packing_preserves_all_cases_banks_value_scale_and_padding(
    pair_cli, misses, pages, stride
):
    args = SimpleNamespace(seed=19, batch=2, plan_stride=stride, buffer_slots=stride)
    geometries = layout_geometries(2, misses, pages * PAGE_BYTES, 2)
    host = torch.randint(
        0,
        256,
        (pages, 37440),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(17),
    )
    untouched_original = host.clone()
    expected_host = host.clone()
    prepared = []
    for index, (count, geometry) in enumerate(zip(misses, geometries)):
        plan = pair_cli.make_plan(torch, args, geometry, index, count)
        frozen = copy.deepcopy(plan)
        adjacent, evidence = prepare_adjacent_source(torch, host, plan, geometry)
        assert adjacent["dst"] is plan["dst"]
        assert adjacent["src"].stride(1) == plan["src"].stride(1) == stride
        assert (
            evidence["all_banks_payload_checked"]
            and evidence["original_sources_preserved"]
        )
        assert len(evidence["random"]) == len(evidence["adjacent"]) == 2
        assert all(
            row["adjacent_pairs"] == geometry["entries"] - 1
            for row in evidence["adjacent"]
        )
        for key in ("src", "dst", "active_src"):
            assert torch.equal(plan[key], frozen[key])
        assert torch.equal(adjacent["src"][:, :, count:], plan["src"][:, :, count:])
        # Build the expected HOST changes with literal C4 layout, including a
        # source/destination crossing token 63/64 and the 65th miss.
        for old, new in zip(
            plan["active_src"].flatten().tolist(),
            adjacent["active_src"].flatten().tolist(),
        ):
            sp, so = divmod(old, 64)
            dp, do = divmod(new, 64)
            for size, sa, da in [
                (576, sp * 37440 + so * 576, dp * 37440 + do * 576),
                (8, sp * 37440 + 36864 + so * 8, dp * 37440 + 36864 + do * 8),
            ]:
                expected_host.flatten()[da : da + size] = untouched_original.flatten()[
                    sa : sa + size
                ]
        prepared.append((count, plan, adjacent))
    assert torch.equal(host, expected_host)  # Entire pool, including page padding.
    # Validate the FIRST case again after packing the SECOND case: later setup
    # must not overwrite its dense banks or any original random source.
    for count, plan, adjacent in prepared:
        for bank in range(2):
            original_output = physical_copy(
                untouched_original,
                plan["active_src"][bank],
                plan["dst"][:, :count],
                plan["destination_pages"],
            )
            actual = physical_copy(
                host,
                adjacent["active_src"][bank],
                plan["dst"][:, :count],
                plan["destination_pages"],
            )
            assert torch.equal(actual, original_output)
            pair_cli.validate_sm(
                torch,
                host,
                actual,
                plan["active_src"][bank],
                plan["dst"][:, :count],
                check_untouched=True,
            )
            actual.flatten()[-1] = 1
            with pytest.raises(RuntimeError, match="untouched slots or page padding"):
                pair_cli.validate_sm(
                    torch,
                    host,
                    actual,
                    plan["active_src"][bank],
                    plan["dst"][:, :count],
                    check_untouched=True,
                )


def test_original_oracle_detects_corrupt_packed_scale(pair_cli):
    args = SimpleNamespace(seed=3, batch=2, plan_stride=96, buffer_slots=96)
    geometry = layout_geometries(2, [65], 16 * PAGE_BYTES, 2)[0]
    host = torch.randint(0, 256, (16, 37440), dtype=torch.uint8)
    plan = pair_cli.make_plan(torch, args, geometry, 0, 65)
    adjacent, _ = prepare_adjacent_source(torch, host, plan, geometry)
    slot = int(adjacent["active_src"][0, 1, 0])
    page, offset = divmod(slot, 64)
    address = page * 37440 + 36864 + offset * 8
    host.flatten()[address] ^= 1
    actual = physical_copy(
        host, adjacent["active_src"][0], plan["dst"][:, :65], plan["destination_pages"]
    )
    with pytest.raises(RuntimeError, match="scale mismatch"):
        pair_cli.validate_sm(
            torch,
            host,
            actual,
            plan["active_src"][0],
            plan["dst"][:, :65],
            check_untouched=True,
        )


@pytest.mark.parametrize(
    "corrupt",
    ["overlap", "duplicate_source", "duplicate_destination", "different_oracle"],
)
def test_invalid_plans_fail_before_packing_writes_host(pair_cli, corrupt):
    args = SimpleNamespace(seed=0, batch=2, plan_stride=96, buffer_slots=96)
    geometry = layout_geometries(2, [65], 16 * PAGE_BYTES, 2)[0]
    host = torch.randint(0, 256, (16, 37440), dtype=torch.uint8)
    plan = pair_cli.make_plan(torch, args, geometry, 0, 65)
    if corrupt == "overlap":
        plan["active_src"][0, 0, 0] = 0
    elif corrupt == "duplicate_source":
        plan["active_src"][1, 0, 0] = plan["active_src"][0, 0, 0]
    elif corrupt == "duplicate_destination":
        plan["dst"][1, 0] = plan["dst"][0, 0]
    else:
        plan["src"][0, 0, 0] += 1
    if corrupt != "different_oracle":
        plan["src"][:, :, :65] = plan["active_src"]
    frozen = host.clone()
    with pytest.raises(ValueError):
        prepare_adjacent_source(torch, host, plan, geometry)
    assert torch.equal(host, frozen)


def test_virtual_page_footprint_counts_value_and_separate_scale():
    rows = source_footprint(torch, torch.tensor([[[63, 64]]]))
    assert rows == [dict(c4_pages=2, virtual_4k_pages=3, adjacent_pairs=1, items=2)]
    row = source_footprint(torch, torch.arange(64).reshape(1, 1, 64))[0]
    assert row["c4_pages"] == 1 and row["virtual_4k_pages"] == 10
    assert row["adjacent_pairs"] == 63


def test_pair_order_balances_baseline_first_on_each_gpu():
    methods = copy_methods(dict(source_layout_control=True, device_source_control=True))
    order = abba_order([0, 1], 2)
    for device in [0, 1]:
        host_first = []
        for trial, current in enumerate(order):
            if current == device:
                pairs = layout_method_pairs(methods, trial)
                host_first.append(pairs.index("sm") < pairs.index("sm_adjacent"))
                reversed_pairs = layout_method_pairs(methods, trial, True)
                assert (
                    reversed_pairs.index("sm") < reversed_pairs.index("sm_adjacent")
                    if not host_first[-1]
                    else reversed_pairs.index("sm")
                    > reversed_pairs.index("sm_adjacent")
                )
        assert host_first.count(True) == host_first.count(False) == 2


def report_fixture():
    config = dict(
        source_layout_control=True,
        device_source_control=True,
        batch=2,
        misses=[2],
        devices=[0, 1],
        bulk_ce_mib=1,
        phases=["compute"],
        order=[0, 1, 1, 0],
        profile=False,
    )
    report = dict(
        success=True,
        config=config,
        host_register_flags=1,
        correctness=[],
        waves=[],
        profile_waves=[],
        layout_plans=[],
        layout_bindings=[],
    )
    for case, method, device in expected_case_keys(report):
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
    report["layout_plans"] = [
        dict(
            case="B2-miss2",
            banks=2,
            payload_bytes=2336,
            source_row_stride=96,
            dense_token_start=0,
            dense_bank_tokens=64,
            random_token_start=128,
            random_tokens=1024,
            all_banks_payload_checked=True,
            original_sources_preserved=True,
            destination_plan_shared=True,
            random=[dict(c4_pages=4, virtual_4k_pages=8)] * 2,
            adjacent=[dict(c4_pages=1, virtual_4k_pages=2)] * 2,
        )
    ]
    for device in [0, 1]:
        report["layout_bindings"].append(
            dict(case="B2-miss2", device=device, valid=True)
        )
        for trial, current in enumerate(config["order"]):
            if current != device:
                continue
            for method, times in [
                ("sm", [10, 30]),
                ("sm_adjacent", [5, 15]),
                ("sm_hbm", [5, 10]),
                ("sm_hbm_adjacent", [5, 10]),
                ("ce", [3, 5]),
            ]:
                report["waves"].append(
                    dict(
                        phase="compute",
                        case="B2-miss2",
                        device=device,
                        trial=trial,
                        method=method,
                        copy_us=times,
                        payload_bytes=2336,
                        banks=2,
                    )
                )
            report["waves"].append(
                dict(
                    phase="compute",
                    case="bulk-ce-1MiB",
                    device=device,
                    trial=trial,
                    method="ce",
                    copy_us=[3, 5],
                    payload_bytes=2**20,
                    banks=2,
                )
            )
    return report


def test_compact_report_uses_weighted_time_and_pairs_trials():
    text = layout_compact_report(report_fixture())
    assert "ready_for_main=True" in text
    assert "correctness=12/12 graph=12/12 untouched=8/8" in text
    assert "random/adjacent_us=20.00/10.00" in text
    assert "adjacent/random=0.5000 wave_ratios=[0.5, 0.5] n=4" in text
    assert "H2D_bytes/copy=0" in text
    assert "virtual_4K_pages=8.0/2.0" in text


def test_profile_activity_is_kept_separate_from_unprofiled_event():
    report = report_fixture()
    for method, active, event in [("sm", 16, 24), ("sm_adjacent", 8, 14)]:
        report["profile_waves"].append(
            dict(
                case="B2-miss2",
                device=0,
                method=method,
                activity_check=dict(
                    valid=True, activities=2, activity_gpu_us=active, event_gpu_us=event
                ),
            )
        )
    text = layout_compact_report(report)
    assert "RESULT compute B2-miss2 host d0 random/adjacent_us=20.00/10.00" in text
    assert (
        "PROFILE_RESULT B2-miss2 host d0 phase=profile_compute random/adjacent_activity_us=8.00/4.00"
        in text
    )
    assert "gap_us=4.00/3.00" in text


@pytest.mark.parametrize(
    "failure",
    ["prepare", "binding", "graph", "profile", "missing_case", "duplicate_wave"],
)
def test_incomplete_capture_cannot_pass_layout_gate(failure):
    report = report_fixture()
    if failure == "prepare":
        report["layout_plans"] = []
    elif failure == "binding":
        report["layout_bindings"].pop()
    elif failure == "graph":
        report["correctness"][0]["graph_replay_checked"] = False
    elif failure == "profile":
        report["config"]["profile"] = True
        report["profile_success"] = False
    elif failure == "missing_case":
        report["waves"] = [w for w in report["waves"] if w["case"] != "B2-miss2"]
    else:
        report["waves"].append(copy.deepcopy(report["waves"][0]))
    assert "ready_for_main=False" in layout_compact_report(report)


@pytest.mark.parametrize("failure", ["payload", "banks", "trial", "nan"])
def test_bad_timing_join_cannot_emit_a_comparison(failure):
    report = report_fixture()
    wave = next(w for w in report["waves"] if w["method"] == "sm_adjacent")
    if failure == "payload":
        wave["payload_bytes"] += 1
    elif failure == "banks":
        wave["copy_us"].pop()
    elif failure == "trial":
        wave["trial"] = 8
    else:
        wave["copy_us"][0] = float("nan")
    with pytest.raises(ValueError):
        layout_compact_report(report)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
