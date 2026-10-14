import copy
import itertools
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_iterator_perf import (
    HBM_METHODS,
    HOST_METHODS,
    SOURCE_FILES,
    iterator_bindings,
    iterator_compact_report,
    iterator_method_order,
    load_iterator_evidence,
    match_iterator_artifacts,
    reviewed_window,
    timing_coverage,
)
from sglang.benchmark.hisparse_h2d_pair import (
    abba_order,
    activity_check,
    copy_methods,
    expected_case_keys,
    iterator_call_options,
    source_kind,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("reverse", [False, True])
def test_order_balances_every_device_separately(reverse):
    local = Counter()
    observed = {0: [], 1: []}
    for device in abba_order([0, 1], 3):
        order = iterator_method_order(local[device], reverse)
        observed[device].append(order)
        local[device] += 1
    for orders in observed.values():
        assert len(orders) == 6
        for names in (HOST_METHODS, HBM_METHODS):
            filtered = [tuple(m for m in order if m in names) for order in orders]
            assert set(filtered) == set(itertools.permutations(names))
            for position in range(3):
                assert Counter(row[position] for row in filtered) == dict.fromkeys(
                    names, 2
                )
        assert sum(order[0] in HOST_METHODS for order in orders) == 3
        assert all(order[-1] == "ce" for order in orders)


def test_iterator_call_keeps_tensor_bindings_and_strips_legacy_only_options():
    config = dict(copy_iterator_control=True, device_source_control=True)
    assert copy_methods(config) == (*HOST_METHODS, *HBM_METHODS, "ce")
    for control in (
        "source_layout_control",
        "source_order_control",
        "copy_vector_control",
        "copy_load_policy_control",
    ):
        with pytest.raises(ValueError):
            copy_methods(dict(config, **{control: True}))
    with pytest.raises(ValueError):
        copy_methods(dict(config, device_source_control=False))
    counts = object()
    kwargs = dict(
        miss_count=counts,
        num_blocks=4,
        block_size=1024,
        item_size_bytes=584,
        is_dsv4_layout=True,
        load_policy="legacy",
    )
    for depth in (1, 2):
        actual = iterator_call_options(kwargs, f"sm_iter_d{depth}")
        assert actual == dict(
            miss_count=counts, num_blocks=4, block_size=1024, depth=depth
        )
    assert "load_policy" in kwargs  # Reporting metadata is not modified.


def test_bindings_reject_a_different_plan_or_destination():
    shared = {
        name: torch.zeros(2)
        for name in ("miss_dst", "miss_count", "num_real_reqs", "device_buffer")
    }
    shared.update(num_blocks=4, block_size=1024, load_policy="legacy")
    plan = torch.zeros((2, 3), dtype=torch.int64)
    sources = [torch.zeros(8), torch.zeros(8)]
    cases = {
        name: SimpleNamespace(
            sm_kwargs=dict(shared),
            plan_src=plan,
            source=sources[int(name in HBM_METHODS)],
            device=0,
            label="B32-miss100",
            banks=2,
            nbytes=100,
        )
        for name in (*HOST_METHODS, *HBM_METHODS)
    }
    assert all(iterator_bindings(cases).values())
    cases["sm_iter_d2"].plan_src = plan.clone()
    assert not iterator_bindings(cases)["same_source_plan"]
    cases["sm_iter_d2"].sm_kwargs["device_buffer"] = torch.zeros(2)
    assert not iterator_bindings(cases)["device_buffer"]


@pytest.mark.parametrize("depth", [1, 2])
def test_profile_matches_the_specific_iterator_depth(depth):
    wave = dict(
        scope="scope",
        device=0,
        banks=2,
        method=f"sm_iter_d{depth}",
        payload_bytes=1000,
        copy_us=[6, 7],
    )
    events = [dict(ph="X", name="scope", ts=100, dur=100, cat="user_annotation")]
    events.extend(
        dict(
            ph="X",
            name=f"void copy_cache_planned_iterator_kernel<1024, {depth}>(...)",
            ts=110 + 20 * i,
            dur=5,
            cat="kernel",
            args=dict(device=0),
        )
        for i in range(2)
    )
    assert activity_check(events, wave)["valid"]
    wave["method"] = f"sm_iter_d{3-depth}"
    assert not activity_check(events, wave)["valid"]
    wave["method"] = "sm"
    assert not activity_check(events, wave)["valid"]


def synthetic_kernel():
    """Control-word fixture, NOT an executable cubin or hardware evidence."""
    rows = []
    special = {
        0x8B0: ("LDG.E.64.CONSTANT", 5, 1, 0),
        0x900: ("LDG.E.64.CONSTANT", 5, 1, 0),
        0x920: ("LDG.E.64.CONSTANT", 5, 7, 0),
        0x950: ("BRA", 7, 7, 2),
        0xA90: ("LDG.E.64.CONSTANT", 2, 7, 0),
        0xAD0: ("LDG.E.64.CONSTANT", 3, 1, 0),
        0xAF0: ("LDG.E.64.CONSTANT", 4, 7, 0),
        0xD00: ("STG.E.64", 7, 7, 32),
    }
    for offset in range(0, 272 * 16, 16):
        opcode, write, read, wait = special.get(offset, ("NOP", 7, 7, 0))
        word = (write << 46) | (read << 49) | (wait << 52)
        text = "@P1 BRA 0xd70 ;" if offset == 0x950 else f"{opcode} ;"
        rows.append(
            dict(
                offset=f"{offset:04x}",
                opcode=opcode,
                text=text,
                raw=[f"/*{offset:04x}*/ {text}", f"/* 0x{word:016x} */"],
            )
        )
    return dict(arch="sm_90a", instructions=rows)


def test_barrier_one_is_allowed_but_v1_barrier_five_wait_is_rejected():
    kernel = synthetic_kernel()
    assert reviewed_window(kernel)["intervening_waits"] == [dict(offset="0950", mask=2)]
    row = kernel["instructions"][0x950 // 16]
    word = (7 << 46) | (7 << 49) | (0x22 << 52)
    row["raw"][1] = f"/* 0x{word:016x} */"
    with pytest.raises(ValueError, match="A completion required"):
        reviewed_window(kernel)


@pytest.mark.parametrize("fault", ["missing_word", "shfl", "branch", "load_shape"])
def test_unreviewed_control_flow_or_incomplete_words_fail(fault):
    kernel = synthetic_kernel()
    row = kernel["instructions"][0x950 // 16]
    if fault == "missing_word":
        row["raw"] = row["raw"][:1]
    elif fault == "shfl":
        row["opcode"] = "SHFL.IDX"
    elif fault == "branch":
        row["text"] = "@P1 BRA 0x0 ;"
    else:
        kernel["instructions"][0x8B0 // 16]["opcode"] = "LDG.E.128.CONSTANT"
    with pytest.raises(ValueError):
        reviewed_window(kernel)


def test_evidence_binds_source_correctness_codegen_and_loaded_libraries(tmp_path):
    repo, evidence = tmp_path / "repo", tmp_path / "evidence"
    for name in SOURCE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    artifacts = [
        dict(variant=v, sha256=v) for v in ("legacy", "iterator_d1", "iterator_d2")
    ]
    rows = [
        dict(
            device=d,
            variant=a["variant"],
            source=s,
            eager=True,
            all_bytes_and_padding=True,
            source_is_pinned=True,
            capture_stream_device=d,
            graph_phases=list(range(6)),
        )
        for d in (0, 1)
        for a in artifacts
        for s in ("host", "hbm")
    ]
    check = dict(
        success=True,
        correctness=rows,
        jit_artifacts=artifacts,
        code=dict(sha256={name: sha256(repo / name) for name in SOURCE_FILES}),
    )
    check_path, codegen_path = (
        evidence / "check/iterator.json",
        evidence / "codegen/codegen.json",
    )
    check_path.parent.mkdir(parents=True)
    codegen_path.parent.mkdir(parents=True)
    check_path.write_text(json.dumps(check))
    codegen = dict(
        success=True,
        input_sha256=sha256(check_path),
        artifacts=[
            dict(
                **a,
                resources=dict(complete=True, no_local_or_stack=True),
                kernels=[synthetic_kernel()],
            )
            for a in artifacts
        ],
    )
    codegen_path.write_text(json.dumps(codegen))
    loaded = load_iterator_evidence(evidence, repo)
    assert match_iterator_artifacts(loaded, artifacts)
    changed = copy.deepcopy(artifacts)
    changed[-1]["sha256"] = "recompiled"
    with pytest.raises(ValueError, match="Loaded JIT artifacts differ"):
        match_iterator_artifacts(loaded, changed)
    (repo / SOURCE_FILES[0]).write_text("changed source")
    with pytest.raises(ValueError, match="source/API changed"):
        load_iterator_evidence(evidence, repo)


def synthetic_report():
    config = dict(
        copy_iterator_control=True,
        device_source_control=True,
        batch=32,
        misses=[100],
        devices=[0, 1],
        phases=["compute"],
        order=abba_order([0, 1], 3),
        bulk_ce_mib=64,
        profile=False,
    )
    report = dict(
        config=config,
        success=True,
        host_register_flags=1,
        correctness=[],
        waves=[],
        iterator_evidence=dict(valid=True),
        iterator_artifacts_match=True,
        iterator_bindings=[
            dict(case="B32-miss100", device=d, valid=True, checks=dict(shared=True))
            for d in (0, 1)
        ],
    )
    times = dict(
        zip((*HOST_METHODS, *HBM_METHODS, "ce"), (100, 80, 60, 20, 18, 16, 40))
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
        for trial, current in enumerate(config["order"]):
            if current != device:
                continue
            local = config["order"][:trial].count(device)
            wave = dict(
                phase="compute",
                case=case,
                method=method,
                device=device,
                trial=trial,
                banks=64,
                payload_bytes=1868800,
                h2d_bytes_per_copy=0 if source_kind(method) == "device" else 1868800,
                copy_us=[times[method]] * 64,
            )
            if not case.startswith("bulk-ce-"):
                wave.update(
                    local_trial=local,
                    method_position=iterator_method_order(local).index(method),
                )
            report["waves"].append(wave)
    return report


def test_compact_reports_three_separate_contrasts_and_repeat_blocks():
    report = synthetic_report()
    assert timing_coverage(report) == dict(valid=True, recorded=96, expected=96)
    text = iterator_compact_report(report)
    assert "correctness=16/16 graph=16/16 untouched=12/12 results_valid=True" in text
    assert "D1/L=0.8000" in text and "D2/D1=0.7500" in text and "D2/L=0.6000" in text
    assert "n=384 H2D_bytes/copy=0" in text


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "wrong_trial",
        "wrong_order",
        "nan",
        "bytes",
        "banks",
        "profile",
        "binding",
    ],
)
def test_bad_acquisition_is_never_reported_ready(fault):
    report = synthetic_report()
    wave = next(w for w in report["waves"] if w["method"] == "sm_iter_d2")
    if fault == "missing":
        report["waves"].remove(wave)
    elif fault == "duplicate":
        report["waves"].append(copy.deepcopy(wave))
    elif fault == "wrong_trial":
        wave["trial"] = 123
    elif fault == "wrong_order":
        wave["method_position"] = 6
    elif fault == "nan":
        wave["copy_us"][0] = float("nan")
    elif fault == "bytes":
        wave["h2d_bytes_per_copy"] = 0
    elif fault == "banks":
        wave["banks"] = 32
        wave["copy_us"] = wave["copy_us"][:32]
    elif fault == "profile":
        report["config"]["profile"] = True
        report["profile_success"] = False
    else:
        report["iterator_bindings"][0]["checks"]["shared"] = False
    assert "results_valid=False" in iterator_compact_report(report)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
