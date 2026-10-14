"""CPU checks for iterator fixtures, API isolation and codegen extraction."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.benchmark.hisparse_h2d_iterator import (
    check_iterator_bytes,
    graph_states,
    iterator_expected,
    make_iterator_case,
)
from sglang.benchmark.hisparse_h2d_iterator_codegen import (
    LIBRARIES,
    decode_sm90_control,
    instruction_order,
    loaded_iterator_artifacts,
    resource_summary,
)
from sglang.kernels.ops.kvcache import hisparse
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def case():
    return make_iterator_case(torch, "cpu")


def test_fixture_contract_includes_target_and_graph_boundary_states(case):
    assert case.src.stride() == case.dst.stride() == (1280, 1)
    assert case.host.shape == case.output.shape == (205, 37440)
    for _, counts, real in graph_states(case):
        assert 0 <= real <= 65
        assert all(0 <= counts[r] <= case.capacities[r] for r in range(real))
        expected = iterator_expected(torch, case, counts, real)
        assert (expected[:, 37376:] == 165).all()
    assert sum(case.capacities) == 12957  # Odd tails and more than one task/warp.
    assert (case.src[65] == -1).all() and case.counts[65] == 999


def test_oracle_against_independent_page_slice_copy(case):
    counts = [0] * 66
    for row in (0, 31, 32, 64):
        counts[row] = 2
    expected = torch.full(case.output.shape, 165, dtype=torch.uint8)
    for row in range(65):
        for miss in range(counts[row]):
            s, d = int(case.src_cpu[row, miss]), int(case.dst_cpu[row, miss])
            sp, so = divmod(s, 64)
            dp, do = divmod(d, 64)
            expected[dp, do * 576 : (do + 1) * 576] = case.host[
                sp, so * 576 : (so + 1) * 576
            ]
            expected[dp, 36864 + do * 8 : 36864 + (do + 1) * 8] = case.host[
                sp, 36864 + so * 8 : 36864 + (so + 1) * 8
            ]
    assert torch.equal(iterator_expected(torch, case, counts, 65), expected)
    case.output.copy_(expected)
    check_iterator_bytes(torch, case, counts, 65, "cpu", "valid")
    case.output[0, -1] ^= 1
    with pytest.raises(ValueError, match="mismatch_bytes=1"):
        check_iterator_bytes(torch, case, counts, 65, "cpu", "padding")


def test_graph_oracle_tracks_changed_plan_and_host_contents(case):
    counts = [1] + [0] * 65
    before = iterator_expected(torch, case, counts, 1)
    case.src_cpu[0, :2] = case.src_cpu[0, :2].flip(0)
    assert not torch.equal(before, iterator_expected(torch, case, counts, 1))
    changed = iterator_expected(torch, case, counts, 1)
    case.host.bitwise_xor_(255)
    assert not torch.equal(changed, iterator_expected(torch, case, counts, 1))


def iterator_args(case):
    return dict(
        miss_src=case.src,
        miss_dst=case.dst,
        miss_count=case.counts,
        num_real_reqs=case.real,
        host_cache=case.host,
        device_buffer=case.output,
        depth=1,
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"depth": 0}, "depth"),
        ({"depth": 3}, "depth"),
        ({"num_blocks": 8}, "CTA4"),
        ({"block_size": 512}, "CTA4"),
    ],
)
def test_invalid_experiment_configuration_fails_before_jit(
    case, monkeypatch, change, match
):
    monkeypatch.setattr(
        hisparse, "_jit_copy_iterator_module", lambda *a: pytest.fail("Unexpected JIT")
    )
    with pytest.raises(ValueError, match=match):
        hisparse.copy_cache_planned_mla_iterator(**(iterator_args(case) | change))


def test_rocm_and_invalid_layout_fail_before_jit(case, monkeypatch):
    monkeypatch.setattr(
        hisparse, "_jit_copy_iterator_module", lambda *a: pytest.fail("Unexpected JIT")
    )
    args = iterator_args(case)
    with pytest.raises(ValueError, match="C4 rows"):
        hisparse.copy_cache_planned_mla_iterator(
            **(args | {"host_cache": case.host[:, :-1]})
        )
    with pytest.raises(ValueError, match="shape, dtype or stride"):
        hisparse.copy_cache_planned_mla_iterator(
            **(args | {"miss_count": case.counts.to(torch.int64)})
        )
    with pytest.raises(ValueError, match="CUDA device"):
        hisparse.copy_cache_planned_mla_iterator(**args)
    monkeypatch.setattr(torch.version, "hip", "test")
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        hisparse.copy_cache_planned_mla_iterator(**args)


def test_iterator_cache_is_separate_and_legacy_dispatch_stays_legacy(case, monkeypatch):
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    hisparse._jit_copy_iterator_module.cache_clear()
    monkeypatch.setattr(hisparse, "load_jit", load)
    a = hisparse._jit_copy_iterator_module(1024, 1)
    b = hisparse._jit_copy_iterator_module(1024, 2)
    assert a is not b and hisparse._jit_copy_iterator_module(1024, 1) is a
    assert len(calls) == 2 and all(c[0][0] == "sparse_copy_iterator" for c in calls)
    assert calls[0][1]["cuda_files"] == ["hisparse_copy_iterator.cuh"]
    legacy = []
    monkeypatch.setattr(
        hisparse,
        "_jit_copy_planned_module",
        lambda *a: SimpleNamespace(copy_cache_planned=lambda *a: legacy.append(a)),
    )
    args = iterator_args(case)
    args.pop("depth")
    hisparse.copy_cache_planned_mla(
        **args, item_size_bytes=584, is_dsv4_layout=True, load_policy="legacy"
    )
    assert len(legacy) == 1 and len(calls) == 2
    hisparse._jit_copy_iterator_module.cache_clear()


def manifest(tmp_path):
    lines = []
    for variant, name in LIBRARIES.items():
        path = tmp_path / name
        path.write_bytes(variant.encode())
        lines += [f"0001-0002 r-xp 0000 08:01 42 {path}"] * 2
    return "\n".join(lines)


def test_manifest_requires_exact_loaded_artifacts(tmp_path):
    maps = manifest(tmp_path)
    assert len(loaded_iterator_artifacts(maps)) == 3
    with pytest.raises(ValueError, match="Expected one loaded"):
        loaded_iterator_artifacts("\n".join(maps.splitlines()[:-2]))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="Expected one loaded"):
        loaded_iterator_artifacts(maps + "\n" + manifest(other))


def sass(name="copy_cache_planned_iterator_kernel", count=6):
    lines = [f"arch = sm_90a\nFunction : _Z{name}ILi1024ELi2EEvv"]
    for i in range(count):
        lines += [
            f"/*{i * 16:04x}*/ @P0 LDG.E.64.CONSTANT R{i * 2}, desc[UR6][R20.64] ; /* 0x1111 */",
            "  /* 0x000ea2000c1e9b00 */",
        ]
    lines += [
        "/*00f0*/ MOV R18, R0 ; /* 0x4444 */",
        "/*0100*/ STG.E.64 desc[UR6][R22.64], R0 ; /* 0x2222 */",
        "/*0110*/ @!P0 BRA 0x0 ; /* 0x3333 */",
        "Function : unrelated",
        "/*0010*/ LDL R1, [R2];",
    ]
    return "\n".join(lines)


def test_instruction_report_keeps_predicates_operands_control_and_branches():
    rows = instruction_order(sass(), "copy_cache_planned_iterator_kernel")
    row = rows[0]
    assert len(rows) == 1 and row["arch"] == "sm_90a"
    assert row["read64_before_first_store_static_hint"] == 6
    assert row["first_store_offset"] == "0100"
    assert row["memory_opcodes"] == {"LDG.E.64.CONSTANT": 6, "STG.E.64": 1}
    assert row["order"][0]["text"].startswith("@P0")
    assert len(row["order"][0]["raw"]) == 2
    assert row["order"][-1]["opcode"] == "BRA"
    assert any(i["text"] == "MOV R18, R0 ;" for i in row["read_window"])
    assert not any(i["opcode"] == "MOV" for i in row["order"])
    # Six predicated loads before a store are only a static hint, never an automatic pass.
    assert "performance_ready" not in row
    with pytest.raises(ValueError):
        instruction_order(sass(), "missing_kernel")


def test_resource_report_distinguishes_spills_and_missing_evidence():
    clean = resource_summary("REG:40 STACK:0 SHARED:0 LOCAL:0")
    assert clean["complete"] and clean["no_local_or_stack"] is True
    assert (
        resource_summary("REG:64 STACK:16 SHARED:0 LOCAL:16")["no_local_or_stack"]
        is False
    )
    assert resource_summary("REG:40")["no_local_or_stack"] is None


def test_sm90_control_decodes_reported_loads_and_unrelated_register_wait():
    # Real legacy load/store control words from round 9, and round-13 D2
    # load/SHFL prefixes. The SHFL low 40 bits are unused in this field test.
    for word, write, read, wait in [
        ("000ea2000c1e9b00", 2, 7, 0),
        ("000f2a000c1e9b00", 4, 7, 0),
        ("000f62000c1e9b00", 5, 7, 0),
        ("0043e8000c101b06", 7, 1, 0x04),
        ("0103e8000c101b06", 7, 1, 0x10),
        ("0203e2000c101b06", 7, 1, 0x20),
        ("000368000c1e9b00", 5, 1, 0),
        ("0222a40000000000", 2, 1, 0x22),
        ("000fce0000000000", 7, 7, 0),
        ("000fc60000000000", 7, 7, 0),
    ]:
        decoded = decode_sm90_control(["instruction", f"/* 0x{word} */"], "sm_90a")
        assert (
            decoded["write_barrier"],
            decoded["read_barrier"],
            decoded["wait_mask"],
        ) == (write, read, wait)
    assert decode_sm90_control(
        ["SHFL R22, R6, R5", "/* 0x0222a40000000000 */"], "sm_90a"
    )["wait_barriers"] == [1, 5]


def test_sm90_control_requires_full_upper_word_and_known_architecture():
    assert decode_sm90_control(["/* 0x000368000c1e9b00 */"], "sm_90a") is None
    assert decode_sm90_control(["LDG", "/* 0xc1e9b00 */"], "sm_90a") is None
    assert decode_sm90_control(["LDG", "/* 0x000368000c1e9b00 */"], "sm_80") is None
    report = instruction_order(sass(), "copy_cache_planned_iterator_kernel")[0]
    assert report["controls_decoded"] == 6 and not report["controls_complete"]
    assert report["instructions"][0]["control"]["write_barrier"] == 2


def test_offline_collector_checks_all_hashes_and_never_claims_performance_ready(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/pd_test/collect_h2d_iterator_codegen.py"
    )
    spec = importlib.util.spec_from_file_location("iterator_codegen_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    artifacts = loaded_iterator_artifacts(manifest(tmp_path))
    source = tmp_path / "iterator.json"
    source.write_text(json.dumps(dict(success=True, jit_artifacts=artifacts)))
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/cuobjdump")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--version":
            return SimpleNamespace(stdout="test")
        assert len(command) == 3 and "--demangle" not in command
        name = (
            "copy_cache_planned_kernel"
            if "sparse_copy_planned" in command[2]
            else "copy_cache_planned_iterator_kernel"
        )
        kwargs["stdout"].write(
            sass(name)
            if command[1] == "--dump-sass"
            else "REG:40 STACK:0 SHARED:0 LOCAL:0"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["collect", "--report", str(source), "--output-dir", str(output)]
    )
    assert cli.main() == 0 and len(calls) == 7
    report = json.loads((output / "codegen.json").read_text())
    assert report["success"] and report["performance_ready"] is False
    assert "MOV R18, R0" in (output / "iterator_d2.read-window.txt").read_text()
    compact = (output / "codegen-compact.txt").read_text()
    assert "CTRL_LOAD64 iterator_d2 complete=False" in compact
    assert "W2/R7/M0x00/0x000ea2000c1e9b00" in compact
    Path(artifacts[-1]["path"]).write_bytes(b"changed")
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect", "--report", str(source), "--output-dir", str(tmp_path / "changed")],
    )
    calls.clear()
    assert cli.main() == 1 and not calls


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
