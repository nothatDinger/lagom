import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sglang.benchmark.hisparse_h2d_codegen import (
    POLICIES,
    kernel_memory_ops,
    loaded_copy_artifacts,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def libraries(tmp_path):
    lines = []
    for code in POLICIES:
        path = (
            tmp_path
            / f"sgl_kernel_jit_sparse_copy_planned_1024_True_True_False_{code}.so"
        )
        path.write_bytes(f"fixture-{code}".encode())
        lines.extend([f"0001-0002 r-xp 0000 08:01 42 {path}"] * 2)
    return "\n".join(lines)


def test_manifest_selects_mapped_library_not_newest_cache_entry(tmp_path):
    maps = libraries(tmp_path)
    (tmp_path / "unmapped.so").write_bytes(b"newer")
    artifacts = loaded_copy_artifacts(maps)
    assert [r["code"] for r in artifacts] == [0, 1, 3, 4]
    assert all(len(r["sha256"]) == 64 for r in artifacts)
    with pytest.raises(ValueError, match="Expected one loaded"):
        loaded_copy_artifacts("\n".join(maps.splitlines()[:-2]))
    second = tmp_path / "other"
    second.mkdir()
    with pytest.raises(ValueError, match="Expected one loaded"):
        loaded_copy_artifacts(maps + "\n" + libraries(second))


def sass_fixture(demangled=False):
    name = (
        "void sglang::copy_cache_planned_kernel<1024, true, true, false, 4>()"
        if demangled
        else "_Z25copy_cache_planned_kernelILi1024ELb1ELb1ELb0ELi4EEvv"
    )
    return f"""arch = sm_90a
Function : {name}
/*0010*/ @!P0 LDG.E.128.STRONG.SM R4, [R2.64] ; /* 0x00000000 */
/*0020*/ LDG.E.64.STRONG.SM R8, [R12.64] ; /* 0x00000000 */
/*0030*/ STG.E.64 [R6.64], R8 ; /* 0x00000000 */
Function : unrelated()
/*0010*/ LDG.E.128 R4, [R2.64] ; /* 0x00000000 */
"""


@pytest.mark.parametrize("demangled", [False, True])
def test_sass_counts_only_copy_kernel_and_preserves_memory_instructions(demangled):
    kernels = kernel_memory_ops(sass_fixture(demangled))
    assert len(kernels) == 1 and kernels[0]["arch"] == "sm_90a"
    assert kernels[0]["opcodes"] == {
        "LDG.E.128.STRONG.SM": 1,
        "LDG.E.64.STRONG.SM": 1,
        "STG.E.64": 1,
    }
    assert len(kernels[0]["memory_ops"]) == 3
    with pytest.raises(ValueError):
        kernel_memory_ops("No device code")


def test_offline_dump_hash_checks_before_running_cuobjdump(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[3] / "scripts/pd_test/collect_h2d_codegen.py"
    )
    spec = importlib.util.spec_from_file_location("codegen_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    artifacts = loaded_copy_artifacts(libraries(tmp_path))
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps(dict(jit_artifacts=artifacts)))
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/cuobjdump")
    calls = []

    def run(command, **kwargs):
        from types import SimpleNamespace

        calls.append(command)
        if command[-1] == "--version":
            return SimpleNamespace(stdout="test version")
        # CUDA 12.6/13.0 cuobjdump rejects --demangle. Exercise the CLI with
        # only the supported dump option and an actual manifest library path.
        assert len(command) == 3
        assert command[1] in ("--dump-sass", "--dump-resource-usage")
        assert command[2] in {artifact["path"] for artifact in artifacts}
        kwargs["stdout"].write(
            sass_fixture() if "--dump-sass" in command else "REG: 32"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    output = tmp_path / "dump"
    monkeypatch.setattr(
        sys, "argv", ["collect", "--pair-json", str(pair), "--output-dir", str(output)]
    )
    assert cli.main() == 0
    assert len(calls) == 9
    saved = json.loads((output / "codegen.json").read_text())
    assert saved["success"] and len(saved["kernels"]) == 4
    Path(artifacts[0]["path"]).write_bytes(b"changed")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--pair-json",
            str(pair),
            "--output-dir",
            str(tmp_path / "changed-dump"),
        ],
    )
    calls.clear()
    assert cli.main() == 1
    assert len(calls) == 1  # Only --version; do not inspect a changed library.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
