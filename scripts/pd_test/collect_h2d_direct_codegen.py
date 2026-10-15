"""Read-only cuobjdump check of the exact legacy/direct libraries; no GPU work."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import kernel_memory_ops, sha256
from sglang.benchmark.hisparse_h2d_direct import codegen_shape, evidence_valid
from sglang.benchmark.hisparse_h2d_kv_bridge import report_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    tool = shutil.which(args.cuobjdump)
    if tool is None:
        parser.error(
            "cuobjdump not found; point --cuobjdump at the existing CUDA installation"
        )
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(success=False, input_sha256=sha256(args.bridge_json), kernels=[])
    try:
        acquired = json.loads(args.bridge_json.read_text())
        if (
            not acquired["config"].get("direct_index_control")
            or not report_status(acquired)["ready"]
        ):
            raise ValueError("Require a passed direct-index correctness check")
        report["tool_version"] = subprocess.run(
            [tool, "--version"], text=True, capture_output=True, check=True, timeout=30
        ).stdout
        artifacts = [
            a for a in acquired["jit_artifacts"] if a["variant"] in ("legacy", "direct")
        ]
        for artifact in artifacts:
            variant = artifact["variant"]
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Library hash changed: {artifact['path']}")
            for kind, option in (
                ("sass", "--dump-sass"),
                ("resources", "--dump-resource-usage"),
            ):
                with (args.output_dir / f"{variant}.{kind}.txt").open("w") as output:
                    result = subprocess.run(
                        [tool, option, artifact["path"]],
                        text=True,
                        stdout=output,
                        stderr=subprocess.PIPE,
                        timeout=60,
                    )
                if result.returncode:
                    raise RuntimeError(f"cuobjdump {variant}/{kind}: {result.stderr}")
            name = (
                "hisparse_c4_direct_kernel"
                if variant == "direct"
                else "copy_cache_planned_kernel"
            )
            functions = kernel_memory_ops(
                (args.output_dir / f"{variant}.sass.txt").read_text(), name
            )
            resources = [
                line.strip()
                for line in (args.output_dir / f"{variant}.resources.txt")
                .read_text()
                .splitlines()
                if "REG:" in line
            ]
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Library changed during extraction")
            report["kernels"].append(
                dict(**artifact, functions=functions, resource_lines=resources)
            )
        report["success"] = (
            all(codegen_shape(k) for k in report["kernels"])
            and len(report["kernels"]) == 2
        )
        if not evidence_valid(report, artifacts):
            report["success"] = False
            raise ValueError(
                "Expected three LDG.E.64.CONSTANT + three STG.E.64, no vector/local/spill; keep dumps for review"
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (args.output_dir / "codegen.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        f'CODEGEN success={report["success"]} artifacts={len(report["kernels"])}/2 scalar_shape_valid={report["success"]}',
        "Static SASS counts and resource limits only; not PCIe transactions or dynamic overlap.",
    ]
    for row in report["kernels"]:
        lines.append(f'RESOURCES {row["variant"]} {row["resource_lines"]}')
        for function in row["functions"]:
            lines.append(
                f'SASS {row["variant"]} arch={function["arch"]} opcodes={function["opcodes"]} sha256={row["sha256"]}'
            )
    if report.get("error"):
        lines.append("ERROR " + report["error"])
    compact = "\n".join(lines) + "\n"
    (args.output_dir / "codegen-compact.txt").write_text(compact)
    print(compact, end="")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
