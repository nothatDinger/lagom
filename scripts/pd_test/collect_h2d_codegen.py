"""Dump the exact libraries recorded by vector-control acquisition; no GPU run."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import POLICIES, kernel_memory_ops, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    tool = shutil.which(args.cuobjdump)
    if tool is None:
        parser.error(
            "cuobjdump not found; pass --cuobjdump /path/to/cuda/bin/cuobjdump"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory; existing dumps are not replaced")
    pair = json.loads(args.pair_json.read_text())
    report = dict(success=False, pair_sha256=sha256(args.pair_json), kernels=[])
    try:
        artifacts = pair.get("jit_artifacts", [])
        if len(artifacts) != 4 or {a["policy"] for a in artifacts} != set(
            POLICIES.values()
        ):
            raise ValueError(
                f"Missing/ambiguous artifact manifest: {pair.get('jit_artifacts_error', artifacts)}"
            )
        report["tool_version"] = subprocess.run(
            [tool, "--version"], text=True, capture_output=True, check=True, timeout=30
        ).stdout
        for artifact in artifacts:
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Library hash changed: {artifact['path']}")
            for kind, option in [
                ("sass", "--dump-sass"),
                ("resources", "--dump-resource-usage"),
            ]:
                target = args.output_dir / f"{artifact['policy']}.{kind}.txt"
                with target.open("w") as output:
                    result = subprocess.run(
                        [tool, option, artifact["path"]],
                        text=True,
                        stdout=output,
                        stderr=subprocess.PIPE,
                        timeout=60,
                    )
                if result.returncode:
                    raise RuntimeError(f"cuobjdump {kind}: {result.stderr}")
            kernels = kernel_memory_ops(
                (args.output_dir / f"{artifact['policy']}.sass.txt").read_text()
            )
            resource_lines = [
                line.strip()
                for line in (args.output_dir / f"{artifact['policy']}.resources.txt")
                .read_text()
                .splitlines()
                if "REG:" in line
            ]
            report["kernels"].append(
                dict(**artifact, functions=kernels, resource_lines=resource_lines)
            )
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (args.output_dir / "codegen.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"CODEGEN success={report['success']} artifacts={len(report['kernels'])}/4",
        "Counts below are static SASS instructions, not PCIe request counts; LDG may also read GPU plans.",
    ]
    for artifact in report["kernels"]:
        lines.append(
            f"RESOURCES policy={artifact['policy']} lines={artifact['resource_lines']}"
        )
        for kernel in artifact["functions"]:
            lines.append(
                f"SASS policy={artifact['policy']} arch={kernel['arch']} sha256={artifact['sha256']} memory_opcodes={dict(kernel['opcodes'])}"
            )
    if "error" in report:
        lines.append("ERROR " + report["error"])
    text = "\n".join(lines) + "\n"
    (args.output_dir / "codegen-compact.txt").write_text(text)
    print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
