"""Extract existing legacy/D1/D2 JIT artifacts; no GPU execution."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_iterator_codegen import (
    LIBRARIES,
    instruction_order,
    resource_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    tool = shutil.which(args.cuobjdump)
    if tool is None:
        parser.error("Pass --cuobjdump /existing/CUDA/bin/cuobjdump")
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        success=False,
        input_sha256=sha256(args.report),
        artifacts=[],
        performance_ready=False,
    )
    try:
        acquisition = json.loads(args.report.read_text())
        artifacts = acquisition.get("jit_artifacts", [])
        if (
            acquisition.get("success") is not True
            or len(artifacts) != 3
            or {a["variant"] for a in artifacts} != set(LIBRARIES)
        ):
            raise ValueError(
                "Require successful correctness and exact three-artifact manifest"
            )
        # Check all identities before invoking the tool on any library.
        for artifact in artifacts:
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f'Artifact changed: {artifact["variant"]}')
        report["tool_version"] = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
        for artifact in artifacts:
            variant = artifact["variant"]
            outputs = {}
            for kind, option in (
                ("sass", "--dump-sass"),
                ("resources", "--dump-resource-usage"),
            ):
                path = args.output_dir / f"{variant}.{kind}.txt"
                with path.open("w") as handle:
                    result = subprocess.run(
                        [tool, option, artifact["path"]],
                        text=True,
                        stdout=handle,
                        stderr=subprocess.PIPE,
                        timeout=60,
                    )
                if result.returncode:
                    raise RuntimeError(f"{variant} {kind}: {result.stderr}")
                outputs[kind] = path.read_text()
            kernel_name = (
                "copy_cache_planned_kernel"
                if variant == "legacy"
                else "copy_cache_planned_iterator_kernel"
            )
            kernels = instruction_order(outputs["sass"], kernel_name)
            resource = resource_summary(outputs["resources"])
            order = "\n".join(
                line
                for kernel in kernels
                for row in kernel["order"]
                for line in row["raw"]
            )
            (args.output_dir / f"{variant}.order.txt").write_text(order + "\n")
            window = "\n".join(
                f'{kernel["arch"]} {kernel["function"]}\n'
                + "\n".join(
                    line for row in kernel["read_window"] for line in row["raw"]
                )
                for kernel in kernels
            )
            (args.output_dir / f"{variant}.read-window.txt").write_text(window + "\n")
            report["artifacts"].append(
                dict(**artifact, kernels=kernels, resources=resource)
            )
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    lines = [
        f'CODEGEN success={report["success"]} artifacts={len(report["artifacts"])}/3 performance_ready=False review_required=True',
        "Read64-before-first-store is a static hint: review predicates/branches/addresses and earlier register consumers in *.read-window.txt; no queue-depth measurement.",
    ]
    for artifact in report["artifacts"]:
        variant, resource = artifact["variant"], artifact["resources"]
        lines.append(
            f'RESOURCE {variant} complete={resource["complete"]} no_local_or_stack={resource["no_local_or_stack"]} rows={[r["fields"] for r in resource["rows"]]}'
        )
        for kernel in artifact["kernels"]:
            lines.append(
                f'SASS {variant} arch={kernel["arch"]} opcodes={kernel["memory_opcodes"]} first_store={kernel["first_store_offset"]} read64_before_first_store_static_hint={kernel["read64_before_first_store_static_hint"]}'
            )
            sequence = [f'{row["offset"]}:{row["text"]}' for row in kernel["order"]]
            lines.append(f"ORDER {variant} {sequence}")
    if "error" in report:
        lines.append("ERROR " + report["error"])
    text = "\n".join(lines) + "\n"
    (args.output_dir / "codegen.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "codegen-compact.txt").write_text(text)
    print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
