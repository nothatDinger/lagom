"""Offline extraction only. Human review of timer boundaries remains mandatory."""

import argparse
from collections import Counter
import json
import shutil
import subprocess
import traceback
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_iterator_codegen import (
    instruction_order,
    resource_summary,
)
from sglang.benchmark.hisparse_h2d_windows import VARIANTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cuobjdump", default="cuobjdump")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    report = dict(
        success=False,
        artifacts=[],
        timing_semantics_reviewed=False,
        performance_ready=False,
        check_report=str(args.report.resolve()),
    )
    try:
        report["input_sha256"] = sha256(args.report)
        check = json.loads(args.report.read_text())
        artifacts = check.get("jit_artifacts", [])
        if (
            not check.get("success")
            or check.get("mode") != "check"
            or len(check.get("correctness", [])) != 16
            or len(artifacts) != 4
            or {a["variant"] for a in artifacts} != set(VARIANTS)
        ):
            raise ValueError("Require successful 16-case check and four artifacts")
        tool = shutil.which(args.cuobjdump)
        if not tool:
            raise ValueError("Existing cuobjdump not found")
        report["tool_version"] = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
        for a in artifacts:
            if sha256(a["path"]) != a["sha256"]:
                raise ValueError("Artifact changed")
            outputs = {}
            for kind, option in (
                ("sass", "--dump-sass"),
                ("resources", "--dump-resource-usage"),
            ):
                target = args.output_dir / f"{a['variant']}.{kind}.txt"
                with target.open("w") as f:
                    result = subprocess.run(
                        [tool, option, a["path"]],
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=60,
                    )
                if result.returncode:
                    raise RuntimeError(result.stderr)
                outputs[kind] = target.read_text()
            kernel = (
                "copy_cache_planned_kernel"
                if a["variant"] == "legacy"
                else "copy_cache_windows_kernel"
            )
            instructions = instruction_order(outputs["sass"], kernel)
            resources = resource_summary(outputs["resources"])
            timers = [
                dict(arch=k["arch"], offset=r["offset"], text=r["text"])
                for k in instructions
                for r in k["instructions"]
                if "GLOBALTIMER" in r["text"].upper()
            ]
            row = dict(
                a,
                resources=resources,
                kernels=instructions,
                timers=timers,
                memory_opcodes=dict(
                    Counter(
                        i["opcode"]
                        for k in instructions
                        for i in k["instructions"]
                        if i["opcode"].startswith(("LDG", "STG"))
                    )
                ),
                controls_complete=all(k["controls_complete"] for k in instructions),
                timer_presence_valid=(
                    bool(timers) if a["variant"] in ("issue", "tail") else not timers
                ),
            )
            report["artifacts"].append(row)
        report["no_spill"] = all(
            a["resources"]["no_local_or_stack"] is True for a in report["artifacts"]
        )
        report["success"] = all(
            a["controls_complete"]
            and a["timer_presence_valid"]
            and a["resources"]["complete"]
            for a in report["artifacts"]
        )
    except Exception:
        report["error"] = traceback.format_exc()
    target = args.output_dir / "codegen.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    # Template is deliberately NOT an approval. Its hash binds manual review to this acquisition.
    template = dict(
        timing_semantics_reviewed=False,
        notes="",
        codegen_report=str(target.resolve()),
        codegen_sha256=sha256(target),
    )
    (args.output_dir / "review-template.json").write_text(
        json.dumps(template, indent=2) + "\n"
    )
    lines = [
        f"CODEGEN success={report['success']} artifacts={len(report['artifacts'])}/4 no_spill={report.get('no_spill')} timing_semantics_reviewed=False performance_ready=False"
    ]
    for a in report["artifacts"]:
        lines.append(
            f"RESOURCE {a['variant']} {a['resources']['rows']} SHA256={a['sha256']}"
        )
        lines.append(
            f"TIMER {a['variant']} {a['timers']} controls_complete={a['controls_complete']}"
        )
        lines.append(f"MEMORY {a['variant']} {a['memory_opcodes']}")
    if "error" in report:
        lines.append("ERROR " + report["error"])
    text = "\n".join(lines) + "\n"
    (args.output_dir / "codegen-compact.txt").write_text(text)
    print(text)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
