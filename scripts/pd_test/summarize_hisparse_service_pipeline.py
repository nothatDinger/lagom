"""Join bounded real-service pipeline traces and actual miss counts, per TP rank."""

import argparse
import importlib.util
import json
from pathlib import Path

from collect_h2d_profile import capture_artifacts

ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/sglang/benchmark" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--backend", choices=("split", "pr_fused"), required=True)
    args = parser.parse_args()
    request = json.loads((args.directory / "capture-request.json").read_text())[
        "request"
    ]
    capture = request["profile_id"]
    io = load("service_trace_io", "hisparse_h2d.py")
    analysis = load("service_pipeline", "hisparse_service_pipeline.py")
    status = capture_artifacts(args.directory, request, 4)
    if not status["artifacts_complete"]:
        raise ValueError(f"Incomplete capture: {status['pending']}")
    lines = []
    for artifact in status["ranks"]:
        rank = artifact["rank"]
        probe = json.loads((args.directory / artifact["probe"]).read_text())
        if probe.get("capture_id") != capture:
            raise ValueError("Probe capture mismatch")
        events = io.read_events(
            args.directory / artifact["trace"], capture_id=capture, pid=probe["pid"]
        )
        report = analysis.analyze_service(probe, events, args.backend)
        (args.directory / f"pipeline-TP{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        lines.append(
            f"RANK {rank} backend={args.backend} valid=True GPU={report['gpu_uuid']} samples={len(report['samples'])}"
        )
        for group in report["groups"]:
            compact = {k: v for k, v in group.items() if k != "layers"}
            lines.append("GROUP " + json.dumps(compact))
            for layer in group["layers"]:
                lines.append(
                    f"LAYER B={group['batch']} W={group['verify_width']} "
                    + json.dumps(layer)
                )
    text = "\n".join(lines) + "\n"
    (args.directory / "service-pipeline-compact.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
