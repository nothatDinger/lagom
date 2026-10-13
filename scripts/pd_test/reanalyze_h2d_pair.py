"""Recheck an existing pair.json + trace using only Python's standard library.

No torch import, CUDA context, GPU work, or modification of acquisition files.
Exit zero only when acquisition, copy correctness and all profile scopes pass.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def load_checks():
    # Import the pure helper directly: importing sglang runs torch initialization.
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/benchmark/hisparse_h2d_pair.py"
    )
    spec = importlib.util.spec_from_file_location("h2d_pair_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return json.load(source)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing files are not replaced")
    checks = load_checks()
    report = read_json(args.report)
    previous_profile_success = report.get("profile_success")
    trace = read_json(args.trace)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    checks.validate_profile(report, events)
    correct = checks.correctness_summary(report)
    ready = (
        report.get("success") is True
        and correct["complete"]
        and report["profile_success"]
    )
    report["reanalysis"] = dict(
        scope_selector="unique CPU cat=user_annotation",
        input_report=str(args.report.resolve()),
        input_trace=str(args.trace.resolve()),
        input_report_sha256=sha256(args.report),
        input_trace_sha256=sha256(args.trace),
        previous_profile_success=previous_profile_success,
        ready_for_main=ready,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "pair-reanalyzed.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    compact = checks.compact_report(report)
    (args.output_dir / "pair-compact.txt").write_text(compact)
    print(compact, end="")
    if not ready:
        raise SystemExit(
            "Reanalysis incomplete; return pair-compact.txt before pair-main"
        )


if __name__ == "__main__":
    main()
