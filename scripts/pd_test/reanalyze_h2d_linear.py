"""Recheck existing linear.json; no GPU job or acquisition-file changes.

Exit zero when the SM comparisons pass. Full results_valid may remain false;
invalid CE windows are retained and never used for CE comparisons.
"""

import argparse
import hashlib
import json
from pathlib import Path

from sglang.benchmark.hisparse_h2d_linear import compact_report, report_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing files are not replaced")
    source = args.report.read_bytes()
    report = json.loads(source)
    if report["config"]["mode"] != "main":
        parser.error("Require a main-mode linear.json")
    status = report_status(report)
    helper = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/benchmark/hisparse_h2d_linear.py"
    )
    provenance = dict(
        input_report=str(args.report.resolve()),
        input_report_sha256=hashlib.sha256(source).hexdigest(),
        analyzer_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), helper)
        },
        full_results_valid=status["ready"],
        comparisons=status["comparisons"],
        original_window_quality_unchanged=True,
    )
    compact = "REANALYSIS " + json.dumps(provenance) + "\n" + compact_report(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "linear-compact.txt").write_text(compact)
    (args.output_dir / "reanalysis.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(compact, end="")
    return 0 if status["comparisons"]["sm"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
