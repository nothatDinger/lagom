"""Wait for new steady B32 logs, then collect/analyze one bounded TP4 profile."""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decode-log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", choices=("split", "pr_fused"), required=True)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    if args.output_dir.exists() or args.timeout <= 0:
        p.error("Require new output directory and positive timeout")
    pattern = re.compile(r"#running-req\s*[:=]\s*32\b")
    count = 0
    deadline = time.monotonic() + args.timeout
    # Only new lines: stale B32 from an earlier workload cannot open the gate.
    with args.decode_log.open() as log:
        log.seek(0, 2)
        while count < 2:
            line = log.readline()
            if line:
                if pattern.search(line):
                    count += 1
                elif "#running-req" in line:
                    count = 0
            else:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "No two new B32 logs; do not profile a warmup or tail"
                    )
                time.sleep(0.5)
    script_dir = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "collect_h2d_profile.py"),
            "--output-dir",
            str(args.output_dir),
            "--steps",
            "64",
            "--activity",
            "torch",
            "--wait",
            "--expected-ranks",
            "4",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "summarize_hisparse_service_pipeline.py"),
            str(args.output_dir),
            "--backend",
            args.backend,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
