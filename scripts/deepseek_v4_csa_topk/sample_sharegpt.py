#!/usr/bin/env python3
"""Deterministically retain the first N ShareGPT records."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as src:
        rows = json.load(src)
    if not isinstance(rows, list):
        raise ValueError("ShareGPT dataset must be a JSON array")
    selected = rows[: args.count]
    if len(selected) < args.count:
        raise ValueError(f"dataset has only {len(selected)} rows, need {args.count}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
