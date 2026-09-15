#!/usr/bin/env python3
"""Create a reproducible JSON-array subset for a serving experiment."""

import argparse
import json
import random
from pathlib import Path


def select_rows(rows: list, *, strategy: str, size: int, seed: int) -> list:
    if size <= 0:
        raise ValueError("sample size must be positive")
    if strategy == "all":
        return rows
    if size > len(rows):
        raise ValueError(f"requested {size} rows, but dataset contains {len(rows)}")
    if strategy == "head":
        return rows[:size]
    if strategy == "random":
        return random.Random(seed).sample(rows, size)
    raise ValueError(f"unsupported sampling strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strategy", choices=("head", "random", "all"), default="head")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as source:
        rows = json.load(source)
    if not isinstance(rows, list):
        parser.error("input dataset must be a JSON array")
    try:
        selected = select_rows(
            rows, strategy=args.strategy, size=args.size, seed=args.seed
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected {len(selected)} of {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
