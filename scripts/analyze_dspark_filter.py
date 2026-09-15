#!/usr/bin/env python3
"""Summarize DSpark token-filter JSONL experiment output."""

import argparse
import json
from pathlib import Path


def summarize(records, margin: float) -> dict:
    filtered = [r for r in records if not r["kept"]]
    edge = [
        r
        for r in filtered
        if r.get("threshold") is not None
        and abs(r["survival"] - r["threshold"]) <= margin
    ]

    def rate(rows, field):
        known = [float(r[field]) for r in rows if r.get(field) is not None]
        return None if not known else sum(known) / len(known)

    return {
        "tokens": len(records),
        "filtered_tokens": len(filtered),
        "filtered_confidence_mean": (
            None
            if not filtered
            else sum(r["confidence"] for r in filtered) / len(filtered)
        ),
        "filtered_survival_mean": (
            None
            if not filtered
            else sum(r["survival"] for r in filtered) / len(filtered)
        ),
        "edge_margin": margin,
        "edge_tokens": len(edge),
        "edge_target_accept_rate": rate(edge, "accepted_by_target"),
        "edge_csa_cache_miss_rate": rate(edge, "csa_cache_miss_rate"),
        "edge_csa_samples": sum(r.get("csa_cache_miss_rate") is not None for r in edge),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.margin < 0:
        parser.error("--margin must be non-negative")
    result = {}
    for path in args.inputs:
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        result[str(path)] = summarize(rows, args.margin)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
