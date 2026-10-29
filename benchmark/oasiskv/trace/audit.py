"""Reject a probe experiment if the isolated fixed-V control trajectory changed."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def outputs(directory):
    result = {}
    for line in (directory / "outputs.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = record["request_id"]
        if key in result:
            raise ValueError(f"Duplicate request {key}")
        result[key] = record["output_ids"]
    if not result:
        raise ValueError("No request outputs to audit")
    return result


def windows(directory):
    if (directory / "rejection.json").exists():
        raise ValueError(
            f"Rejected shadow replay in {directory}; inspect rejection.json"
        )
    result = {}
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if r["kind"] != "window":
                continue
            # Generation counters can differ between processes; stable request
            # IDs and request-local window indices identify the same trajectory.
            key = tuple(r[x] for x in ("request_id", "window_id", "layer", "head"))
            if key in result:
                raise ValueError(f"Duplicate window {key}")
            result[key] = {
                x: r[x]
                for x in (
                    "prefix_len",
                    "verify_candidates",
                    "input_ids",
                    "accepted_drafts",
                    "commit_len",
                    "topk",
                    "resident_after",
                    "miss_events",
                )
            }
            # Native Top-K uses atomic scatter, not score/rank order. Keep row
            # boundaries, duplicate IDs and miss multiplicity (also for old files).
            result[key]["topk"] = [sorted(row) for row in r["topk"]]
            result[key]["miss_events"] = [
                dict(event, ids=sorted(event["ids"])) for event in r["miss_events"]
            ]
    if not result:
        raise ValueError("No baseline windows to audit")
    return result


def audit(control, probes):
    a, b = outputs(control), outputs(probes)
    if a != b:
        raise ValueError(
            "Control and probe-run output token IDs differ; reject the quality report"
        )
    left, right = windows(control), windows(probes)
    if left.keys() != right.keys():
        raise ValueError("Control and probe-run window sets differ")
    for key, value in left.items():
        if value != right[key]:
            raise ValueError(
                f"Baseline trajectory/residency changed at {key}; reject the quality report"
            )
    return {
        "passed": True,
        "requests": len(a),
        "layer_windows": len(left),
        "timing_valid": False,
    }


def acceptance_summary(directory):
    # Every C4 layer repeats the acceptance metadata. Count each request window
    # once; grouping by prompt length avoids mixing the 16K and 32K experiments.
    unique = {}
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if r["kind"] != "window":
                continue
            key = (r["request_id"], r["generation"], r["window_id"])
            value = (r["prompt_len"], r["verify_candidates"], r["accepted_drafts"])
            if key in unique and unique[key] != value:
                raise ValueError(f"Acceptance differs between layers for {key}")
            unique[key] = value
    groups = defaultdict(list)
    for length, v, accepted in unique.values():
        groups[(length, v)].append(accepted)
    return [
        {
            "prompt_len": length,
            "verify_candidates": v,
            "recorded_windows": len(values),
            "accepted_drafts_histogram": dict(sorted(Counter(values).items())),
            "mean_committed_tokens": 1 + sum(values) / len(values),
            "scope": "first recorded windows of each request; not whole-run TPS",
        }
        for (length, v), values in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("probes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.control, args.probes)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        f"Isolation audit passed: {report['requests']} requests, {report['layer_windows']} layer windows"
    )


if __name__ == "__main__":
    main()
