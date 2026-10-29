"""Check committed output and matched KV baselines across independent servers.

A mismatch is not by itself proof that a shadow replay changed model state.
The per-window replay audit tests restoration within the probes server.
"""

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
            result[key]["topk_order"] = r.get("topk_order", "native")
            result[key]["miss_events"] = [
                dict(event, ids=sorted(event["ids"])) for event in r["miss_events"]
            ]
    if not result:
        raise ValueError("No baseline windows to audit")
    return result


class AuditMismatch(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def trace_inventory(directory):
    kinds = Counter()
    windows = defaultdict(set)
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            kind = record["kind"]
            kinds[kind] += 1
            windows[kind].add((record["request_id"], record["window_id"]))
    return {
        "records_by_kind": dict(kinds),
        "request_windows_by_kind": {kind: len(keys) for kind, keys in windows.items()},
    }


def difference_sample(left, right):
    """Bound diagnostics even for thousands of differing KV IDs per window."""
    if isinstance(left, list) and isinstance(right, list):
        positions = [
            i
            for i in range(max(len(left), len(right)))
            if i >= len(left) or i >= len(right) or left[i] != right[i]
        ]
        return {
            "control_length": len(left),
            "probes_length": len(right),
            "different_positions": len(positions),
            "samples": [
                {
                    "index": i,
                    "difference": difference_sample(
                        left[i] if i < len(left) else None,
                        right[i] if i < len(right) else None,
                    ),
                }
                for i in positions[:4]
            ],
        }
    if isinstance(left, dict) and isinstance(right, dict):
        return {
            key: difference_sample(left.get(key), right.get(key))
            for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)
        }
    return {"control": left, "probes": right}


def audit(control, probes):
    a, b = outputs(control), outputs(probes)
    left, right = windows(control), windows(probes)
    counts, examples = Counter(), []
    differing_windows = 0
    for key in sorted(left.keys() & right.keys()):
        fields = [field for field in left[key] if left[key][field] != right[key][field]]
        if not fields:
            continue
        differing_windows += 1
        counts.update(fields)
        if len(examples) < 8:
            examples.append(
                {
                    "key": list(key),
                    "fields": {
                        field: difference_sample(left[key][field], right[key][field])
                        for field in fields
                    },
                }
            )
    report = {
        "passed": a == b and left.keys() == right.keys() and not differing_windows,
        "requests": len(a),
        "layer_windows": len(left),
        "output_ids_equal": a == b,
        "window_keys_equal": left.keys() == right.keys(),
        "control_only_keys": sorted(left.keys() - right.keys())[:8],
        "probes_only_keys": sorted(right.keys() - left.keys())[:8],
        "differing_layer_windows": differing_windows,
        "differing_fields": dict(counts),
        "examples": examples,
        "inventory": {
            "control": trace_inventory(control),
            "probes": trace_inventory(probes),
        },
        "scope": "Independent-server matched baseline; window Top-K, not shadow probe Top-K",
        "note": "A mismatch does not identify shadow contamination; check the within-server replay audit separately.",
        "timing_valid": False,
    }
    if not report["passed"]:
        if a != b:
            reason = "Control and probe-run output token IDs differ"
        elif left.keys() != right.keys():
            reason = "Control and probe-run window sets differ"
        else:
            reason = f"Baseline trajectory/residency changed; differing fields: {dict(counts)}"
        raise AuditMismatch(
            reason + "; matched-baseline quality report rejected", report
        )
    return report


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
    try:
        report = audit(args.control, args.probes)
    except ValueError as error:
        failure_path = args.output.with_name(args.output.stem + "-failure.json")
        failure = getattr(error, "report", {"passed": False, "error": str(error)})
        with failure_path.open("x") as stream:
            json.dump(failure, stream, indent=2)
        raise SystemExit(f"{error}; diagnostics: {failure_path}") from error
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        f"Isolation audit passed: {report['requests']} requests, {report['layer_windows']} layer windows"
    )


if __name__ == "__main__":
    main()
