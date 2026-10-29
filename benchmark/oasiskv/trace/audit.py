"""Audit same-snapshot isolation or independent-server reproducibility.

A mismatch is not by itself proof that a shadow replay changed model state.
The per-window replay audit tests restoration within the probes server.
"""

import argparse
import gzip
import hashlib
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


def windows(directory, filename="trace.jsonl.gz", *, paired=False):
    if (directory / "rejection.json").exists():
        raise ValueError(
            f"Rejected shadow replay in {directory}; inspect rejection.json"
        )
    result = {}
    with gzip.open(directory / filename, "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if r["kind"] != "window":
                continue
            # Generation counters can differ between processes; stable request
            # IDs and request-local window indices identify the same trajectory.
            key = tuple(r[x] for x in ("request_id", "window_id", "layer", "head"))
            if paired:
                key = tuple(
                    r[x]
                    for x in (
                        "run_id",
                        "request_id",
                        "generation",
                        "window_id",
                        "layer",
                        "head",
                    )
                )
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
            if paired:
                result[key].update(
                    {
                        field: r[field]
                        for field in (
                            "snapshot_id",
                            "committed_output_ids",
                            "prompt_len",
                            "history_units",
                        )
                    }
                )
    if not result:
        raise ValueError("No baseline windows to audit")
    return result


class AuditMismatch(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_within_server(directory):
    """Same proposal/snapshot before and after shadows; no cross-server premise."""
    for name in ("failure.json", "teardown-failure.json", "rejection.json"):
        if (directory / name).exists():
            raise ValueError(f"Failed run: {directory / name}")
    required = (
        "paired-baseline.jsonl.gz",
        "replay-audit.jsonl",
        "config.json",
        "runtime.json",
    )
    for name in required:
        if not (directory / name).exists():
            raise ValueError(
                f"Missing {name}; collect a new paired run, do not relabel old traces"
            )
    config = json.loads((directory / "config.json").read_text())
    runtime = json.loads((directory / "runtime.json").read_text())
    if config.get("shadows") is not True or not config.get("probes"):
        raise ValueError(
            "Within-server quality audit requires shadows and probe P values"
        )
    left = windows(directory, "paired-baseline.jsonl.gz", paired=True)
    right = windows(directory, paired=True)
    if left.keys() != right.keys():
        raise ValueError("Paired baseline and final window keys differ")
    for key in left:
        fields = [field for field in left[key] if left[key][field] != right[key][field]]
        if fields:
            raise ValueError(f"Within-server baseline changed at {key}: {fields}")

    grouped = defaultdict(dict)
    for key, record in right.items():
        grouped[key[:4]][key[4:]] = record
    diagnostics = {}
    for line in (directory / "replay-audit.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = tuple(
            record[x] for x in ("run_id", "request_id", "generation", "window_id")
        )
        if key in diagnostics:
            raise ValueError(f"Duplicate replay diagnostic {key}")
        diagnostics[key] = record
    if diagnostics.keys() != grouped.keys():
        raise ValueError("Replay diagnostics do not cover exactly the recorded windows")

    actual_outputs = outputs(directory)
    if set(actual_outputs) != {key[1] for key in grouped}:
        raise ValueError("Request outputs and paired windows differ")
    expected_layers = {(int(layer), 0) for layer in runtime["c4_layer_to_model_layer"]}
    for key, layers in grouped.items():
        diagnostic = diagnostics[key]
        if set(layers) != expected_layers or sorted(
            layer for layer, _ in layers
        ) != diagnostic.get("layer_ids"):
            raise ValueError(f"Incomplete paired C4 layer inventory at {key}")
        if not (
            diagnostic.get("paired_audit_version") == 1
            and diagnostic.get("passed") is True
            and diagnostic.get("logits_exact_equal") is True
            and diagnostic.get("hidden_states_exact_equal") is True
            and diagnostic.get("differing_layers") == []
            and diagnostic.get("trial_predictions")
            == diagnostic.get("replayed_predictions")
        ):
            raise ValueError(f"Missing or failed strict replay evidence at {key}")
        canonical = next(iter(layers.values()))
        for field in ("snapshot_id", "input_ids", "topk_order", "verify_candidates"):
            if canonical[field] != diagnostic[field]:
                raise ValueError(f"Replay evidence {field} differs at {key}")
        if canonical["verify_candidates"] != config["verify_candidates"] or canonical[
            "topk_order"
        ] != config.get("topk_order", "native"):
            raise ValueError(f"Trace configuration differs at {key}")
        for record in layers.values():
            for field in (
                "input_ids",
                "commit_len",
                "accepted_drafts",
                "committed_output_ids",
                "prefix_len",
                "prompt_len",
                "snapshot_id",
            ):
                if record[field] != canonical[field]:
                    raise ValueError(f"Inconsistent {field} across layers at {key}")
        predicted = diagnostic["trial_predictions"]
        inputs = canonical["input_ids"]
        if len(predicted) != len(inputs):
            raise ValueError(f"Replay prediction width differs at {key}")
        accepted = 0
        for candidate, token in zip(inputs[1:], predicted):
            if candidate != token:
                break
            accepted += 1
        if (
            canonical["commit_len"] != accepted + 1
            or canonical["accepted_drafts"] != accepted
            or canonical["committed_output_ids"] != predicted[: accepted + 1]
        ):
            raise ValueError(
                f"Greedy acceptance does not match paired baseline at {key}"
            )
        # The initial anchor is the token emitted by prefill. Last-step outputs
        # may be truncated by max_new_tokens; compare the visible committed part.
        offset = canonical["prefix_len"] - canonical["prompt_len"]
        emitted = actual_outputs[key[1]]
        if not 0 <= offset < len(emitted) or emitted[offset] != inputs[0]:
            raise ValueError(f"Committed trajectory anchor differs at {key}")
        visible = min(canonical["commit_len"], len(emitted) - offset - 1)
        if (
            emitted[offset + 1 : offset + 1 + visible]
            != canonical["committed_output_ids"][:visible]
        ):
            raise ValueError(f"Committed output differs at {key}")

    probes = set()
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["kind"] != "probe":
                continue
            key = tuple(
                record[x]
                for x in (
                    "run_id",
                    "request_id",
                    "generation",
                    "window_id",
                    "layer",
                    "head",
                )
            )
            source = right.get(key)
            identity = key + (record["p"], record["mode"])
            if identity in probes or source is None:
                raise ValueError(f"Duplicate or orphan probe {identity}")
            probes.add(identity)
            if (
                record["snapshot_id"] != source["snapshot_id"]
                or record["input_ids"][: len(source["input_ids"])]
                != source["input_ids"]
            ):
                raise ValueError(
                    f"Probe uses a different snapshot/proposal at {identity}"
                )
    expected_probes = {
        key + (p, mode)
        for key in right
        for p in config["probes"]
        for mode in ("exact", "resident")
    }
    if probes != expected_probes:
        raise ValueError("Missing or unexpected P/mode/layer probe records")
    return {
        "passed": True,
        "scope": "within-server-snapshot-paired",
        "requests": len(actual_outputs),
        "request_windows": len(grouped),
        "layer_windows": len(right),
        "probe_records": len(probes),
        "trace_sha256": file_digest(directory / "trace.jsonl.gz"),
        "paired_baseline_sha256": file_digest(directory / "paired-baseline.jsonl.gz"),
        "replay_audit_sha256": file_digest(directory / "replay-audit.jsonl"),
        "cross_server_reproducibility": "not_established_by_this_audit",
        "note": "Conditional prediction potential along this server's actual trajectory; not an online H2D hit rate.",
        "timing_valid": False,
    }


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
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument(
        "--scope", choices=["cross-server", "within-server"], default="cross-server"
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Cross-server differences remain failed diagnostics but do not stop a paired experiment",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.directories) != (1 if args.scope == "within-server" else 2):
        parser.error(
            "within-server needs one probes directory; cross-server needs control and probes"
        )
    if args.diagnostic_only and args.scope != "cross-server":
        parser.error("within-server audit cannot be made diagnostic-only")
    try:
        report = (
            audit_within_server(args.directories[0])
            if args.scope == "within-server"
            else audit(*args.directories)
        )
    except ValueError as error:
        failure_path = args.output.with_name(args.output.stem + "-failure.json")
        failure = getattr(error, "report", {"passed": False, "error": str(error)})
        with failure_path.open("x") as stream:
            json.dump(failure, stream, indent=2)
        if args.diagnostic_only and isinstance(error, AuditMismatch):
            print(f"Cross-server reproducibility differs; diagnostics: {failure_path}")
            return
        raise SystemExit(f"{error}; diagnostics: {failure_path}") from error
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        f"{args.scope} audit passed: {report['requests']} requests, {report['layer_windows']} layer windows"
    )


if __name__ == "__main__":
    main()
