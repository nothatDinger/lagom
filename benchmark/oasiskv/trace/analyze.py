"""Analyze paired baseline/probe traces; never infer performance from Top-K overlap.

This module consumes model measurements. It does not run a model, reconstruct
unobserved queries, simulate cache eviction, or invent transfer completion times.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 1
KEY = ("run_id", "request_id", "generation", "window_id", "layer", "head")
GROUP = (
    "run_id",
    "prompt_len",
    "layer",
    "head",
    "unit",
    "unit_tokens",
    "bytes_per_unit",
    "verify_candidates",
    "accepted_drafts",
    "next_verify_candidates",
    "p",
    "mode",
)
COUNTS = (
    "prediction_units",
    "prediction_resident_units",
    "prefetch_candidate_units",
    "prefetch_candidate_bytes",
    "next_history_units",
    "covered_history_units",
    "next_all_units",
    "next_unavailable_units",
    "next_miss_unique_units",
    "next_miss_events",
    "next_miss_bytes",
    "potential_miss_units",
    "potential_first_fetch_bytes",
    "unused_candidate_units",
    "probe_demand_h2d_bytes",
)


def _integer(record, name, minimum=0):
    value = record[name]
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _ids(values):
    if not isinstance(values, list) or any(
        type(v) is not int or v < -1 for v in values
    ):
        raise ValueError("KV IDs must be integer lists, with only -1 as padding")
    return {v for v in values if v >= 0}


def _rows(values, length):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"expected {length} Top-K rows")
    return [_ids(row) for row in values]


def validate(record):
    if record.get("schema_version") != VERSION:
        raise ValueError("unsupported trace schema_version")
    if record.get("kind") not in ("window", "probe"):
        raise ValueError("kind must be window or probe")
    for field in ("run_id", "request_id", "snapshot_id"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{field} must be a nonempty string")
    for field in ("generation", "window_id", "layer", "head", "verify_candidates"):
        _integer(record, field)
    if record.get("unit") not in ("token", "block", "c4"):
        raise ValueError("unit must be token, block or c4")
    _integer(record, "unit_tokens", 1)
    _integer(record, "bytes_per_unit", 1)
    timing = record.get("target_forward_ms")
    if timing is not None and (
        type(timing) not in (int, float) or not math.isfinite(timing) or timing < 0
    ):
        raise ValueError("target_forward_ms must be null or finite and nonnegative")
    v = record["verify_candidates"]
    if record["kind"] == "window":
        for field in ("prefix_len", "prompt_len", "accepted_drafts", "history_units"):
            _integer(record, field)
        if record["accepted_drafts"] > v:
            raise ValueError("accepted_drafts exceeds verified candidates")
        if _integer(record, "commit_len", 1) != record["accepted_drafts"] + 1:
            raise ValueError("commit_len must include accepted drafts plus one token")
        if record["history_units"] * record["unit_tokens"] > record["prefix_len"]:
            raise ValueError("history_units includes uncommitted KV")
        _ids(record["resident_after"])
        _ids(record["input_ids"])
        if len(record["input_ids"]) != 1 + v or -1 in record["input_ids"]:
            raise ValueError("input_ids must be the anchor and verified candidates")
        rows = _rows(record["topk"], 1 + v)
        events = record.get("miss_events")
        if events is not None:
            if not isinstance(events, list):
                raise ValueError("miss_events must be a list or null")
            seen_rows = set()
            for event in events:
                row = _integer(event, "row")
                ids = _ids(event["ids"])
                if row >= len(rows) or row in seen_rows:
                    raise ValueError(
                        "miss_events needs unique, valid verification rows"
                    )
                if len(ids) != len(event["ids"]) or not ids <= rows[row]:
                    raise ValueError("miss IDs must be unique and belong to that row")
                seen_rows.add(row)
            if len(seen_rows) != len(rows):
                raise ValueError(
                    "miss_events must include every row, even zero-miss rows"
                )
    else:
        p = _integer(record, "p", 1)
        if record.get("mode") not in ("resident", "exact"):
            raise ValueError("probe mode must be resident or exact")
        _rows(record["topk"], p)
        _ids(record["input_ids"])
        if len(record["input_ids"]) != 1 + v + p or -1 in record["input_ids"]:
            raise ValueError(
                "probe input_ids must include anchor, verify prefix and probes"
            )
        io = _integer(record, "probe_demand_h2d_bytes")
        if record["mode"] == "resident" and io:
            raise ValueError("resident probe must not cause synchronous history H2D")


def _key(record):
    return tuple(record[field] for field in KEY)


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def _ratios(row):
    row.update(
        history_coverage=_ratio(
            row["covered_history_units"], row["next_history_units"]
        ),
        potential_unique_miss_coverage=_ratio(
            row["potential_miss_units"], row["next_miss_unique_units"]
        ),
        candidate_precision=_ratio(
            row["potential_miss_units"], row["prefetch_candidate_units"]
        ),
        # Neither an offline set intersection nor an exact-attention replay
        # observes when background transfers finish or whether they survive eviction.
        timely_miss_coverage=None,
        saved_critical_path_ms=None,
    )


def compare(source, following, probe, cap):
    for record in (following, probe):
        for field in ("unit", "unit_tokens", "bytes_per_unit"):
            if record[field] != source[field]:
                raise ValueError(f"inconsistent {field} between matched records")
    if probe["snapshot_id"] != source["snapshot_id"]:
        raise ValueError("probe must replay the exact source snapshot")
    if probe["verify_candidates"] != source["verify_candidates"] or (
        probe["input_ids"][: len(source["input_ids"])] != source["input_ids"]
    ):
        raise ValueError("probe changed the baseline verification prefix")
    if following["prefix_len"] != source["prefix_len"] + source["commit_len"]:
        raise ValueError("next window does not follow the actual committed prefix")
    if following["prompt_len"] != source["prompt_len"]:
        raise ValueError("prompt length changed within a request generation")

    history = source["history_units"]
    resident = _ids(source["resident_after"])
    # Frequency voting is a declared offline policy. Native Top-K output need
    # not be score-sorted, so tensor position must not be treated as a rank.
    # Deduplicate within a row; repeated padding/IDs cannot gain weight.
    votes = defaultdict(float)
    for row in probe["topk"]:
        seen = set()
        for unit in row:
            if 0 <= unit < history and unit not in seen:
                votes[unit] += 1
                seen.add(unit)
    prediction = set(votes)
    fetch = sorted(prediction - resident, key=lambda unit: (-votes[unit], unit))[:cap]
    fetch = set(fetch)
    needed = set().union(*(_ids(row) for row in following["topk"]))
    old_needed = {unit for unit in needed if unit < history}
    events = following.get("miss_events")
    misses = (
        None if events is None else [unit for event in events for unit in event["ids"]]
    )
    miss_set = None if misses is None else set(misses)
    useful = None if miss_set is None else fetch & miss_set
    size = source["bytes_per_unit"]
    row = {field: source[field] for field in KEY}
    row.update({field: source[field] for field in GROUP if field in source})
    row.update(
        next_verify_candidates=following["verify_candidates"],
        p=probe["p"],
        mode=probe["mode"],
        prediction_units=len(prediction),
        prediction_resident_units=len(prediction & resident),
        prefetch_candidate_units=len(fetch),
        prefetch_candidate_bytes=len(fetch) * size,
        next_history_units=len(old_needed),
        covered_history_units=len(prediction & old_needed),
        next_all_units=len(needed),
        next_unavailable_units=len(needed - old_needed),
        next_miss_unique_units=None if miss_set is None else len(miss_set),
        next_miss_events=None if misses is None else len(misses),
        next_miss_bytes=None if misses is None else len(misses) * size,
        potential_miss_units=None if useful is None else len(useful),
        # Count each predicted unit once, even if baseline reloads it repeatedly.
        potential_first_fetch_bytes=None if useful is None else len(useful) * size,
        unused_candidate_units=None if useful is None else len(fetch - useful),
        delta_target_ms=(
            probe["target_forward_ms"] - source["target_forward_ms"]
            if probe.get("target_forward_ms") is not None
            and source.get("target_forward_ms") is not None
            else None
        ),
        probe_demand_h2d_bytes=probe["probe_demand_h2d_bytes"],
    )
    _ratios(row)
    return row


def analyze(records, cap):
    if type(cap) is not int or cap < 0:
        raise ValueError("prefetch cap must be a nonnegative integer")
    windows, probes = {}, {}
    orders = set()
    for record in records:
        validate(record)
        orders.add(record.get("topk_order", "native"))
        if not orders <= {"native", "logical"} or len(orders) > 1:
            raise ValueError("Cannot mix different topk_order experiments")
        key = _key(record)
        table = windows if record["kind"] == "window" else probes
        if record["kind"] == "probe":
            key += (record["p"], record["mode"])
        if key in table:
            raise ValueError(f"duplicate trace identity: {key}")
        table[key] = record
    skipped, rows = Counter(), []
    for probe in probes.values():
        key = _key(probe)
        source = windows.get(key)
        if source is None:
            raise ValueError(f"probe has no source window: {key}")
        # window_id is request-local, not the scheduler's global forward_ct.
        next_key = key[:3] + (key[3] + 1,) + key[4:]
        following = windows.get(next_key)
        if following is None:
            skipped["no_immediate_next_window"] += 1
            continue
        rows.append(compare(source, following, probe, cap))
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP)].append(row)
    summary = []
    for key, members in sorted(groups.items()):
        entry = dict(zip(GROUP, key))
        entry["samples"] = len(members)
        entry["samples_with_miss_events"] = sum(
            row["next_miss_events"] is not None for row in members
        )
        for name in COUNTS:
            values = [row[name] for row in members]
            # Do not combine partial I/O evidence into an apparently complete
            # group ratio. Per-sample metrics remain available in samples.csv.
            entry[name] = None if None in values else sum(values)
        costs = [
            row["delta_target_ms"]
            for row in members
            if row["delta_target_ms"] is not None
        ]
        entry["samples_with_timing"] = len(costs)
        entry["mean_delta_target_ms"] = sum(costs) / len(costs) if costs else None
        _ratios(entry)
        summary.append(entry)
    return {
        "schema_version": VERSION,
        "topk_order": next(iter(orders), "native"),
        "prefetch_cap_units": cap,
        "candidate_policy": "probe-frequency; ties-by-logical-ID; dedup; committed-history; subtract-residency; cap",
        "measurement": "offline prediction potential; no eviction or asynchronous timing simulation",
        "baseline_records": len(windows),
        "probe_records": len(probes),
        "matched_records": len(rows),
        "skipped": dict(skipped),
        "groups": summary,
        "samples": rows,
    }


def read_records(paths):
    for path in paths:
        if (path.parent / "rejection.json").exists():
            raise ValueError(
                f"Rejected shadow replay in {path.parent}; inspect rejection.json"
            )
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate(record)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--max-prefetch-units", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(read_records(args.traces), args.max_prefetch_units)
    if not report["matched_records"]:
        parser.error("no consecutive baseline windows with matching probe records")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    for name in ("samples", "groups"):
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[name][0]))
            writer.writeheader()
            writer.writerows(report[name])
    print(
        f"Matched {report['matched_records']} layer/head samples; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
