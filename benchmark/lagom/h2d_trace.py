"""Validate bounded, same-batch DSPARK traces and compare the two H2D arms."""

import json
import math
import statistics
from pathlib import Path

PHASES = ("target.swap_in_gather", "target.swap_in_reread", "target.swap_in_commit")
C4_BYTES = 584


def validate_swap_in_step(row, path, forced, markers_only):
    gather, reread, commit = PHASES
    for name in ("step", *PHASES):
        ms = row.get(name + "_ms")
        empty_span = markers_only and name == reread
        if empty_span and ms is None:
            continue
        if (
            type(ms) not in (int, float)
            or not math.isfinite(ms)
            or ms < 0
            or (ms == 0 and not empty_span)
        ):
            raise ValueError(
                f"Missing or invalid {name} span at step {row['step']} in {path}"
            )
    for field in (
        gather + "_entries",
        commit + "_entries",
        commit + "_admissions",
        reread + "_resident_hits",
        reread + "_entries",
        reread + "_mismatched_layers",
    ):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(
                f"Missing or invalid {field} counter at step {row['step']} in {path}"
            )
    expected = row[reread + "_resident_hits"] if forced else 0
    copied = row[reread + "_entries"]
    if copied != expected or row[reread + "_mismatched_layers"]:
        raise ValueError(
            f"Resident-hit copy count mismatch at step {row['step']} in {path}: "
            f"expected {expected}, copied {copied}, mismatched layers {row[reread + '_mismatched_layers']}"
        )


def read_cell(folder, cell):
    gather, reread, commit = PHASES
    folder = Path(folder)
    config = folder / "trace_config.json"
    # Earlier rank files stored the profiler setting in the attempt config,
    # and the process ID only in their filenames.
    profiled = cell.get("nsys", False)
    configured_nsys = False
    if config.is_file():
        configured_nsys = json.loads(config.read_text()).get("nsys", False)
        if configured_nsys != profiled:
            raise ValueError(
                f"Profiler setting differs from the requested cell in {config}"
            )
    ranks = {}
    implementation = None
    forced = cell["force_h2d_each_step"]
    for path in sorted((folder / "steps").glob("rank-*.json")):
        if path.name.endswith(".trace.json"):
            continue
        trace = json.loads(path.read_text())
        mode = trace.get("swap_in") or {}
        current = mode.get("implementation")
        if trace.get("schema_version") != 2 or current not in ("adaptive", "legacy"):
            raise ValueError(
                f"Independent reread copy counters require a new schema v2 trace: {path}"
            )
        if mode.get("force_h2d_each_step") is not forced:
            raise ValueError(
                f"Recorded force_h2d_each_step differs from the requested arm in {path}"
            )
        if cell.get("method") == "dhisparse" and current != "adaptive":
            raise ValueError(
                f"Expected the adaptive swap-in path for this PD cell in {path}"
            )
        if implementation is not None and implementation != current:
            raise ValueError(f"Mixed swap-in implementations across ranks in {folder}")
        implementation = current
        markers_only = current == "adaptive" and not forced
        if trace.get("nsys", configured_nsys) != profiled:
            raise ValueError(
                f"Profiler setting differs from the requested cell in {path}"
            )
        rank = str(trace["rank"])
        if rank in ranks:
            raise ValueError(f"Duplicate trace for TP rank {rank}")
        records, rows = trace["records"], trace["summary"]
        steps = list(range(cell["trace_steps"]))
        if any([r["step"] for r in data] != steps for data in (records, rows)):
            raise ValueError(f"Incomplete step window in {path}")
        bs = cell["trace_batch_size"]
        for rec, row in zip(records, rows):
            if not (
                rec["cuda_graph"]
                and rec.get("draft_cuda_graph", False)
                and rec["batch_size"] == bs
                and len(rec["accepted_tokens"]) == bs
                and len(rec["prefix_lens"]) == bs
                and min(rec["prefix_lens"]) >= cell["input_len"] - 1
            ):
                raise ValueError(f"Unexpected graph, prefix or decode batch in {path}")
            if min(rec["accepted_tokens"]) <= 0:
                raise ValueError(f"Invalid committed token count in {path}")
            validate_swap_in_step(row, path, forced, markers_only)

        def mean(key):
            return statistics.mean(r[key] for r in rows)

        native = mean(gather + "_entries") + mean(commit + "_entries")
        rereads = mean(reread + "_entries")
        tokens = statistics.mean(statistics.mean(r["accepted_tokens"]) for r in records)
        ranks[rank] = {
            "tokens_per_request_step": tokens,
            "mean_prefix_tokens": statistics.mean(
                statistics.mean(r["prefix_lens"]) for r in records
            ),
            "native_h2d_rows": native,
            "reread_rows": rereads,
            "eligible_hits": mean(reread + "_resident_hits"),
            "admissions": mean(commit + "_admissions"),
            "h2d_payload_bytes_per_step": (native + rereads) * C4_BYTES,
        }
        if profiled:
            ranks[rank]["pid"] = (
                trace["pid"] if "pid" in trace else int(path.stem.rsplit("-pid", 1)[-1])
            )
            continue
        step_ms = mean("step_ms")
        spans = [r.get(reread + "_ms") for r in rows]
        span_ms = statistics.mean(spans) if None not in spans else None
        reread_ms = 0.0 if markers_only else span_ms
        step_times = [r["step_ms"] for r in rows]
        ranks[rank].update(
            {
                "step_ms": step_ms,
                "step_iqr_ms": (
                    statistics.quantiles(step_times, n=4, method="inclusive")[::2]
                    if len(step_times) > 1
                    else None
                ),
                "decode_tpot_ms": step_ms / tokens,
                "gather_ms": mean(gather + "_ms"),
                "reread_ms": reread_ms,
                "reread_span_ms": span_ms,
                "commit_ms": mean(commit + "_ms"),
                "reread_pct_of_step": 100 * reread_ms / step_ms,
                "reread_ms_per_token": reread_ms / tokens,
            }
        )
    if set(ranks) != {"0", "1", "2", "3"}:
        raise ValueError(
            f"Expected four complete TP4 decode traces, found ranks {sorted(ranks)}"
        )
    if profiled and len({r["pid"] for r in ranks.values()}) != len(ranks):
        raise ValueError("Expected a distinct profiled process for every decode rank")
    return {"cell": cell, "implementation": implementation, "ranks": ranks}


def compare_cells(baseline, forced):
    """Acceptance on each rank. TP workers observe the same steps and are not independent samples."""
    a, b = baseline["cell"].copy(), forced["cell"].copy()
    if a.get("nsys") or b.get("nsys"):
        raise ValueError("H2D latency comparison requires two profiler-free traces")
    implementation = baseline.get("implementation")
    if implementation not in ("adaptive", "legacy") or implementation != forced.get(
        "implementation"
    ):
        raise ValueError(
            "Expected the same independently traced swap-in implementation in both arms"
        )
    if (
        a.pop("force_h2d_each_step") is not False
        or b.pop("force_h2d_each_step") is not True
        or a != b
    ):
        raise ValueError("Expected matched baseline and forced cells")
    if (
        set(baseline["ranks"]) != {"0", "1", "2", "3"}
        or baseline["ranks"].keys() != forced["ranks"].keys()
    ):
        raise ValueError("Expected four matching decode ranks in both arms")
    ranks = {}
    for rank, base in baseline["ranks"].items():
        force = forced["ranks"][rank]
        delta = {
            k: force[k] - base[k]
            for k in (
                "step_ms",
                "commit_ms",
                "admissions",
                "eligible_hits",
                "native_h2d_rows",
                "h2d_payload_bytes_per_step",
                "tokens_per_request_step",
                "decode_tpot_ms",
            )
        }
        # The adaptive control runs no reread kernel. Subtract its marker
        # overhead only when measured over the whole window.
        reference = base["reread_span_ms"]
        increment = force["reread_span_ms"] - (reference or 0)
        delta["reread_ms"] = increment
        residual = delta["step_ms"] - increment
        base_iqr, force_iqr = base["step_iqr_ms"], force["step_iqr_ms"]

        def close(key):
            return abs(delta[key]) <= 0.10 * base[key]

        checks = {
            "additional_host_payload": delta["h2d_payload_bytes_per_step"] > 0
            and force["reread_rows"] > 0,
            "step_iqr_separated": (
                force_iqr[0] > base_iqr[1] if base_iqr and force_iqr else None
            ),
            "similar_commit_work": all(
                close(k) for k in ("admissions", "eligible_hits", "native_h2d_rows")
            ),
            "similar_commit_time": close("commit_ms"),
            "similar_acceptance": close("tokens_per_request_step"),
            "step_tracks_reread": delta["step_ms"] > 0
            and increment > 0
            and abs(residual) <= max(0.05 * base["step_ms"], 0.20 * increment),
        }
        if implementation == "legacy":
            reference_kind = "legacy_scan"
        else:
            reference_kind = (
                "empty_markers" if reference is not None else "uncorrected_force_span"
            )
        ranks[rank] = {
            "delta": delta,
            "reread_reference": reference_kind,
            "step_minus_reread_delta_ms": residual,
            "checks": checks,
        }
    return {
        "passed": all(all(r["checks"].values()) for r in ranks.values()),
        "ranks": ranks,
    }
