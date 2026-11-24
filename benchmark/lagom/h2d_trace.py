"""Validate bounded, same-batch DSPARK traces and compare the two H2D arms."""

import json
import math
import statistics
from pathlib import Path

PHASES = ("target.swap_in_gather", "target.swap_in_reread", "target.swap_in_commit")
C4_BYTES = 584


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
    for path in sorted((folder / "steps").glob("rank-*.json")):
        if path.name.endswith(".trace.json"):
            continue
        trace = json.loads(path.read_text())
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
            for name in ("step", *PHASES):
                if not math.isfinite(row[name + "_ms"]) or row[name + "_ms"] <= 0:
                    raise ValueError(f"Missing or invalid {name} span in {path}")
            expected = (
                row[reread + "_resident_hits"] if cell["force_h2d_each_step"] else 0
            )
            if row[reread + "_entries"] != expected:
                raise ValueError(
                    f"Resident-hit copy count differs from the selected arm in {path}"
                )

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
        times = [r[reread + "_ms"] for r in rows]
        reread_ms = statistics.mean(times)
        ranks[rank].update(
            {
                "step_ms": step_ms,
                "decode_tpot_ms": step_ms / tokens,
                "gather_ms": mean(gather + "_ms"),
                "reread_ms": reread_ms,
                "reread_se_ms": (
                    statistics.stdev(times) / math.sqrt(len(times))
                    if len(times) > 1
                    else None
                ),
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
    return {"cell": cell, "ranks": ranks}


def compare_cells(baseline, forced):
    """Acceptance on each rank. TP workers observe the same steps and are not independent samples."""
    a, b = baseline["cell"].copy(), forced["cell"].copy()
    if a.get("nsys") or b.get("nsys"):
        raise ValueError("H2D latency comparison requires two profiler-free traces")
    if (
        a.pop("force_h2d_each_step") is not False
        or b.pop("force_h2d_each_step") is not True
        or a != b
    ):
        raise ValueError("Expected matched baseline and forced cells")
    ranks = {}
    for rank, base in baseline["ranks"].items():
        force = forced["ranks"][rank]
        delta = {
            k: force[k] - base[k]
            for k in (
                "step_ms",
                "reread_ms",
                "commit_ms",
                "admissions",
                "eligible_hits",
                "native_h2d_rows",
                "h2d_payload_bytes_per_step",
                "tokens_per_request_step",
                "decode_tpot_ms",
            )
        }
        se = [arm["reread_se_ms"] for arm in (base, force)]
        noise = None if None in se else 2 * math.hypot(*se)
        residual = delta["step_ms"] - delta["reread_ms"]

        def close(key):
            return abs(delta[key]) <= 0.10 * base[key]

        checks = {
            "additional_host_payload": delta["h2d_payload_bytes_per_step"] > 0
            and force["reread_rows"] > 0,
            "reread_delta_exceeds_noise": noise is not None
            and delta["reread_ms"] > noise,
            "similar_commit_work": all(
                close(k) for k in ("admissions", "eligible_hits", "native_h2d_rows")
            ),
            "similar_commit_time": close("commit_ms"),
            "similar_acceptance": close("tokens_per_request_step"),
            "step_tracks_reread": delta["step_ms"] > 0
            and abs(residual) <= max(0.05 * base["step_ms"], 0.20 * delta["reread_ms"]),
        }
        ranks[rank] = {
            "delta": delta,
            "step_minus_reread_delta_ms": residual,
            "two_standard_errors_ms": noise,
            "checks": checks,
        }
    return {
        "passed": all(all(r["checks"].values()) for r in ranks.values()),
        "ranks": ranks,
    }
