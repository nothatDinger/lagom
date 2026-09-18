#!/usr/bin/env python3
"""Aggregate benchmark and intrusive H2D-probe output and plot the experiment."""

import argparse
import csv
import json
import sys
from pathlib import Path

BATCH_SIZES = (1, 4, 8, 16, 32)


def last_json(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no JSON records in {path}")
    return rows[-1]


def aggregate_trace_cycles(traces: list[dict]) -> list[dict]:
    """Combine layer records and bind each commit to its trace occurrence.

    ``decode_step`` restarts when a request-pool slot is reused, so it is not a
    file-wide identifier.  Commit records must be associated in stream order
    rather than collapsed into a dictionary keyed only by that value.
    """
    cycles = []
    for row in traces:
        if row.get("event") == "commit":
            step = int(row["decode_step"])
            if not cycles or cycles[-1]["commit"] is not None:
                raise ValueError(
                    f"commit record for decode step {step} has no preceding H2D cycle"
                )
            cycle = cycles[-1]
            if cycle["decode_step"] != step:
                raise ValueError(
                    "commit/H2D decode-step mismatch: "
                    f"commit={step}, H2D={cycle['decode_step']}"
                )
            cycle["commit"] = row
            continue

        if int(row["layer_id"]) == 0:
            cycles.append(
                {
                    "decode_step": int(row["decode_step"]),
                    "h2d_ms": 0.0,
                    "misses": [0] * len(row["miss_tokens_per_request"]),
                    "layers": 0,
                    "commit": None,
                    "request_pool_indices": row.get("request_pool_indices"),
                }
            )
        if not cycles:
            raise ValueError("trace does not start with layer 0")
        cycle = cycles[-1]
        if cycle["commit"] is not None:
            raise ValueError("H2D layer record follows a commit without a new layer 0")
        cycle["h2d_ms"] += float(row["h2d_ms"])
        cycle["layers"] += 1
        counts = row["miss_tokens_per_request"]
        if len(counts) != len(cycle["misses"]):
            raise ValueError("request count changed within one layer cycle")
        cycle["misses"] = [
            total + int(count) for total, count in zip(cycle["misses"], counts)
        ]
    return cycles


def align_cycle_requests(cycle: dict) -> tuple[list[int], list[int]]:
    """Align H2D counters with commit values, tolerating legacy padded rows."""
    commit = cycle["commit"]
    misses = [int(value) for value in cycle["misses"]]
    accepted = [int(value) for value in commit["cumulative_accepted_tokens"]]
    h2d_ids = cycle.get("request_pool_indices")
    commit_ids = commit.get("request_pool_indices")

    if h2d_ids is not None and commit_ids is not None:
        if len(h2d_ids) != len(misses) or len(commit_ids) != len(accepted):
            raise ValueError("request identifiers do not match their trace values")
        misses_by_id = dict(zip(h2d_ids, misses))
        if len(misses_by_id) != len(h2d_ids) or set(misses_by_id) != set(commit_ids):
            raise ValueError(
                "request identities changed between H2D and commit records at "
                f"decode step {cycle['decode_step']}: "
                f"H2D={h2d_ids}, commit={commit_ids}"
            )
        return [misses_by_id[req_id] for req_id in commit_ids], accepted

    if len(misses) > len(accepted) and not any(misses[len(accepted) :]):
        # Older traces included zero counters for padded execution-bucket rows.
        # Live requests occupy the leading rows, and the planner guarantees
        # that padded rows cannot report a miss.
        misses = misses[: len(accepted)]
    if len(accepted) != len(misses):
        raise ValueError(
            "request count changed between H2D and commit records at "
            f"decode step {cycle['decode_step']}: "
            f"H2D={len(misses)}, commit={len(accepted)}. If this is an old "
            "trace, regenerate it with current instrumentation; nonzero "
            "unmatched rows cannot be assigned safely."
        )
    return misses, accepted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSV, SVG, and Markdown reports from a CSA H2D batch-size run."
    )
    parser.add_argument(
        "--results-dir",
        default="results/deepseek_v4_csa_topk_h2d_batchsize",
        help=(
            "run directory containing bs<BATCH_SIZE>/benchmark.jsonl and "
            "bs<BATCH_SIZE>/h2d_trace.tp0.jsonl (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=BATCH_SIZES,
        metavar="BATCH_SIZE",
        help="Request batch sizes to analyze (default: 1 4 8 16 32)",
    )
    args = parser.parse_args()
    root = Path(args.results_dir)
    summary = []
    curves = {}
    curves_per_layer = {}
    skipped = []
    diagnostics = []
    for batch_size in args.batch_sizes:
        benchmark_path = root / f"bs{batch_size}" / "benchmark.jsonl"
        trace_path = root / f"bs{batch_size}" / "h2d_trace.tp0.jsonl"
        missing = [
            str(path.relative_to(root))
            for path in (benchmark_path, trace_path)
            if not path.is_file()
        ]
        if missing:
            reason = f"missing {', '.join(missing)}"
            skipped.append((batch_size, reason))
            print(
                f"warning: skipping batch size {batch_size}: {reason}", file=sys.stderr
            )
            continue
        metric = last_json(benchmark_path)
        traces = [
            json.loads(x) for x in trace_path.read_text().splitlines() if x.strip()
        ]
        if not traces:
            reason = f"empty {trace_path.relative_to(root)}"
            skipped.append((batch_size, reason))
            print(
                f"warning: skipping batch size {batch_size}: {reason}", file=sys.stderr
            )
            continue
        # A layer-0 record starts one decode/verify invocation. Aggregate both
        # latency and physical token-entry copies across every C4 layer.
        cycles = aggregate_trace_cycles(traces)
        h2d = [cycle["h2d_ms"] for cycle in cycles]
        tpot = float(metric["mean_tpot_ms"])
        summary.append(
            {
                "batch_size": batch_size,
                "mean_h2d_ms": sum(h2d) / len(h2d),
                "mean_tpot_ms": tpot,
                "h2d_over_tpot": (sum(h2d) / len(h2d)) / tpot,
            }
        )
        by_accepted = {}
        by_accepted_per_layer = {}
        for cycle in cycles:
            commit = cycle["commit"]
            if commit is None:
                continue
            misses, accepted = align_cycle_requests(cycle)
            for accepted_tokens, miss_count in zip(accepted, misses):
                by_accepted.setdefault(int(accepted_tokens), []).append(miss_count)
                by_accepted_per_layer.setdefault(int(accepted_tokens), []).append(
                    miss_count / cycle["layers"]
                )
        curves[batch_size] = {
            accepted: sum(vals) / len(vals) for accepted, vals in by_accepted.items()
        }
        curves_per_layer[batch_size] = {
            accepted: sum(vals) / len(vals)
            for accepted, vals in by_accepted_per_layer.items()
        }

        first = traces[0]
        buffer_size = int(
            first.get(
                "device_buffer_size",
                first["verify_width"] * int(first.get("top_k", 512)),
            )
        )
        max_tokens = max(
            (
                int(inp) + int(out)
                for inp, out in zip(
                    metric.get("input_lens", []), metric.get("output_lens", [])
                )
            ),
            default=0,
        )
        if max_tokens and (max_tokens + 3) // 4 <= buffer_size:
            diagnostics.append(
                f"batch size {batch_size}: all measured requests fit the C4 device buffer "
                f"(max sequence ~= {(max_tokens + 3) // 4} C4 entries, "
                f"buffer={buffer_size}); zero H2D misses are expected."
            )

    if not summary:
        raise SystemExit(
            "no complete batch-size groups found; each batch-size group needs benchmark.jsonl and h2d_trace.tp0.jsonl"
        )

    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with open(
        root / "h2d_tokens_by_accepted_tokens.csv", "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "batch_size",
                "cumulative_accepted_tokens",
                "mean_h2d_token_layer_entries_per_request",
                "mean_h2d_tokens_per_request_per_layer",
            ]
        )
        for batch_size, points in curves.items():
            writer.writerows(
                (batch_size, accepted, value, curves_per_layer[batch_size][accepted])
                for accepted, value in sorted(points.items())
            )

    width, height, pad = 900, 520, 60
    max_x = max((max(points, default=0) for points in curves.values()), default=1) or 1
    max_y = (
        max((max(points.values(), default=0) for points in curves.values()), default=1)
        or 1
    )
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<path d="M {pad} {pad} V {height - pad} H {width - pad}" fill="none" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle">Cumulative accepted tokens in request</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">Mean H2D transfer per request (token-layer entries)</text>',
    ]
    for color, (batch_size, points) in zip(colors, curves.items()):
        coords = " ".join(
            f"{pad + accepted / max_x * (width - 2 * pad):.1f},{height - pad - value / max_y * (height - 2 * pad):.1f}"
            for accepted, value in sorted(points.items())
        )
        svg.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{width - pad - 80}" y="{pad + 20 * list(curves).index(batch_size)}" fill="{color}">Batch size={batch_size}</text>'
        )
    svg.append("</svg>")
    (root / "h2d_tokens_by_accepted_tokens.svg").write_text(
        "\n".join(svg), encoding="utf-8"
    )

    table = [
        "| Batch size | Mean H2D (ms) | Mean TPOT (ms) | H2D / TPOT |",
        "|---:|---:|---:|---:|",
    ]
    table += [
        f"| {r['batch_size']} | {r['mean_h2d_ms']:.4f} | {r['mean_tpot_ms']:.4f} | {r['h2d_over_tpot']:.4f} |"
        for r in summary
    ]
    (root / "REPORT.md").write_text(
        "# DeepSeek V4 CSA H2D batch-size experiment\n\n"
        + "\n".join(table)
        + (
            "\n\n## Incomplete groups\n\n"
            + "\n".join(
                f"- Batch size {batch_size}: {reason}" for batch_size, reason in skipped
            )
            if skipped
            else ""
        )
        + (
            "\n\n## Sampling diagnostics\n\n"
            + "\n".join(f"- {message}" for message in diagnostics)
            if diagnostics
            else ""
        )
        + "\n\n![H2D tokens by accepted tokens](h2d_tokens_by_accepted_tokens.svg)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
