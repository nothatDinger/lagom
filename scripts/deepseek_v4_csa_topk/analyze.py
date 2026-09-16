#!/usr/bin/env python3
"""Aggregate benchmark and intrusive H2D-probe output and plot the experiment."""

import argparse
import csv
import json
import sys
from pathlib import Path

KS = (512, 1024, 2048, 4096)


def last_json(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no JSON records in {path}")
    return rows[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/deepseek_v4_csa_topk")
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    args = parser.parse_args()
    root = Path(args.results_dir)
    summary = []
    curves = {}
    curves_per_layer = {}
    skipped = []
    diagnostics = []
    for k in args.ks:
        benchmark_path = root / f"k{k}" / "benchmark.jsonl"
        trace_path = root / f"k{k}" / "h2d_trace.tp0.jsonl"
        missing = [
            str(path.relative_to(root))
            for path in (benchmark_path, trace_path)
            if not path.is_file()
        ]
        if missing:
            reason = f"missing {', '.join(missing)}"
            skipped.append((k, reason))
            print(f"warning: skipping K={k}: {reason}", file=sys.stderr)
            continue
        metric = last_json(benchmark_path)
        traces = [
            json.loads(x) for x in trace_path.read_text().splitlines() if x.strip()
        ]
        if not traces:
            reason = f"empty {trace_path.relative_to(root)}"
            skipped.append((k, reason))
            print(f"warning: skipping K={k}: {reason}", file=sys.stderr)
            continue
        # A layer-0 record starts one decode/verify invocation. Aggregate both
        # latency and physical token-entry copies across every C4 layer.
        cycles = []
        for row in traces:
            if int(row["layer_id"]) == 0:
                cycles.append(
                    {
                        "decode_step": int(row["decode_step"]),
                        "h2d_ms": 0.0,
                        "misses": [0] * len(row["miss_tokens_per_request"]),
                        "layers": 0,
                    }
                )
            if not cycles:
                raise ValueError("trace does not start with layer 0")
            cycles[-1]["h2d_ms"] += float(row["h2d_ms"])
            cycles[-1]["layers"] += 1
            counts = row["miss_tokens_per_request"]
            if len(counts) != len(cycles[-1]["misses"]):
                raise ValueError("request count changed within one layer cycle")
            cycles[-1]["misses"] = [
                total + int(count) for total, count in zip(cycles[-1]["misses"], counts)
            ]
        h2d = [cycle["h2d_ms"] for cycle in cycles]
        tpot = float(metric["mean_tpot_ms"])
        summary.append(
            {
                "k": k,
                "mean_h2d_ms": sum(h2d) / len(h2d),
                "mean_tpot_ms": tpot,
                "h2d_over_tpot": (sum(h2d) / len(h2d)) / tpot,
            }
        )
        by_step = {}
        by_step_per_layer = {}
        for cycle in cycles:
            step = cycle["decode_step"]
            by_step.setdefault(step, []).extend(cycle["misses"])
            by_step_per_layer.setdefault(step, []).extend(
                count / cycle["layers"] for count in cycle["misses"]
            )
        curves[k] = {step: sum(vals) / len(vals) for step, vals in by_step.items()}
        curves_per_layer[k] = {
            step: sum(vals) / len(vals) for step, vals in by_step_per_layer.items()
        }

        first = traces[0]
        buffer_size = int(first.get("device_buffer_size", first["verify_width"] * k))
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
                f"K={k}: all measured requests fit the C4 device buffer "
                f"(max sequence ~= {(max_tokens + 3) // 4} C4 entries, "
                f"buffer={buffer_size}); zero H2D misses are expected."
            )

    if not summary:
        raise SystemExit(
            "no complete K groups found; each group needs benchmark.jsonl and h2d_trace.tp0.jsonl"
        )

    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with open(
        root / "h2d_tokens_by_step.csv", "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "k",
                "decode_step",
                "mean_h2d_token_layer_entries_per_request",
                "mean_h2d_tokens_per_request_per_layer",
            ]
        )
        for k, points in curves.items():
            writer.writerows(
                (k, step, value, curves_per_layer[k][step])
                for step, value in sorted(points.items())
            )

    width, height, pad = 900, 520, 60
    max_x = max((max(points, default=0) for points in curves.values()), default=1) or 1
    max_y = (
        max((max(points.values(), default=0) for points in curves.values()), default=1)
        or 1
    )
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<path d="M {pad} {pad} V {height-pad} H {width-pad}" fill="none" stroke="black"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle">Decode step</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">Mean H2D transfer per request (token-layer entries)</text>',
    ]
    for color, (k, points) in zip(colors, curves.items()):
        coords = " ".join(
            f"{pad + step/max_x*(width-2*pad):.1f},{height-pad-value/max_y*(height-2*pad):.1f}"
            for step, value in sorted(points.items())
        )
        svg.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{width-pad-80}" y="{pad+20*list(curves).index(k)}" fill="{color}">K={k}</text>'
        )
    svg.append("</svg>")
    (root / "h2d_tokens_by_decode_step.svg").write_text(
        "\n".join(svg), encoding="utf-8"
    )

    table = [
        "| K | Mean H2D (ms) | Mean TPOT (ms) | H2D / TPOT |",
        "|---:|---:|---:|---:|",
    ]
    table += [
        f'| {r["k"]} | {r["mean_h2d_ms"]:.4f} | {r["mean_tpot_ms"]:.4f} | {r["h2d_over_tpot"]:.4f} |'
        for r in summary
    ]
    (root / "REPORT.md").write_text(
        "# DeepSeek V4 CSA Top-K experiment\n\n"
        + "\n".join(table)
        + (
            "\n\n## Incomplete groups\n\n"
            + "\n".join(f"- K={k}: {reason}" for k, reason in skipped)
            if skipped
            else ""
        )
        + (
            "\n\n## Sampling diagnostics\n\n"
            + "\n".join(f"- {message}" for message in diagnostics)
            if diagnostics
            else ""
        )
        + "\n\n![H2D tokens by decode step](h2d_tokens_by_decode_step.svg)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
