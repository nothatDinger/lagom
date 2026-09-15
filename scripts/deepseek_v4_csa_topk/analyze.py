#!/usr/bin/env python3
"""Aggregate benchmark and intrusive H2D-probe output and plot the experiment."""

import argparse
import csv
import json
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
    args = parser.parse_args()
    root = Path(args.results_dir)
    summary = []
    curves = {}
    for k in KS:
        metric = last_json(root / f"k{k}" / "benchmark.jsonl")
        traces = [
            json.loads(x)
            for x in (root / f"k{k}" / "h2d_trace.tp0.jsonl").read_text().splitlines()
            if x.strip()
        ]
        # Sum per-layer stalls into a decode-step latency, then average steps.
        h2d = []
        for row in traces:
            if int(row["layer_id"]) == 0:
                h2d.append(0.0)
            if not h2d:
                raise ValueError("trace does not start with layer 0")
            h2d[-1] += float(row["h2d_ms"])
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
        for row in traces:
            if int(row["layer_id"]) != 0:
                continue
            counts = row["miss_tokens_per_request"]
            by_step.setdefault(int(row["decode_step"]), []).extend(counts)
        curves[k] = {step: sum(vals) / len(vals) for step, vals in by_step.items()}

    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with open(
        root / "h2d_tokens_by_step.csv", "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(["k", "decode_step", "mean_h2d_tokens_per_request"])
        for k, points in curves.items():
            writer.writerows((k, step, value) for step, value in sorted(points.items()))

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
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">Mean H2D transfer per request (tokens)</text>',
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
        + "\n\n![H2D tokens by decode step](h2d_tokens_by_decode_step.svg)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
