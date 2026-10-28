"""Export point/repeat summaries and a Fig.11-shaped plot without imputing failures."""

import argparse
import csv
import math
import statistics
from pathlib import Path

from common import read_json

METRICS = (
    "output_tps",
    "decode_window_tps",
    "running_batch_mean",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
)
KEYS = ("protocol", "model", "method", "input_length", "output_length", "concurrency")


def collect(root):
    rows = []
    for path in sorted(root.glob("*/result.json")):
        if ".attempt-" not in path.parent.name:
            rows.append(read_json(path))
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        groups.setdefault(key, []).append(row)
    aggregate = []
    for key, attempts in sorted(groups.items()):
        record = dict(zip(KEYS, key))
        valid = [r for r in attempts if r["status"] == "ok"]
        record.update(valid_repeats=len(valid), attempted_repeats=len(attempts))
        for metric in METRICS:
            values = [r[metric] for r in valid if r.get(metric) is not None]
            record[metric] = statistics.mean(values) if values else None
            record[metric + "_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        aggregate.append(record)
    return rows, aggregate


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(root, rows, tps_field):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read_json(root / "run.json")
    config, protocol = manifest["config"], manifest["protocol"]
    if protocol not in ("fig11-sglang-port", "fig11-sglang-bench"):
        raise ValueError("Smoke runs are startup checks, not Fig.11 measurements")
    if protocol == "fig11-sglang-bench" and tps_field == "decode_window_tps":
        raise ValueError(
            "Native bench does not export decode-window TPS; use output_tps"
        )
    rows = [r for r in rows if r["protocol"] == protocol]
    columns = sorted({(r["model"], r["input_length"]) for r in rows})
    metrics = [
        (
            tps_field,
            (
                "Output TPS (token/s)"
                if tps_field == "output_tps"
                else "Decode-window TPS (token/s)"
            ),
        ),
        ("running_batch_mean", "Observed decode batch"),
        ("tpot_mean_ms", "TPOT (ms)"),
    ]
    figure, axes = plt.subplots(
        3, len(columns), figsize=(4 * len(columns), 9), squeeze=False
    )
    xs = config["concurrencies"]
    for column, (model, length) in enumerate(columns):
        for row_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, column]
            for method in config["methods"]:
                selected = {
                    r["concurrency"]: r
                    for r in rows
                    if r["model"] == model
                    and r["input_length"] == length
                    and r["method"] == method
                }
                ys = [selected.get(c, {}).get(metric) for c in xs]
                ax.plot(
                    xs,
                    [y if y is not None else math.nan for y in ys],
                    marker="o",
                    label=method,
                )
                for c, y in zip(xs, ys):
                    std = selected.get(c, {}).get(metric + "_std")
                    if y is not None and std is not None:
                        ax.errorbar(c, y, yerr=std, capsize=3, color="gray")
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, [str(c) for c in xs])
            ax.grid(alpha=0.25)
            ax.set_ylabel(label)
            if row_index == 0:
                ax.set_title(
                    f"{model}\ninput={length}, output={config['output_length']}"
                )
            if row_index == 2:
                ax.set_xlabel("Maximum concurrent requests")
    axes[0, 0].legend()
    inputs = (
        "native bench ShareGPT repeat/truncate"
        if protocol == "fig11-sglang-bench"
        else "ShareGPT packed inputs"
    )
    figure.suptitle(
        f"OasisKV SGLang port — {inputs}; mean ± repeat sample SD\nMissing/failed points are gaps; see CSV for repeat counts. Not paper measurements.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf"):
        figure.savefig(root / f"fig11-{tps_field}.{extension}", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--tps-field", choices=["output_tps", "decode_window_tps"], default="output_tps"
    )
    args = parser.parse_args()
    rows, aggregates = collect(args.results)
    if not rows:
        parser.error("No result.json files; no measurements to report")
    save_csv(args.results / "points.csv", rows)
    save_csv(args.results / "summary.csv", aggregates)
    if args.plot:
        plot(args.results, aggregates, args.tps_field)
    print(
        f"{len(rows)} attempts, {sum(r['status'] == 'ok' for r in rows)} valid; results in {args.results}"
    )


if __name__ == "__main__":
    main()
