#!/usr/bin/env python3
"""Analyze CSA Top-K set overlap inside DSpark verification steps."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PAIR_FIELDS = (
    "forward_ct",
    "rid",
    "draft_token_left",
    "draft_token_right",
    "left_position",
    "right_position",
    "distance",
    "intersection",
    "union",
    "jaccard",
)


def similarity_rows(data):
    """Return all unordered token pairs and their adjacent-pair subset."""
    pairs = []
    for step in data.get("records", []):
        for req in step.get("reqs") or []:
            intersections = req.get("csa_topk_intersections")
            unions = req.get("csa_topk_unions")
            if intersections is None or unions is None:
                continue
            tokens = req.get("draft_tokens") or []
            width = min(int(req.get("verify_len", len(tokens))), len(tokens))
            width = min(width, len(intersections), len(unions))
            for left in range(width):
                width_right = min(width, len(intersections[left]), len(unions[left]))
                for right in range(left + 1, width_right):
                    intersection = int(intersections[left][right])
                    union = int(unions[left][right])
                    if union <= 0:
                        continue
                    pairs.append(
                        {
                            "forward_ct": step["forward_ct"],
                            "rid": req.get("rid"),
                            "draft_token_left": tokens[left],
                            "draft_token_right": tokens[right],
                            "left_position": left,
                            "right_position": right,
                            "distance": right - left,
                            "intersection": intersection,
                            "union": union,
                            "jaccard": intersection / union,
                        }
                    )
    return pairs, [row for row in pairs if row["distance"] == 1]


def _write_csv(path, rows, fields=PAIR_FIELDS):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pairs, adjacent = similarity_rows(data)
    _write_csv(output / "all_token_pairs.csv", pairs)
    _write_csv(output / "adjacent_draft_token_pairs.csv", adjacent)

    by_distance = defaultdict(lambda: [0, 0, 0])
    for row in pairs:
        aggregate = by_distance[row["distance"]]
        aggregate[0] += 1
        aggregate[1] += row["intersection"]
        aggregate[2] += row["union"]
    distance_rows = [
        {
            "distance": distance,
            "pair_count": count,
            "intersection": intersection,
            "union": union,
            "weighted_jaccard": intersection / union,
        }
        for distance, (count, intersection, union) in sorted(by_distance.items())
        if union
    ]
    _write_csv(
        output / "similarity_by_distance.csv",
        distance_rows,
        ("distance", "pair_count", "intersection", "union", "weighted_jaccard"),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [row["distance"] for row in distance_rows],
        [row["weighted_jaccard"] for row in distance_rows],
        marker="o",
    )
    axis.set(
        xlabel="Draft-token position distance",
        ylabel="CSA Top-K Jaccard overlap (intersection / union)",
        title="DSpark verification-step CSA Top-K similarity",
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "similarity_by_distance.png", dpi=180)
    plt.close(figure)

    adjacent_intersection = sum(row["intersection"] for row in adjacent)
    adjacent_union = sum(row["union"] for row in adjacent)
    adjacent_jaccard = (
        adjacent_intersection / adjacent_union if adjacent_union else float("nan")
    )
    output.joinpath("report.md").write_text(
        "# DSpark verification-step CSA Top-K similarity\n\n"
        f"Samples: {data.get('sampling', {}).get('count', 'unknown')}  \n"
        f"All within-step token pairs: {len(pairs)}  \n"
        f"Adjacent draft-token pairs: {len(adjacent)}  \n"
        f"Adjacent weighted Jaccard (intersection / union): {adjacent_jaccard:.6f}\n\n"
        "The counters are summed over all observed CSA layers before division. "
        "See `all_token_pairs.csv`, `adjacent_draft_token_pairs.csv`, "
        "`similarity_by_distance.csv`, and `similarity_by_distance.png`.\n"
    )


if __name__ == "__main__":
    main()
