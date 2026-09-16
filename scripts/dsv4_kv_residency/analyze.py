#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    data = json.loads(Path(a.input).read_text())
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in data["records"]:
        for req in step.get("reqs") or []:
            conf = req.get("confidence")
            hits = req.get("kv_hit_counts")
            totals = req.get("kv_topk_counts")
            accepted = req.get("correct_drafts", 0)
            if not conf or not hits or accepted <= 0:
                continue
            baseline = conf[min(accepted - 1, len(conf) - 1)]
            for pos in range(accepted, min(len(conf), len(hits))):
                if totals[pos]:
                    rows.append(
                        {
                            "forward_ct": step["forward_ct"],
                            "rid": req.get("rid"),
                            "draft_position": pos,
                            "confidence": conf[pos],
                            "last_accepted_confidence": baseline,
                            "confidence_gap": baseline - conf[pos],
                            "kv_hits": hits[pos],
                            "kv_topk": totals[pos],
                            "hit_ratio": hits[pos] / totals[pos],
                        }
                    )
    with (out / "unaccepted_tokens.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0]) if rows else ["confidence_gap", "hit_ratio"]
        )
        w.writeheader()
        w.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(
        [r["confidence_gap"] for r in rows],
        [r["hit_ratio"] for r in rows],
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    ax.set(
        xlabel="confidence(last accepted) - confidence(unaccepted)",
        ylabel="CSA KV HBM hit ratio",
        title="DSpark rejected drafts: KV residency vs confidence gap",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "kv_hit_ratio_vs_confidence_gap.png", dpi=180)
    plt.close(fig)
    (out / "report.md").write_text(
        f"# DSv4 DSpark + HiSparse KV residency\n\nSamples: {data['sampling']['count']} ({data['sampling']['method']})  \nUnaccepted draft-token observations: {len(rows)}\n\nArtifacts: `unaccepted_tokens.csv`, `kv_hit_ratio_vs_confidence_gap.png`, `raw_records.json`.\n"
    )


if __name__ == "__main__":
    main()
