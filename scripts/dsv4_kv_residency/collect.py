#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path


DATASET_NAMES = ("sharegpt", "longbench", "longbench-v2")


def _dataset_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    files = []
    for suffix in ("*.jsonl", "*.json", "*.parquet"):
        files.extend(path.rglob(suffix))
    if not files:
        raise ValueError(f"No JSON, JSONL, or Parquet dataset files found under {path}")
    return sorted(set(files))


def _rows(path):
    for file in _dataset_files(path):
        if file.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet datasets require pandas and pyarrow"
                ) from exc
            yield from pd.read_parquet(file).to_dict(orient="records")
            continue
        if file.suffix == ".jsonl":
            with file.open() as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, dict):
                            yield row
            continue
        data = json.loads(file.read_text())
        if isinstance(data, dict):
            data = data.get("data", data.get("train", []))
        if isinstance(data, list):
            yield from (row for row in data if isinstance(row, dict))


def _longbench_v2_prompt(row):
    context = row.get("context", "")
    question = row.get("question", row.get("input", ""))
    choices = row.get("choices")
    if choices is None:
        choices = [
            row.get(letter, row.get(f"choice_{letter}", "")) for letter in "ABCD"
        ]
    if not context or not question:
        return ""
    rendered_choices = "\n".join(
        f"({letter}) {choice}" for letter, choice in zip("ABCD", choices)
    )
    return (
        "Please read the following text and answer the question below.\n"
        f"<text>\n{context}\n</text>\n\n"
        f"Question: {question}\nChoices:\n{rendered_choices}\n\n"
        'Format your response as: "The correct answer is (insert answer here)".'
    )


def prompts(path, dataset_name="sharegpt"):
    for row in _rows(path):
        if dataset_name == "longbench-v2":
            text = _longbench_v2_prompt(row)
        elif dataset_name == "longbench":
            context = row.get("context", "")
            question = row.get("input", row.get("question", ""))
            text = row.get("prompt", "") or (
                f"{context}\n\nQuestion: {question}\nAnswer:"
                if context and question
                else ""
            )
        else:
            conv = row.get("conversations", [])
            text = next(
                (
                    x.get("value", "")
                    for x in conv
                    if x.get("from") in ("human", "user")
                ),
                "",
            )
        if text:
            yield text


def main():
    import requests

    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-name", choices=DATASET_NAMES, default="sharegpt")
    p.add_argument("--sample-count", type=int, default=100)
    p.add_argument("--sample-method", choices=("first", "random"), default="first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    health = a.base_url + "/health"
    while True:
        try:
            if requests.get(health).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    # A full budget keeps DSpark's normal confidence ordering but asks the
    # target to evaluate every proposal, which is required for counterfactual
    # CSA residency of proposals that ultimately fail acceptance.
    budget = requests.post(
        a.base_url + "/set_internal_state",
        json={"server_args": {"dspark_force_budget_frac": 1.0}},
    )
    budget.raise_for_status()
    items = list(prompts(a.dataset, a.dataset_name))
    if not items:
        raise ValueError(
            f"No valid {a.dataset_name} prompts found in dataset path {a.dataset}"
        )
    random.Random(a.seed).shuffle(items) if a.sample_method == "random" else None
    chosen = items[: a.sample_count]
    for i, prompt in enumerate(chosen):
        r = requests.post(
            a.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": a.max_new_tokens,
                },
            },
        )
        r.raise_for_status()
        print(f"completed {i + 1}/{len(chosen)}", flush=True)
    state = requests.get(a.base_url + "/server_info")
    state.raise_for_status()
    internals = state.json()["internal_states"]
    records = []
    for rank in internals:
        dump = rank.get("dspark_info_record", {})
        records.extend(dump.get("records", dump if isinstance(dump, list) else []))
    Path(a.output).write_text(
        json.dumps(
            {
                "sampling": {
                    "dataset": a.dataset_name,
                    "method": a.sample_method,
                    "count": len(chosen),
                    "seed": a.seed,
                },
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
