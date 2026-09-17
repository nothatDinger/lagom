#!/usr/bin/env python3
"""Convert local LongBench datasets to the ShareGPT benchmark format."""

import argparse
import json
from pathlib import Path


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"LongBench dataset path does not exist: {path}")
    files = (
        sorted(path.rglob("*.jsonl"))
        + sorted(path.rglob("*.parquet"))
        + sorted(path.rglob("*.json"))
    )
    if not files:
        raise ValueError(f"no .jsonl, .parquet, or .json files found under {path}")
    return files


def read_rows(path: Path):
    if path.suffix == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from data if isinstance(data, list) else [data]
        return
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def format_row(row: dict, variant: str) -> tuple[str, str]:
    if variant == "longbench":
        prompt = f"{row['context']}\n\n{row['input']}"
        answers = row.get("answers") or []
        return prompt, str(answers[0]) if answers else "No reference answer"
    prompt = (
        f"{row['context']}\n\nQuestion: {row['question']}\n"
        f"A. {row['choice_A']}\nB. {row['choice_B']}\n"
        f"C. {row['choice_C']}\nD. {row['choice_D']}\nAnswer:"
    )
    return prompt, str(row.get("answer", "No reference answer"))


def prepare(input_path: Path, output_path: Path, variant: str, count: int) -> None:
    selected = []
    for path in input_files(input_path):
        for row in read_rows(path):
            prompt, answer = format_row(row, variant)
            selected.append(
                {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": answer},
                    ]
                }
            )
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"dataset has only {len(selected)} rows, need {count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("longbench", "longbench_v2"), required=True
    )
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    prepare(args.input, args.output, args.variant, args.count)


if __name__ == "__main__":
    main()
