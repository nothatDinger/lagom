"""Pack ShareGPT text into exact token-length inputs, without re-tokenization."""

import argparse
import random
from pathlib import Path

from common import HERE, digest, load_config, read_json, sha256, write_json


def conversation_text(record):
    if not isinstance(record, dict):
        return ""
    turns = record.get("conversations", record.get("conversation", []))
    if not isinstance(turns, list):
        return ""
    parts = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        content = turn.get("value", turn.get("content", ""))
        if isinstance(content, str) and content.strip():
            parts.append(f"{turn.get('from', turn.get('role', 'unknown'))}: {content}")
    return "\n\n".join(parts)


def packed_prompts(records, tokenizer, count, length, seed):
    """Disjoint slices of a shuffled token stream; cycle only after exhausting it."""
    order = [i for i, record in enumerate(records) if conversation_text(record)]
    if not order:
        raise ValueError("ShareGPT contains no usable text")
    rng = random.Random(seed)
    remaining = []
    position = 0
    document = 0
    for _ in range(count):
        prompt = []
        while len(prompt) < length:
            if position == len(remaining):
                if document % len(order) == 0:
                    rng.shuffle(order)
                remaining = tokenizer.encode(
                    conversation_text(records[order[document % len(order)]]) + "\n\n",
                    add_special_tokens=False,
                )
                position = 0
                document += 1
                if not remaining:
                    raise ValueError("Tokenizer produced an empty document")
            take = min(length - len(prompt), len(remaining) - position)
            prompt.extend(remaining[position : position + take])
            position += take
        yield prompt


def tokenizer_fingerprint(path):
    path = Path(path)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    )
    files = {name: sha256(path / name) for name in names if (path / name).is_file()}
    if not files:
        raise ValueError(f"No local tokenizer files found in {path}")
    return files


def prepare(config, destination, models):
    import numpy as np
    import transformers
    from transformers import AutoTokenizer

    source = Path(config["dataset"])
    source_hash = sha256(source)
    records = read_json(source)
    if not isinstance(records, list):
        raise ValueError("Expected a ShareGPT JSON array")
    count = config["num_requests"] + config["warmup_requests"]
    length = max(config["input_lengths"])
    for model in models:
        path = config["models"][model]["path"]
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        manifest = {
            "schema_version": 1,
            "model": model,
            "model_path": path,
            "dataset_path": str(source),
            "dataset_sha256": source_hash,
            "tokenizer_files": tokenizer_fingerprint(path),
            "transformers_version": transformers.__version__,
            "seed": config["seed"],
            "count": count,
            "max_input_length": length,
            "warmup_requests": config["warmup_requests"],
            "construction": "shuffled-all-turns-token-stream-v1; short inputs are row prefixes; no chat template or BOS",
        }
        folder = Path(destination) / model
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "manifest.json").exists() or (folder / "prompts.npy").exists():
            raise FileExistsError(
                f"Prepared data already exists: {folder}; use a new directory"
            )
        temporary = folder / "prompts.npy.tmp"
        array = np.lib.format.open_memmap(
            temporary, mode="w+", dtype="int32", shape=(count, length)
        )
        for i, prompt in enumerate(
            packed_prompts(records, tokenizer, count, length, config["seed"])
        ):
            array[i] = prompt
            if (i + 1) % 128 == 0:
                print(f"{model}: {i + 1}/{count} prompts", flush=True)
        array.flush()
        del array
        temporary.replace(folder / "prompts.npy")
        manifest["prompts_sha256"] = sha256(folder / "prompts.npy")
        manifest["fingerprint"] = digest(manifest)
        write_json(folder / "manifest.json", manifest)
        print(f"Prepared {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    config = load_config(args.config)
    models = args.models or list(config["models"])
    if set(models) - config["models"].keys():
        parser.error("Unknown model")
    prepare(config, args.output, models)


if __name__ == "__main__":
    main()
