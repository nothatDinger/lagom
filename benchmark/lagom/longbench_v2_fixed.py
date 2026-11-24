"""Exact-length LongBench-v2 inputs; serving/scheduling/metrics remain SGLang's."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from array import array
from pathlib import Path

FORMAT = "lagom-longbench-v2-token-ids-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer_signature(tokenizer):
    # Token IDs are sent directly, so the ID-to-token mapping must agree.
    return digest(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode())


def question_suffix(row):
    return (
        "\n\nQuestion: " + row["question"] + "\n"
        + "\n".join(f"{letter}. {row['choice_' + letter]}" for letter in "ABCD")
        + "\nAnswer:"
    )


def fixed_prompt(context_ids, suffix_ids, length, wrap):
    """Keep the context's head/tail and all question/options, with no padding."""
    context_budget = length - len(wrap(suffix_ids))
    if context_budget <= 0 or len(context_ids) < context_budget:
        return None
    head = (context_budget + 1) // 2
    tail = context_budget - head
    context = context_ids[:head] + (context_ids[-tail:] if tail else [])
    ids = wrap(context + suffix_ids)
    if len(ids) != length:
        raise ValueError("Tokenizer special-token wrapping is not length additive")
    return ids


def write_ids(path, ids):
    values = array("I", ids)
    if values.itemsize != 4:
        raise RuntimeError("This platform does not have 32-bit unsigned int arrays")
    if sys.byteorder != "little":
        values.byteswap()
    data = values.tobytes()
    path.write_bytes(data)
    return digest(data)


def read_ids(path, checksum, length):
    data = path.read_bytes()
    if len(data) != 4 * length or digest(data) != checksum:
        raise ValueError(f"Corrupt token bank: {path}")
    values = array("I")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def request_order(count, size, seed):
    if count < 1 or size < 1:
        raise ValueError("No eligible long samples (or invalid request count)")
    rng = random.Random(seed)
    order = []
    while len(order) < size:
        cycle = list(range(count))
        rng.shuffle(cycle)
        order.extend(cycle)
    return order[:size]


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Not a prepared LongBench-v2 manifest: {path}")
    if manifest["output_len"] != 1024:
        raise ValueError("The overall protocol requires 1024 output tokens")
    return manifest


def prepare(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if args.max_requests < 1 or any(n < 1 for n in args.input_lens):
        raise ValueError("Lengths and request count must be positive")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    signature = tokenizer_signature(tokenizer)
    if args.dataset_path:
        path = Path(args.dataset_path).resolve()
        loader = "parquet" if path.suffix == ".parquet" else "json"
        rows = load_dataset(loader, data_files=str(path), split="train")
        source = {"path": str(path), "fingerprint": rows._fingerprint}
    else:
        rows = load_dataset(
            "THUDM/LongBench-v2", split="train", revision=args.dataset_revision
        )
        source = {
            "dataset": "THUDM/LongBench-v2",
            "revision": args.dataset_revision,
            "fingerprint": rows._fingerprint,
        }
    rows = rows.shuffle(seed=args.seed)
    lengths = sorted(set(args.input_lens))
    banks = {length: [] for length in lengths}
    for length in lengths:
        (out / str(length)).mkdir()
    for index, row in enumerate(rows):
        context_ids = tokenizer.encode(row["context"], add_special_tokens=False)
        suffix_ids = tokenizer.encode(question_suffix(row), add_special_tokens=False)
        for length, bank in banks.items():
            if len(bank) == args.max_requests:
                continue
            ids = fixed_prompt(
                context_ids, suffix_ids, length, tokenizer.build_inputs_with_special_tokens
            )
            if ids is None:
                continue
            name = f"{len(bank):05d}.u32"
            checksum = write_ids(out / str(length) / name, ids)
            bank.append({
                "file": name, "sha256": checksum,
                "source_id": str(row.get("_id", row.get("id", index))),
                "context_tokens_before_crop": len(context_ids),
                "suffix_tokens": len(suffix_ids),
            })
        if (index + 1) % 25 == 0:
            print(f"Scanned {index + 1}: " + str({k: len(v) for k, v in banks.items()}), flush=True)
        if all(len(bank) == args.max_requests for bank in banks.values()):
            break
    for length, bank in banks.items():
        if not bank:
            raise ValueError(f"No genuine LongBench-v2 context long enough for {length}")
        manifest = {
            "format": FORMAT, "input_len": length, "output_len": 1024,
            "seed": args.seed, "tokenizer": args.tokenizer,
            "tokenizer_signature": signature, "source": source,
            "context_crop": "head_tail", "special_tokens": "build_inputs_with_special_tokens",
            "prompts": bank,
            "order": request_order(len(bank), args.max_requests, args.seed),
        }
        (out / str(length) / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"{length}: {len(bank)} distinct samples, {args.max_requests} ordered requests")


def fixed_dataset(args, tokenizer, model_id=None):
    from sglang.benchmark.datasets.common import DatasetRow

    manifest_path = Path(args.dataset_path).resolve()
    m = load_manifest(manifest_path)
    if args.backend != "sglang" or not args.tokenize_prompt:
        raise ValueError("Use --backend sglang --tokenize-prompt for exact input IDs")
    if args.disable_ignore_eos or args.disable_stream or args.temperature != 0:
        raise ValueError("Overall requires streaming, ignore_eos, temperature=0")
    if args.sharegpt_output_len != m["output_len"] or args.seed != m["seed"]:
        raise ValueError("Benchmark seed/output length differs from prepared data")
    if tokenizer_signature(tokenizer) != m["tokenizer_signature"]:
        raise ValueError("Tokenizer vocabulary differs from the prepared token bank")
    if not 0 < args.num_prompts <= len(m["order"]):
        raise ValueError("Prepare enough requests for the largest concurrency * 4")
    cache = {}
    result = []
    for index in m["order"][:args.num_prompts]:
        if index not in cache:
            item = m["prompts"][index]
            cache[index] = read_ids(
                manifest_path.parent / item["file"], item["sha256"], m["input_len"]
            )
        result.append(DatasetRow(
            prompt=cache[index], prompt_len=m["input_len"], output_len=m["output_len"]
        ))
    return result


def main():
    # Intercept only dataset construction in this process. Use upstream SGLang
    # concurrency, warmup, streaming client, timing and result serialization.
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        from sglang.benchmark import serving

        del sys.argv[1]
        serving.get_dataset = fixed_dataset
        serving.cli_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset-path", help="Local LongBench-v2 JSON/JSONL or parquet")
    parser.add_argument("--dataset-revision", help="HF dataset revision to pin")
    parser.add_argument("--output", required=True, help="New token-bank directory")
    parser.add_argument("--input-lens", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--max-requests", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
