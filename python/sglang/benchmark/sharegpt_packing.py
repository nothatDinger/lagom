"""Fixed-length token streams for local ShareGPT benchmarks."""

import hashlib
import json
import random


def pack_sharegpt(dataset, encode, *, count, length, seed):
    """Deterministically pack distinct source dialogues; never pad with random IDs."""
    if count < 1 or length < 1:
        raise ValueError("ShareGPT prompt count and length must be positive")
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    rows, pending, sources = [], [], []
    for index in order:
        dialogue = dataset[index].get(
            "conversations", dataset[index].get("conversation", [])
        )
        text = "\n".join(
            f"{turn.get('from', turn.get('role', 'user'))}: {turn.get('value', turn.get('content', ''))}"
            for turn in dialogue
        )
        if not text:
            continue
        ids = encode(text + "\n\n")
        if not ids:
            continue
        pending.extend(ids)
        sources.append(index)
        while len(pending) >= length and len(rows) < count:
            tokens, pending = pending[:length], pending[length:]
            rows.append(
                {
                    "input_ids": tokens,
                    "source_rows": list(sources),
                    "token_sha256": hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                }
            )
            sources = [index] if pending else []
        if len(rows) == count:
            return rows
    raise ValueError(
        f"ShareGPT has insufficient tokens: need {count} packed prompts of {length} tokens"
    )
