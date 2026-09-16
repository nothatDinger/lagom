#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path
import requests


def prompts(path):
    data = json.loads(Path(path).read_text())
    for row in data:
        conv = row.get("conversations", [])
        text = next(
            (x.get("value", "") for x in conv if x.get("from") in ("human", "user")), ""
        )
        if text:
            yield text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--sample-count", type=int, default=100)
    p.add_argument("--sample-method", choices=("first", "random"), default="first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--server-ready-timeout-seconds", type=float, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    health = a.base_url + "/health"
    ready_deadline = (
        None
        if a.server_ready_timeout_seconds <= 0
        else time.monotonic() + a.server_ready_timeout_seconds
    )
    while True:
        try:
            if requests.get(health).ok:
                break
        except requests.RequestException:
            pass
        if ready_deadline is not None and time.monotonic() >= ready_deadline:
            raise TimeoutError(
                f"server did not become ready within {a.server_ready_timeout_seconds}s"
            )
        time.sleep(10)
    # A full budget keeps DSpark's normal confidence ordering but asks the
    # target to evaluate every proposal, which is required for counterfactual
    # CSA residency of proposals that ultimately fail acceptance.
    budget = requests.post(
        a.base_url + "/set_internal_state",
        json={"server_args": {"dspark_force_budget_frac": 1.0}},
    )
    budget.raise_for_status()
    items = list(prompts(a.dataset))
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
