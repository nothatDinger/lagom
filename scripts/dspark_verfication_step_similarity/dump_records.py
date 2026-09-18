#!/usr/bin/env python3
"""Wait for a server, configure DSpark, and dump its debug records."""

import argparse
import json
import time
from pathlib import Path

import requests


def wait_until_ready(base_url, restart_delay):
    while True:
        try:
            if requests.get(f"{base_url}/health", timeout=10).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    if restart_delay:
        time.sleep(restart_delay)


def enable_full_verify_budget(base_url):
    response = requests.post(
        f"{base_url}/set_internal_state",
        json={"server_args": {"dspark_force_budget_frac": 1.0}},
        timeout=30,
    )
    response.raise_for_status()


def dump_records(base_url, output, sampling):
    response = requests.get(f"{base_url}/server_info", timeout=120)
    response.raise_for_status()
    records = []
    for rank in response.json()["internal_states"]:
        dump = rank.get("dspark_info_record", {})
        records.extend(dump.get("records", dump if isinstance(dump, list) else []))
    Path(output).write_text(json.dumps({"sampling": sampling, "records": records}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "dump"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--restart-delay", type=float, default=0)
    parser.add_argument("--output")
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--input-len", type=int, default=0)
    parser.add_argument("--output-len", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        wait_until_ready(args.base_url, args.restart_delay)
        enable_full_verify_budget(args.base_url)
        return
    if not args.output:
        parser.error("dump requires --output")
    dump_records(
        args.base_url,
        args.output,
        {
            "dataset": args.dataset_name,
            "count": args.count,
            "random_input_len": args.input_len,
            "random_output_len": args.output_len,
        },
    )


if __name__ == "__main__":
    main()
