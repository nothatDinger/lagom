#!/usr/bin/env python3
"""Short, reproducible ShareGPT-derived C=1 comparison. See README.md."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODES = ("lagom", "hisparse", "dspark")
ROOT = Path(__file__).resolve().parents[2]


def validate(args):
    if (
        args.w <= 0
        or args.p <= 0
        or args.requests <= 0
        or args.input_len <= 0
        or args.output_len <= 0
    ):
        raise ValueError("window, lengths and request count must be positive")
    if args.top_k <= 0 or (args.w + args.p + 1) * args.top_k > min(args.hot, 8192):
        raise ValueError(
            "need (W+P+1)*K <= min(hot,8192); increase --hot or reduce W/P"
        )
    if not 1 < args.hot <= 32768 or args.hot % 64:
        raise ValueError("--hot must be a multiple of 64 in [64,32768]")
    if not 0 < args.trace_steps <= 256 or args.trace_skip < 0:
        raise ValueError("trace steps must be 1..256, skip >=0")
    if args.tp <= 0 or args.warmup_output <= 0:
        raise ValueError("TP and warmup output must be positive")
    args.trace_output = max(
        args.trace_output, (args.trace_skip + args.trace_steps + 4) * (args.w + 1) + 1
    )
    if any(
        x <= 0 for x in (args.startup_timeout, args.request_timeout, args.mode_budget)
    ):
        raise ValueError("timeouts must be positive")


def server_command(args, mode, port):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--speculative-draft-model-path",
        args.draft,
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-num-draft-tokens",
        str(args.w + 1),
        "--speculative-dspark-block-size",
        str(args.w),
        "--tp-size",
        str(args.tp),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-running-requests",
        "1",
        "--disable-radix-cache",
        "--context-length",
        str(
            max(args.input_len + args.output_len, args.input_len + args.trace_output)
            + args.w
            + args.p
            + 128
        ),
        "--chunked-prefill-size",
        str(args.chunked_prefill),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--cuda-graph-config",
        '{"decode":{"bs":[1]}}',
        "--random-seed",
        str(args.seed),
    ]
    if mode != "dspark":
        config = {
            "host_to_device_ratio": 2,
            "device_buffer_size": args.hot,
            "force_h2d_each_step": False,
        }
        if mode == "lagom":
            config.update(
                prefetcher="lagom", prefetcher_config={"prediction_steps": args.p}
            )
        command += ["--enable-hisparse", "--hisparse-config", json.dumps(config)]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.server_args_json:
        extra = json.loads(Path(args.server_args_json).read_text())
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            raise ValueError("server-args-json must contain a JSON string array")
        owned = {x for x in command if x.startswith("--")} | {
            "--enable-hisparse",
            "--hisparse-config",
            "--disable-cuda-graph",
            "--disaggregation-mode",
        }
        if any(x.split("=", 1)[0] in owned for x in extra):
            raise ValueError(
                "extra arguments cannot override experiment-controlled options"
            )
        command += extra
    return command


def pack_sharegpt(dataset, encode, *, count, length, seed):
    """Deterministically pack distinct source dialogues; never pad with random IDs."""
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


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    text_config = getattr(config, "text_config", config)
    actual_k = (
        text_config.get("index_topk")
        if isinstance(text_config, dict)
        else getattr(text_config, "index_topk", None)
    )
    if actual_k != args.top_k:
        raise ValueError(
            f"model index_topk={actual_k}, expected --top-k={args.top_k}; correct the capacity settings"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    source = Path(args.sharegpt)
    with source.open() as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise TypeError("ShareGPT must be a JSON array of conversation records")
    rows = pack_sharegpt(
        dataset,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        count=args.requests + 1,
        length=args.input_len,
        seed=args.seed,
    )
    (args.output / "prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return rows


def generate(url, tokens, output_len, timeout):
    payload = {
        "input_ids": tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start, first, first_count, latest = time.perf_counter(), None, 0, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if time.perf_counter() - start > timeout:
                raise TimeoutError("generation exceeded request wall-clock budget")
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            if "error" in item:
                raise RuntimeError(item["error"])
            meta = item.get("meta_info", {})
            if meta.get("completion_tokens", 0) > 0 and first is None:
                first, first_count = time.perf_counter(), meta["completion_tokens"]
            latest = item
    elapsed = time.perf_counter() - start
    if first is None or latest is None:
        raise RuntimeError("server returned no token-bearing SSE response")
    meta = latest["meta_info"]
    if (
        meta.get("prompt_tokens") != len(tokens)
        or meta.get("completion_tokens") != output_len
    ):
        raise RuntimeError(
            f"wrong token counts: wanted {len(tokens)}/{output_len}, got {meta}"
        )
    ttft = first - start
    return {
        "input_tokens": len(tokens),
        "output_tokens": output_len,
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "decode_s": elapsed - ttft,
        "decode_tokens": output_len - first_count,
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
        "output_sha256": hashlib.sha256(latest.get("text", "").encode()).hexdigest(),
        "meta_info": meta,
    }


def aggregate(mode, requests):
    elapsed = sum(x["elapsed_s"] for x in requests)
    decode = sum(x["decode_s"] for x in requests)
    decode_tokens = sum(x["decode_tokens"] for x in requests)
    return {
        "mode": mode,
        "requests": len(requests),
        "concurrency": 1,
        "input_tokens": sum(x["input_tokens"] for x in requests),
        "output_tokens": sum(x["output_tokens"] for x in requests),
        "output_tokens_per_s": sum(x["output_tokens"] for x in requests) / elapsed,
        "decode_tokens_per_s": decode_tokens / decode if decode else None,
        "mean_ttft_s": sum(x["ttft_s"] for x in requests) / len(requests),
        "mean_tpot_ms": 1000 * decode / decode_tokens if decode_tokens else None,
        "elapsed_s": elapsed,
    }


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process):
    # A failed launcher may leave TP workers alive in its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # Also reap orphan workers if their launcher exited before they did.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_mode(args, mode, stage, rows):
    folder = args.output / stage / mode
    folder.mkdir(parents=True)
    port = free_port()
    command = server_command(args, mode, port)
    env = os.environ.copy()
    env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    # Never leak diagnostic markers from a parent shell into measured throughput.
    env.pop("SGLANG_LAGOM_PERF_TRACE", None)
    env.pop("SGLANG_DSPARK_DEBUG_DUMP", None)
    env.pop("SGLANG_OASISKV_TRACE_CONFIG", None)
    if stage == "trace":
        config = folder / "trace_config.json"
        config.write_text(
            json.dumps(
                {
                    "output_dir": str(folder / "steps"),
                    "steps": args.trace_steps,
                    "skip": args.trace_skip,
                    "nsys": args.nsys,
                }
            )
        )
        env["SGLANG_LAGOM_PERF_TRACE"] = str(config)
        if args.nsys:
            command = [
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--cuda-graph-trace=node",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "--output=" + str(folder / "nsys"),
            ] + command
    (folder / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    k: env[k]
                    for k in ("SGLANG_RAGGED_VERIFY_MODE", "SGLANG_LAGOM_PERF_TRACE")
                    if k in env
                },
            },
            indent=2,
        )
    )
    url = f"http://127.0.0.1:{port}"
    print(f"[{stage}/{mode}] {shlex.join(command)}", flush=True)
    with (folder / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited; inspect {folder / 'server.log'}"
                    )
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup timed out")
                time.sleep(1)
            with urllib.request.urlopen(
                url + "/get_server_info", timeout=10
            ) as response:
                (folder / "server_info.json").write_bytes(response.read())
            started = time.monotonic()
            if stage == "throughput":
                generate(
                    url,
                    rows[0]["input_ids"],
                    args.warmup_output,
                    min(args.request_timeout, args.mode_budget),
                )
                results = []
                for index, row in enumerate(rows[1:]):
                    remaining = args.mode_budget - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("mode runtime budget exhausted")
                    result = generate(
                        url,
                        row["input_ids"],
                        args.output_len,
                        min(args.request_timeout, remaining),
                    )
                    result["prompt_sha256"] = row["token_sha256"]
                    results.append(result)
                    (folder / "requests.json").write_text(json.dumps(results, indent=2))
                    print(
                        f"  request {index + 1}: {result['elapsed_s']:.2f}s, accept={result['spec_accept_length']}",
                        flush=True,
                    )
                summary = aggregate(mode, results)
                (folder / "summary.json").write_text(json.dumps(summary, indent=2))
                return summary
            generate(
                url,
                rows[1]["input_ids"],
                args.trace_output,
                min(args.request_timeout, args.mode_budget),
            )
            # Extra short decode permits nonblocking export if the last sampled
            # side transfer completed after the final step of the trace request.
            remaining = args.mode_budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("trace runtime budget exhausted")
            generate(url, rows[0]["input_ids"], 8, min(args.request_timeout, remaining))
            traces = list((folder / "steps").glob("rank-*.json"))
            traces = [p for p in traces if not p.name.endswith(".trace.json")]
            if len(traces) != args.tp:
                raise RuntimeError(
                    f"expected {args.tp} rank traces, found {len(traces)}; inspect server.log"
                )
            for path in traces:
                data = json.loads(path.read_text())
                if len(data["records"]) != args.trace_steps or not all(
                    r["cuda_graph"] and r.get("draft_cuda_graph", False)
                    for r in data["records"]
                ):
                    raise RuntimeError(
                        f"trace incomplete or draft/target fell back to eager: {path}"
                    )
                required = {
                    "step",
                    "draft",
                    "draft.model",
                    "target_verify",
                    "target.attention_core",
                    "target.attention_block",
                    "target.mlp",
                }
                if mode == "lagom":
                    required.update(
                        {
                            "target.kv_plan",
                            "target.kv_refill",
                            "target.kv_prefetch",
                            "target.prefetch_wait",
                        }
                    )
                elif mode == "hisparse":
                    required.add("target.kv_refill_fused")
                for step in range(args.trace_steps):
                    found = {
                        r["name"]
                        for r in data["regions"]
                        if r["step"] == step and r["end_ns"] >= r["start_ns"] > 0
                    }
                    if required - found:
                        raise RuntimeError(
                            f"missing GPU markers in {path}, step {step}: {sorted(required - found)}"
                        )
        finally:
            stop(process)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument(
        "--sharegpt",
        required=True,
        help="ShareGPT JSON array; long inputs are packed from dialogues",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--w", type=int, default=3)
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--hot", type=int, default=4096)
    p.add_argument("--top-k", type=int, default=512)
    p.add_argument("--input-len", type=int, default=32000)
    p.add_argument("--output-len", type=int, default=1024)
    p.add_argument("--requests", type=int, default=2)
    p.add_argument("--warmup-output", type=int, default=32)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--trace-modes", nargs="+", choices=MODES, default=["lagom"])
    p.add_argument("--phase", choices=("throughput", "trace", "all"), default="all")
    p.add_argument("--trace-steps", type=int, default=12)
    p.add_argument("--trace-skip", type=int, default=2)
    p.add_argument("--trace-output", type=int, default=160)
    p.add_argument(
        "--nsys",
        action="store_true",
        help="also collect correlated CUDA/NVTX/CPU trace, bounded by cudaProfilerApi",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunked-prefill", type=int, default=4096)
    p.add_argument("--mem-fraction", type=float, default=0.8)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--request-timeout", type=float, default=180)
    p.add_argument(
        "--mode-budget",
        type=float,
        default=600,
        help="seconds after startup, not counting checkpoint load/capture",
    )
    p.add_argument("--server-args-json")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    validate(args)
    if (
        os.getenv("SGLANG_DEBUG_HISPARSE_SKIP_IO", "0").lower()
        not in ("0", "false", "")
        or float(os.getenv("SGLANG_SIMULATE_ACC_LEN", "0")) != 0
    ):
        raise ValueError("disable fake IO / simulated acceptance before benchmarking")
    if args.dry_run:
        for mode in args.modes:
            print(shlex.join(server_command(args, mode, 30000)))
        print(
            f"throughput: {args.requests} x {args.input_len}/{args.output_len}, C=1; trace: {args.trace_modes}, {args.trace_steps} steps"
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = vars(args).copy()
    manifest["output"] = str(args.output)
    manifest["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest["input_format"] = "packed_sharegpt_token_ids_no_chat_template"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = prepare(args)
    summaries = []
    if args.phase in ("all", "throughput"):
        for mode in args.modes:
            summaries.append(run_mode(args, mode, "throughput", rows))
        with (args.output / "throughput.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(json.dumps(summaries, indent=2))
    if args.phase in ("all", "trace"):
        for mode in args.trace_modes:
            run_mode(args, mode, "trace", rows)


if __name__ == "__main__":
    main()
