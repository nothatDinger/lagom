"""Shared protocol definitions. Importing this module does not import SGLang/CUDA."""

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def load_config(path):
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema_version")
    for key in (
        "num_requests",
        "repeats",
        "warmup_requests",
        "output_length",
        "warmup_output_length",
        "context_length",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("input_lengths", "concurrencies"):
        values = config[key]
        if (
            not values
            or any(type(v) is not int or v < 1 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid {key}")
    if max(config["concurrencies"]) > 256:
        raise ValueError("Maximum concurrency for this protocol is 256")
    if config["num_requests"] < max(config["concurrencies"]):
        raise ValueError("num_requests must cover the largest concurrency")
    if (
        max(config["input_lengths"])
        + max(config["output_length"], config["warmup_output_length"])
        + 2
        > config["context_length"]
    ):
        raise ValueError("Context must hold input, output and lookahead workspace")
    if not config["methods"] or set(config["methods"]) - {"dense", "oasiskv"}:
        raise ValueError("Methods must be dense and/or oasiskv")
    for name, model in config["models"].items():
        if type(model["tp"]) is not int or model["tp"] < 1:
            raise ValueError(f"Invalid TP for {name}")
        devices = model["cuda_visible_devices"].split(",")
        if len(devices) != model["tp"] or len(set(devices)) != len(devices):
            raise ValueError(f"GPU list must have exactly TP distinct devices: {name}")
    return config


def matrix(config, models=None, methods=None, smoke=False):
    models = models or list(config["models"])
    methods = methods or config["methods"]
    if set(models) - config["models"].keys() or set(methods) - set(config["methods"]):
        raise ValueError("Unknown model or method filter")
    lengths = config["input_lengths"][:1] if smoke else config["input_lengths"]
    concurrencies = [1] if smoke else config["concurrencies"]
    for model, length, concurrency, repeat, method in itertools.product(
        models,
        lengths,
        concurrencies,
        range(1 if smoke else config["repeats"]),
        methods,
    ):
        yield {
            "model": model,
            "input_length": length,
            "concurrency": concurrency,
            "repeat": repeat,
            "method": method,
            "num_requests": 2 if smoke else config["num_requests"],
            "output_length": 32 if smoke else config["output_length"],
            "protocol": "smoke" if smoke else "fig11-sglang-port",
        }


def select_workload(config, concurrencies=None, input_lengths=None, *, smoke=False):
    """Filter configured points without changing request count or their meaning."""
    if smoke and (concurrencies is not None or input_lengths is not None):
        raise ValueError(
            "--smoke is fixed at concurrency=1, 2 requests, 32 output tokens; "
            "omit --smoke for a concurrency sweep"
        )
    selected = dict(config)
    for key, values in (
        ("concurrencies", concurrencies),
        ("input_lengths", input_lengths),
    ):
        if values is None:
            continue
        if (
            not values
            or len(set(values)) != len(values)
            or set(values) - set(config[key])
        ):
            raise ValueError(f"{key} must be distinct values from {config[key]}")
        selected[key] = [value for value in config[key] if value in values]
    return selected


def point_name(point):
    return "{protocol}-{model}-{method}-i{input_length}-o{output_length}-c{concurrency}-r{repeat}".format(
        **point
    )


def server_command(config, point, port, endpoint, draft_path=None):
    model = config["models"][point["model"]]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model["path"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp",
        str(model["tp"]),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--context-length",
        str(config["context_length"]),
        "--max-running-requests",
        str(point["concurrency"]),
        "--cuda-graph-config",
        json.dumps(
            {"decode": {"backend": "full", "max_bs": point["concurrency"]}},
            separators=(",", ":"),
        ),
        "--mem-fraction-static",
        str(config["mem_fraction_static"]),
        "--chunked-prefill-size",
        str(config["chunked_prefill_size"]),
        "--disable-radix-cache",
        "--random-seed",
        str(config["seed"]),
        "--stream-interval",
        "1",
        "--enable-forward-pass-metrics",
        "--forward-pass-metrics-ipc-name",
        endpoint,
    ]
    if point["method"] == "oasiskv":
        command += [
            "--attention-backend",
            "oasiskv",
            "--speculative-draft-model-path",
            draft_path or model["draft"],
            "--hisparse-config",
            json.dumps(config["oasiskv"], separators=(",", ":")),
        ]
    else:
        command += ["--attention-backend", "fa3"]
    return command


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    return values[low] + (values[math.ceil(position)] - values[low]) * (position - low)


def summarize_forward_metrics(fpm):
    """Scheduler decode requests, excluding idle heartbeats and prefill-only passes."""
    decode = [f for f in fpm if f["scheduled_requests"]["num_decode_requests"] > 0]
    batches = [f["scheduled_requests"]["num_decode_requests"] for f in decode]
    all_prefill = sum(
        f["scheduled_requests"].get("num_prefill_requests", 0) for f in fpm
    )
    gaps = sum(
        max(0, b["counter_id"] - a["counter_id"] - 1) for a, b in zip(fpm, fpm[1:])
    )
    return {
        "running_batch_mean": sum(batches) / len(batches) if batches else None,
        "running_batch_p50": percentile(batches, 0.5),
        "running_batch_max": max(batches) if batches else None,
        "decode_forward_samples": len(batches),
        "prefill_request_samples": all_prefill,
        "fpm_sequence_gaps": gaps,
    }


def summarize(requests, fpm, start, end):
    """Wall-clock throughput and request-weighted TPOT; never count draft tokens."""
    good = [r for r in requests if r["success"]]
    tpots = [
        1000 * (r["last_token_s"] - r["first_token_s"]) / (r["output_tokens"] - 1)
        for r in good
        if r["output_tokens"] > 1 and r["first_chunk_tokens"] == 1
    ]
    first = min((r["first_token_s"] for r in good), default=0)
    last = max((r["last_token_s"] for r in good), default=0)
    total = sum(r["output_tokens"] for r in good)
    return {
        "successful_requests": len(good),
        "failed_requests": len(requests) - len(good),
        "total_output_tokens": total,
        "duration_s": end - start,
        "output_tps": total / (end - start) if end > start else None,
        "decode_window_tps": (
            sum(max(0, r["output_tokens"] - 1) for r in good) / (last - first)
            if last > first
            else None
        ),
        "tpot_mean_ms": sum(tpots) / len(tpots) if tpots else None,
        "tpot_p50_ms": percentile(tpots, 0.5),
        "tpot_p95_ms": percentile(tpots, 0.95),
        "ttft_mean_ms": (
            sum(1000 * (r["first_token_s"] - r["start_s"]) for r in good) / len(good)
            if good
            else None
        ),
        **summarize_forward_metrics(fpm),
        "tpot_samples": len(tpots),
    }
