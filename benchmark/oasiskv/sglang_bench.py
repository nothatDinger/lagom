"""Native SGLang bench client for the OasisKV sweep (no CUDA imports here)."""

import asyncio
import json
import math
import os
import signal
import sys
from pathlib import Path

from common import ROOT, summarize_forward_metrics, write_json


def dataset_metadata(path, required):
    """Fail before model loading instead of letting native bench download a fallback."""
    with open(path) as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise TypeError("ShareGPT must be a JSON array")
    eligible = 0
    for record in data:
        turns = record.get("conversations", record.get("conversation", []))
        if len(turns) < 2:
            continue
        if any(not isinstance(turn.get("value"), str) for turn in turns[:2]):
            raise ValueError("ShareGPT first two turns must have string values")
        eligible += bool(turns[0]["value"].strip())
    if eligible < required:
        raise ValueError(f"Only {eligible} nonempty two-turn prompts; need {required}")
    return {
        "construction": "sglang-random-sharegpt-repeat-truncate-token-ids",
        "eligible_prompts": eligible,
        "warmup_reuses_measurement_prompts": True,
    }


def bench_command(config, point, port, output, *, warmup=False):
    model = config["models"][point["model"]]
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model["path"],
        "--tokenizer",
        model["path"],
        "--dataset-name",
        "random",
        "--dataset-path",
        config["dataset"],
        "--random-input-len",
        str(point["input_length"]),
        "--random-output-len",
        str(config["warmup_output_length"] if warmup else point["output_length"]),
        "--random-range-ratio",
        "1",
        "--tokenize-prompt",
        "--num-prompts",
        str(config["warmup_requests"] if warmup else point["num_requests"]),
        "--max-concurrency",
        str(1 if warmup else point["concurrency"]),
        "--request-rate",
        "inf",
        "--seed",
        str(config["seed"]),
        "--temperature",
        "0",
        # Warmup is a separate invocation so FPM and request timers share a phase.
        "--warmup-requests",
        "0",
        "--output-file",
        str(output),
        "--output-details",
        "--disable-tqdm",
        "--tag",
        point["protocol"] + ("-warmup" if warmup else ""),
    ]


def read_result(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected one native bench result; refuse appended/mixed runs")
    # Native bench emits Infinity for request_rate=inf. Preserve its raw JSONL;
    # only finite, audited metrics are copied into our strict result.json.
    return json.loads(lines[0])


def audit_result(raw, point):
    """Audit native output-details; do not substitute requested tokens for failures."""
    reasons = []
    count, length, output = (
        point["num_requests"],
        point["input_length"],
        point["output_length"],
    )
    expected = {
        "backend": "sglang",
        "dataset_name": "random",
        "random_input_len": length,
        "random_output_len": output,
        "random_range_ratio": 1,
        "max_concurrency": point["concurrency"],
        "completed": count,
        "total_input_tokens": count * length,
        "total_output_tokens": count * output,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            reasons.append(f"native {key}={raw.get(key)!r}, expected {value!r}")
    if raw.get("request_rate") != math.inf:
        reasons.append("native request_rate must be inf")
    for key, value in (("input_lens", length), ("output_lens", output), ("errors", "")):
        items = raw.get(key, [])
        if len(items) != count or any(item != value for item in items):
            reasons.append(f"native {key}: missing requests, errors or length mismatch")
    for key in ("generated_texts", "ttfts", "itls"):
        items = raw.get(key, [])
        if len(items) != count or not all(items):
            reasons.append(f"native {key}: missing streaming output/timing samples")
    accept = raw.get("accept_length")
    if accept is not None and (
        not isinstance(accept, (int, float)) or not math.isfinite(accept) or accept < 0
    ):
        reasons.append("invalid native accept_length")
    elif point["method"] == "oasiskv" and accept is not None and accept > 1.001:
        reasons.append("draft acceptance observed; lookahead must force rejection")
    mapping = {
        "output_tps": "output_throughput",
        "duration_s": "duration",
        "tpot_mean_ms": "mean_tpot_ms",
        "tpot_p50_ms": "median_tpot_ms",
        "tpot_p95_ms": "p95_tpot_ms",
        "ttft_mean_ms": "mean_ttft_ms",
    }
    metrics = {}
    for target, source in mapping.items():
        value = raw.get(source)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            reasons.append(f"invalid native metric {source}={value!r}")
            value = None
        metrics[target] = value
    if (
        metrics["output_tps"] is not None
        and metrics["duration_s"] is not None
        and not math.isclose(
            metrics["output_tps"] * metrics["duration_s"], count * output, rel_tol=1e-6
        )
    ):
        reasons.append("native throughput/duration/token count disagree")
    completed = raw.get("completed", 0)
    if not isinstance(completed, int) or not 0 <= completed <= count:
        completed = 0
    metrics.update(
        successful_requests=completed,
        failed_requests=count - completed,
        # Retain only finite numeric fields; native JSONL holds all raw values.
        total_output_tokens=sum(
            v for v in raw.get("output_lens", []) if isinstance(v, int) and v >= 0
        ),
        decode_window_tps=None,
        tpot_samples=sum(
            v > 1 for v in raw.get("output_lens", []) if isinstance(v, int)
        ),
        metric_source="sglang.benchmark.serving",
        stream_boundary="first-nonempty-text (native bench)",
        acceptance_telemetry=("unavailable" if accept is None else "server-average"),
    )
    return metrics, reasons


async def run_bench(config, point, folder, port, env, *, warmup=False):
    stem = "bench-warmup" if warmup else "bench"
    output = (folder / f"{stem}.jsonl").resolve()
    if output.exists():
        raise FileExistsError(output)
    command = bench_command(config, point, port, output, warmup=warmup)
    write_json(folder / f"{stem}-launch.json", {"argv": command})
    # Open once before requests; the child writes directly to this descriptor.
    with open(folder / f"{stem}.log", "w") as log:  # noqa: ASYNC230
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            env={**env, "SGLANG_IS_IN_CI": "0", "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = await asyncio.wait_for(process.wait(), config["point_timeout_s"])
            if code:
                raise RuntimeError(f"Native bench exited {code}; see {stem}.log")
        finally:
            if process.returncode is None:
                # Only terminate this bench's process group, never another GPU job.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
    return read_result(output)


async def measure(config, point, folder, port, env, collector):
    warmup = await run_bench(config, point, folder, port, env, warmup=True)
    _, reasons = audit_result(
        warmup,
        {
            **point,
            "num_requests": config["warmup_requests"],
            "output_length": config["warmup_output_length"],
            "concurrency": 1,
        },
    )
    if reasons:
        raise ValueError(f"Native warmup failed: {reasons}")
    await collector.wait_idle()
    collector.phase = "measurement"
    raw = await run_bench(config, point, folder, port, env)
    await collector.wait_idle()
    collector.phase = "done"
    metrics, reasons = audit_result(raw, point)
    metrics.update(summarize_forward_metrics(collector.rows))
    if (
        not metrics["decode_forward_samples"]
        or metrics["fpm_sequence_gaps"]
        or collector.errors
    ):
        reasons.append("missing/invalid forward telemetry")
    if (metrics["running_batch_max"] or 0) > point["concurrency"]:
        reasons.append("server running batch exceeds configured concurrency")
    return {
        **metrics,
        "status": "invalid" if reasons else "ok",
        "invalid_reasons": reasons,
        "telemetry_errors": collector.errors,
    }
