#!/usr/bin/env python3
"""Compare pure decode at an actual full running batch of 8/16/32/64.

One simultaneous cohort per mode. Throughput uses a bounded window of CUDA
events, with detailed layer tracing in a separate optional server run.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import pairwise

if __package__:
    from . import run as common
else:
    import run as common

REQUEST_PREFIX = "lagom-full-batch-"


def parser():
    p = common.parser(high_concurrency=True)
    p.description = __doc__
    p.add_argument("--max-concurrency", type=int, choices=(8, 16, 32, 64), default=8)
    p.add_argument("--measure-steps", type=int, default=32)
    p.add_argument("--full-batch-warmup", type=int, default=4)
    return p


def validate(args):
    common.validate(args)
    if not 1 <= args.measure_steps <= 256 or args.full_batch_warmup < 0:
        raise ValueError("measure-steps must be 1..256, full-batch-warmup >=0")
    if args.nsys and args.phase == "throughput":
        raise ValueError("--nsys requires --phase trace or all")
    steps = (
        args.trace_steps + args.trace_skip
        if args.phase == "trace"
        else args.measure_steps + args.full_batch_warmup
    )
    if args.phase == "all":
        steps = max(steps, args.trace_steps + args.trace_skip)
    minimum = (steps + 3) * (args.w + 1) + 1
    if args.output_len < minimum:
        raise ValueError(
            f"--output-len must be >= {minimum} to allow the full-batch window and output-cap headroom"
        )


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def aggregate_decode(reports, *, mode, batch_size, steps):
    if not reports:
        raise ValueError("no full-batch decode reports")
    reference = reports[0]["records"]
    if len(reference) != steps:
        raise ValueError("incomplete decode window")
    cohort = sorted(reference[0]["request_ids"])
    iters = [r["forward_iter"] for r in reference]
    if len(set(cohort)) != batch_size or any(b != a + 1 for a, b in pairwise(iters)):
        raise ValueError("invalid cohort or forward sequence")
    for report in reports:
        records = report["records"]
        if (
            report["batch_size"] != batch_size
            or len(records) != steps
            or report["window_ms"] <= 0
        ):
            raise ValueError("incomplete full-batch decode window")
        if [r["forward_iter"] for r in records] != iters:
            raise ValueError("TP ranks sampled different decode windows")
        for i, record in enumerate(records):
            if (
                record["batch_size"] != batch_size
                or sorted(record["request_ids"]) != cohort
                or len(record["accepted_tokens"]) != batch_size
            ):
                raise ValueError("decode sample is not the same full running cohort")
            if not record["cuda_graph"] or not record.get("draft_cuda_graph", False):
                raise ValueError("draft/target fell back to eager in measured decode")
            if (
                record["step_ms"] <= 0
                or record["end_ms"] < record["start_ms"]
                or (i and record["start_ms"] + 1e-4 < records[i - 1]["end_ms"])
            ):
                raise ValueError("invalid CUDA event timings")
            if any(
                not 1 <= n <= report["verify_width"] for n in record["accepted_tokens"]
            ):
                raise ValueError("invalid accepted output count")
            if any(
                n <= 2 * report["verify_width"]
                for n in record["remaining_output_budget_cpu"]
            ):
                raise ValueError("decode sample is too close to the output cap")
            accepted = dict(zip(record["request_ids"], record["accepted_tokens"]))
            expected = dict(
                zip(reference[i]["request_ids"], reference[i]["accepted_tokens"])
            )
            if accepted != expected:
                raise ValueError("TP ranks disagree on accepted token counts")
    rows = []
    for i, record in enumerate(reference):
        rows.append(
            {
                "step": i,
                "forward_iter": record["forward_iter"],
                "running_batch_size": batch_size,
                "output_tokens": sum(record["accepted_tokens"]),
                "gpu_step_ms_max_rank": max(
                    r["records"][i]["step_ms"] for r in reports
                ),
                "cpu_submit_ms_max_rank": max(
                    r["records"][i]["cpu_submit_ms"] for r in reports
                ),
                "inter_step_gap_ms_max_rank": (
                    max(
                        max(
                            0,
                            r["records"][i]["start_ms"] - r["records"][i - 1]["end_ms"],
                        )
                        for r in reports
                    )
                    if i
                    else None
                ),
            }
        )
    tokens = sum(row["output_tokens"] for row in rows)
    window_ms = max(r["window_ms"] for r in reports)
    step_times = [row["gpu_step_ms_max_rank"] for row in rows]
    summary = {
        "mode": mode,
        "running_batch_size": batch_size,
        "measured_steps": steps,
        "output_tokens": tokens,
        "window_ms_max_rank": window_ms,
        "decode_tokens_per_s": tokens * 1000 / window_ms,
        "mean_per_request_tokens_per_s": tokens * 1000 / window_ms / batch_size,
        "gpu_forward_tokens_per_s": tokens * 1000 / sum(step_times),
        "mean_step_cycle_ms": window_ms / steps,
        "mean_gpu_step_ms": sum(step_times) / steps,
        "p50_gpu_step_ms": percentile(step_times, 0.5),
        "p95_gpu_step_ms": percentile(step_times, 0.95),
        "mean_accepted_tokens_per_request_per_step": tokens / batch_size / steps,
    }
    return summary, rows


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cohort(args, url, rows, folder, deadline):
    barrier = threading.Barrier(args.max_concurrency)
    pool = ThreadPoolExecutor(max_workers=args.max_concurrency)

    def submit(index, row):
        barrier.wait(timeout=30)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("mode runtime budget exhausted")
        rid = f"{REQUEST_PREFIX}{index:04d}"
        result = common.generate(
            url,
            row["input_ids"],
            args.output_len,
            min(args.request_timeout, remaining),
            rid=rid,
        )
        result.update(rid=rid, prompt_sha256=row["token_sha256"])
        return result

    results = []
    started = time.monotonic()
    try:
        futures = [pool.submit(submit, i, row) for i, row in enumerate(rows[1:])]
        for future in as_completed(
            futures, timeout=max(0.01, deadline - time.monotonic())
        ):
            results.append(future.result())
            (folder / "requests.json").write_text(json.dumps(results, indent=2))
    finally:
        # On error the enclosing server finally kills our process group, which
        # closes in-flight requests; do not block cleanup waiting on sockets.
        pool.shutdown(wait=False, cancel_futures=True)
    (folder / "cohort.json").write_text(
        json.dumps(
            {
                "requests": len(results),
                "client_wall_s_including_prefill": time.monotonic() - started,
            },
            indent=2,
        )
    )


def run_mode(args, mode, phase, rows):
    folder = args.output / phase / mode
    folder.mkdir(parents=True)
    port = common.free_port()
    command = common.server_command(
        args, mode, port, max_running_requests=args.max_concurrency
    )
    env = os.environ.copy()
    env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    for name in (
        "SGLANG_LAGOM_PERF_TRACE",
        "SGLANG_LAGOM_DECODE_PERF",
        "SGLANG_DSPARK_DEBUG_DUMP",
        "SGLANG_OASISKV_TRACE_CONFIG",
    ):
        env.pop(name, None)
    config_file = folder / "measure_config.json"
    if phase == "throughput":
        config = {
            "output_dir": str(folder / "decode"),
            "batch_size": args.max_concurrency,
            "steps": args.measure_steps,
            "warmup": args.full_batch_warmup,
            "verify_width": args.w + 1,
            "request_prefix": REQUEST_PREFIX,
        }
        env["SGLANG_LAGOM_DECODE_PERF"] = str(config_file)
    else:
        config = {
            "output_dir": str(folder / "steps"),
            "required_batch_size": args.max_concurrency,
            "steps": args.trace_steps,
            "skip": args.trace_skip,
            "nsys": args.nsys,
        }
        env["SGLANG_LAGOM_PERF_TRACE"] = str(config_file)
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
    config_file.write_text(json.dumps(config, indent=2))
    (folder / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    k: env[k]
                    for k in (
                        "SGLANG_RAGGED_VERIFY_MODE",
                        "SGLANG_LAGOM_DECODE_PERF",
                        "SGLANG_LAGOM_PERF_TRACE",
                    )
                    if k in env
                },
            },
            indent=2,
        )
    )
    print(
        f"[{phase}/{mode}, C={args.max_concurrency}] {shlex.join(command)}", flush=True
    )
    url = f"http://127.0.0.1:{port}"
    with (folder / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=common.ROOT,
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
                if time.monotonic() >= deadline:
                    raise TimeoutError("server startup timed out")
                time.sleep(1)
            with urllib.request.urlopen(
                url + "/get_server_info", timeout=10
            ) as response:
                (folder / "server_info.json").write_bytes(response.read())
            deadline = time.monotonic() + args.mode_budget
            common.generate(
                url,
                rows[0]["input_ids"],
                args.warmup_output,
                min(args.mode_budget, args.request_timeout),
                rid="lagom-warmup",
            )
            run_cohort(args, url, rows, folder, deadline)
            # A final tiny-output request lets pending event queries export on
            # a prefill/decode callback even after the measured cohort finishes.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("mode runtime budget exhausted before export")
            common.generate(
                url,
                rows[0]["input_ids"],
                8,
                min(args.request_timeout, remaining),
                rid="lagom-flush",
            )
            if phase == "trace":
                common.validate_traces(
                    folder / "steps",
                    args.tp,
                    args.trace_steps,
                    mode,
                    batch_size=args.max_concurrency,
                )
                return None
            files = [
                p
                for p in (folder / "decode").glob("rank-*.json")
                if not p.name.endswith(".status.json")
            ]
            if len(files) != args.tp:
                raise RuntimeError(
                    f"No complete full-batch window on all ranks: expected {args.tp} reports, got {len(files)}. "
                    f"Need {args.full_batch_warmup}+{args.measure_steps} consecutive decode steps at actual batch "
                    f"{args.max_concurrency}. Inspect decode/*.status.json and server.log for capacity/retractions; "
                    "no partial-batch throughput is reported."
                )
            summary, steps = aggregate_decode(
                [json.loads(p.read_text()) for p in sorted(files)],
                mode=mode,
                batch_size=args.max_concurrency,
                steps=args.measure_steps,
            )
            (folder / "summary.json").write_text(json.dumps(summary, indent=2))
            write_csv(folder / "decode_steps.csv", steps)
            return summary
        finally:
            common.stop(process)


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
            print(
                shlex.join(
                    common.server_command(
                        args, mode, 30000, max_running_requests=args.max_concurrency
                    )
                )
            )
        print(
            f"actual full batch={args.max_concurrency}; warmup={args.full_batch_warmup}, measured decode steps={args.measure_steps}"
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = vars(args).copy()
    manifest["output"] = str(args.output)
    manifest["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=common.ROOT, text=True
    ).strip()
    manifest["input_format"] = "packed_sharegpt_token_ids_no_chat_template"
    manifest["measurement"] = "consecutive_actual_full_batch_decode_cuda_events"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = common.prepare(args, prompt_count=args.max_concurrency + 1)
    summaries = []
    if args.phase in ("throughput", "all"):
        for mode in args.modes:
            summaries.append(run_mode(args, mode, "throughput", rows))
            write_csv(args.output / "decode_throughput.csv", summaries)
            print(json.dumps(summaries[-1], indent=2), flush=True)
    if args.phase in ("trace", "all"):
        for mode in args.trace_modes:
            run_mode(args, mode, "trace", rows)


if __name__ == "__main__":
    main()
