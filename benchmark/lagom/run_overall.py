"""Sequential 8xH100, TP4+TP4 Mooncake overall benchmark (CPU-only dry-run)."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from longbench_v2_fixed import load_manifest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ("hisparse", "dhisparse", "lagom")
INPUT_LENS = (32768, 65536, 131072)
CONCURRENCIES = (8, 16, 32, 64, 128, 256)
OUTPUT_LEN = 1024
CLEANUP_RESERVE = 20
TRACE_ENV = (
    "SGLANG_LAGOM_PERF_TRACE", "SGLANG_LAGOM_DECODE_PERF",
    "SGLANG_DSPARK_DEBUG_DUMP", "SGLANG_OASISKV_TRACE_CONFIG",
)
SUMMARY_FIELDS = (
    "method", "input_len", "max_concurrency", "num_requests", "status",
    "output_tps", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_ttft_ms", "benchmark_seconds", "cell_seconds", "completed",
    "error", "attempt",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_config(path):
    defaults = json.loads((HERE / "overall.example.json").read_text())
    supplied = json.loads(Path(path).read_text())
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = defaults | supplied
    cfg["python"] = cfg["python"] or sys.executable
    if "/" in cfg["python"]:
        cfg["python"] = str((ROOT / cfg["python"]).resolve())
    cfg["tokenizer"] = cfg["tokenizer"] or cfg["model"]
    positive = (
        "w", "p", "top_k", "device_buffer_size", "chunked_prefill_size",
        "page_size", "startup_timeout_seconds", "cell_timeout_seconds", "host_to_device_ratio",
    )
    if any(type(cfg[k]) is not int or cfg[k] <= 0 for k in positive):
        raise ValueError("Window, capacity and timeout settings must be positive integers")
    if not CLEANUP_RESERVE + 10 < cfg["cell_timeout_seconds"] <= 21600:
        raise ValueError("Each cell must have a timeout in (30, 21600] seconds")
    if not 0 < cfg["mem_fraction_static"] < 1 or cfg["host_to_device_ratio"] <= 0:
        raise ValueError("Invalid memory settings")
    hot = cfg["device_buffer_size"]
    if hot % 64 or not 64 <= hot <= 32768:
        raise ValueError("device_buffer_size must be a multiple of 64 in [64,32768]")
    if (cfg["w"] + cfg["p"] + 1) * cfg["top_k"] > min(hot, 8192):
        raise ValueError("Require (W+P+1)*index_topk <= min(device_buffer_size,8192)")
    if cfg["decode_extra_slots"] < 0:
        raise ValueError("decode_extra_slots must be nonnegative")
    gpu_sets = [set(range(cfg[k], cfg[k] + 4)) for k in ("prefill_base_gpu", "decode_base_gpu")]
    if gpu_sets[0] & gpu_sets[1] or gpu_sets[0] | gpu_sets[1] != set(range(8)):
        raise ValueError("P/D must partition the eight visible GPUs into two TP4 groups")
    ports = [cfg[k] for k in (
        "prefill_port", "decode_port", "router_port", "bootstrap_port",
        "prefill_nccl_port", "decode_nccl_port",
    )]
    if len(set(ports)) != len(ports) or any(not 1024 <= p <= 65535 for p in ports):
        raise ValueError("Use six distinct, unprivileged ports")
    if not isinstance(cfg["env"], dict):
        raise ValueError("env must be a JSON object")
    for key in ("prefill_extra_args", "decode_extra_args", "router_extra_args"):
        if not isinstance(cfg[key], list) or not all(isinstance(x, str) for x in cfg[key]):
            raise ValueError(f"{key} must be a list of argv strings")
    return cfg


def matrix(methods, lengths, concurrencies):
    return [
        {"method": method, "input_len": length, "max_concurrency": concurrency,
         "num_requests": 4 * concurrency}
        for method in METHODS if method in methods
        for length in sorted(set(lengths))
        for concurrency in sorted(set(concurrencies))
    ]


def check_extra(base, extra):
    controlled = {word.split("=")[0] for word in base if word.startswith("--")}
    forbidden = (
        "--speculative", "--hisparse", "--enable-hisparse",
        "--disable-cuda-graph", "--cuda-graph", "--enable-dp-attention",
        "--dp-size", "--pp-size", "--dcp-size", "--attn-cp-size",
    )
    for word in extra:
        flag = word.split("=")[0]
        if flag in controlled or flag.startswith(forbidden) or flag in ("--tp", "--tp-size"):
            raise ValueError(f"Extra args cannot override protocol-controlled flag {flag}")
    return base + extra


def commands(cfg, cell, data_dir, folder):
    method, length, concurrency = (
        cell["method"], cell["input_len"], cell["max_concurrency"]
    )
    spec = method != "hisparse"
    context = length + OUTPUT_LEN + cfg["w"] + cfg["p"] + 128
    graph_bs = [1 << i for i in range(concurrency.bit_length())]
    urls = {role: f"http://127.0.0.1:{cfg[role + '_port']}"
            for role in ("prefill", "decode", "router")}
    launch = {}
    for role in ("prefill", "decode"):
        argv = [
            cfg["python"], "-m", "sglang.launch_server",
            "--model-path", cfg["model"], "--trust-remote-code",
            "--tp-size", "4", "--base-gpu-id", str(cfg[role + "_base_gpu"]),
            "--host", "127.0.0.1", "--port", str(cfg[role + "_port"]),
            "--nccl-port", str(cfg[role + "_nccl_port"]),
            "--disaggregation-mode", role,
            "--disaggregation-transfer-backend", "mooncake",
            "--disaggregation-bootstrap-port", str(cfg["bootstrap_port"]),
            "--context-length", str(context),
            "--max-running-requests", str(concurrency),
            "--chunked-prefill-size", str(cfg["chunked_prefill_size"]),
            "--page-size", str(cfg["page_size"]),
            "--mem-fraction-static", str(cfg["mem_fraction_static"]),
            "--disable-radix-cache", "--random-seed", str(cfg["seed"]),
            "--moe-runner-backend", cfg["moe_runner_backend"],
        ]
        if spec:
            # P injects target hidden states into the draft context KV pool;
            # Mooncake transfers that pool along with target KV to D.
            argv += [
                "--speculative-algorithm", "DSPARK",
                "--speculative-draft-model-path", cfg["draft"],
                "--speculative-num-draft-tokens", str(cfg["w"] + 1),
                "--speculative-dspark-block-size", str(cfg["w"]),
                "--speculative-moe-runner-backend", cfg["draft_moe_runner_backend"],
            ]
        if role == "decode":
            hisparse = {
                "device_buffer_size": cfg["device_buffer_size"],
                "host_to_device_ratio": cfg["host_to_device_ratio"],
                "force_h2d_each_step": False,
            }
            if method == "lagom":
                hisparse.update(
                    prefetcher="lagom", prefetcher_config={"prediction_steps": cfg["p"]}
                )
            argv += [
                "--enable-hisparse", "--hisparse-config", canonical(hisparse),
                "--disaggregation-decode-extra-slots", str(cfg["decode_extra_slots"]),
                "--cuda-graph-config", canonical({"decode": {"backend": "full", "bs": graph_bs}}),
            ]
        launch[role] = check_extra(argv, cfg[role + "_extra_args"])
    router = [
        cfg["python"], "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--prefill", urls["prefill"], str(cfg["bootstrap_port"]),
        "--decode", urls["decode"], "--host", "127.0.0.1",
        "--port", str(cfg["router_port"]), "--disable-retries",
        "--request-timeout-secs", str(cfg["cell_timeout_seconds"]),
        "--queue-timeout-secs", str(cfg["cell_timeout_seconds"]),
        "--worker-startup-timeout-secs", str(cfg["startup_timeout_seconds"]),
    ]
    launch["router"] = check_extra(router, cfg["router_extra_args"])
    launch["benchmark"] = [
        cfg["python"], str(HERE / "longbench_v2_fixed.py"), "bench",
        "--backend", "sglang", "--base-url", urls["router"],
        "--model", cfg["model"], "--tokenizer", cfg["tokenizer"],
        "--dataset-name", "longbench_v2",
        "--dataset-path", str(data_dir / str(length) / "manifest.json"),
        "--tokenize-prompt", "--num-prompts", str(cell["num_requests"]),
        "--max-concurrency", str(concurrency), "--request-rate", "inf",
        "--sharegpt-output-len", str(OUTPUT_LEN), "--temperature", "0",
        "--seed", str(cfg["seed"]), "--warmup-requests", "1",
        "--output-file", str(folder / "benchmark.jsonl"), "--output-details",
        "--disable-tqdm",
    ]
    return launch, urls


def source_preflight():
    relative = "python/sglang/srt/managers/hisparse_coordinator.py"
    text = (ROOT / relative).read_text()
    tree = ast.parse(text)
    evidence = {}
    for name, callee in (
        ("swap_in_selected_pages_spec", "_run_swap_in_verify_window"),
        ("_run_swap_in_verify_window", "load_cache_to_device_buffer_spec_mla"),
    ):
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        ok = any(
            isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == callee
            for node in matches for n in ast.walk(node)
        )
        evidence[name] = {
            "file": relative, "line": matches[0].lineno if matches else None,
            "calls": callee, "found": ok,
        }
    hook = "python/sglang/srt/arg_groups/speculative_hook.py"
    hook_text = (ROOT / hook).read_text()
    blocked = "Lagom currently requires colocated prefill and decode." in hook_text
    return {
        "oneshot_source_evidence": evidence,
        "oneshot_source_check_passed": all(v["found"] for v in evidence.values()),
        "lagom_pd_explicitly_blocked": blocked,
        "lagom_pd_guard_file": hook,
        "note": "Source inspection only; passing does not establish PD/CUDA Graph correctness.",
    }


def validate_server_info(info, cfg, cell, role):
    """Reject silent backend/method/graph fallbacks before timing requests."""
    expected = {
        "disaggregation_mode": role, "disaggregation_transfer_backend": "mooncake",
        "tp_size": 4, "max_running_requests": cell["max_concurrency"],
        "enable_hisparse": role == "decode",
        "moe_runner_backend": cfg["moe_runner_backend"],
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"{role} resolved {key}={info.get(key)!r}, expected {value!r}")
    algo = info.get("speculative_algorithm")
    if cell["method"] == "hisparse":
        if algo is not None:
            raise ValueError("Hisparse baseline unexpectedly enabled speculation")
    else:
        if str(algo).upper() != "DSPARK":
            raise ValueError(f"{role} did not resolve DSPARK")
        if info.get("speculative_num_draft_tokens") != cfg["w"] + 1:
            raise ValueError(f"{role} resolved a different speculative window")
        if info.get("speculative_moe_runner_backend") != cfg["draft_moe_runner_backend"]:
            raise ValueError(f"{role} resolved a different draft MoE backend")
    if role == "decode":
        if info.get("disable_cuda_graph") is not False:
            raise ValueError("Decode CUDA Graph is disabled or not reported")
        graph = info.get("cuda_graph_config")
        if isinstance(graph, str):
            graph = json.loads(graph)
        expected_bs = [1 << i for i in range(cell["max_concurrency"].bit_length())]
        if (
            not isinstance(graph, dict)
            or graph.get("decode", {}).get("backend") != "full"
            or graph.get("decode", {}).get("bs") != expected_bs
        ):
            raise ValueError("Resolved decode graph backend/batch sizes differ from the protocol")
        hs = info.get("hisparse_config")
        if isinstance(hs, str):
            hs = json.loads(hs)
        if not isinstance(hs, dict):
            raise ValueError("Resolved HiSparse config is missing")
        for key in ("device_buffer_size", "host_to_device_ratio"):
            if hs.get(key) != cfg[key]:
                raise ValueError(f"Resolved HiSparse {key} differs from the shared config")
        if hs.get("force_h2d_each_step", False):
            raise ValueError("force_h2d_each_step must remain false")
        if cell["method"] == "lagom":
            if hs.get("prefetcher") != "lagom" or hs.get("prefetcher_config", {}).get("prediction_steps") != cfg["p"]:
                raise ValueError("Lagom prefetch did not resolve with the requested P")
        elif hs.get("prefetcher"):
            raise ValueError("A baseline unexpectedly enabled prefetch")


def runtime_env(cfg):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg["env"].items()})
    for key in TRACE_ENV:
        env.pop(key, None)
    if env.get("SGLANG_DEBUG_HISPARSE_SKIP_IO", "0").lower() not in ("", "0", "false"):
        raise ValueError("Disable fake HiSparse IO before benchmarking")
    if float(env.get("SGLANG_SIMULATE_ACC_LEN", "0")) != 0:
        raise ValueError("Disable simulated speculative acceptance")
    env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = str(cfg["cell_timeout_seconds"])
    env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = str(cfg["cell_timeout_seconds"])
    return env


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Cell/startup deadline exceeded")
    return value


def check_processes(processes):
    for role, (proc, _) in processes.items():
        if proc.poll() is not None:
            raise RuntimeError(f"{role} exited with code {proc.returncode}; see {role}.log")


def get_json(url, deadline):
    with urllib.request.urlopen(url, timeout=min(5, remaining(deadline))) as response:
        return json.load(response)


def wait_ready(url, processes, deadline):
    while True:
        check_processes(processes)
        try:
            with urllib.request.urlopen(url + "/health", timeout=min(2, remaining(deadline))) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(min(0.25, remaining(deadline)))


def spawn(role, argv, folder, env, processes):
    log = (folder / f"{role}.log").open("w")
    try:
        proc = subprocess.Popen(
            argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    processes[role] = (proc, log)


def cleanup(processes):
    # Signal whole sessions, including TP workers whose launcher already exited.
    def send(sig):
        for proc, _ in processes.values():
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass

    send(signal.SIGTERM)
    end = time.monotonic() + 5
    while time.monotonic() < end and any(p.poll() is None for p, _ in processes.values()):
        time.sleep(0.1)
    send(signal.SIGKILL)
    for proc, log in processes.values():
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        log.close()


def validate_result(result, cell):
    n, length = cell["num_requests"], cell["input_len"]
    if result.get("completed") != n:
        raise ValueError(f"Only {result.get('completed')} of {n} formal requests completed")
    if result.get("max_concurrency") != cell["max_concurrency"]:
        raise ValueError("Benchmark concurrency does not match this cell")
    if result.get("input_lens") != [length] * n or result.get("output_lens") != [OUTPUT_LEN] * n:
        raise ValueError("Actual per-request token lengths do not match the protocol")
    if result.get("total_input_tokens") != n * length or result.get("total_output_tokens") != n * OUTPUT_LEN:
        raise ValueError("Aggregate token counts do not match the protocol")
    if len(result.get("errors", [])) != n or any(result["errors"]):
        raise ValueError("Benchmark contains request errors")
    for key in ("duration", "output_throughput", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid benchmark metric {key}: {value}")
    if not math.isclose(result["output_throughput"], n * OUTPUT_LEN / result["duration"], rel_tol=1e-6):
        raise ValueError("TPS does not match output tokens / measurement duration")
    return {
        "output_tps": result["output_throughput"],
        "mean_tpot_ms": result["mean_tpot_ms"],
        "median_tpot_ms": result["median_tpot_ms"],
        "p99_tpot_ms": result["p99_tpot_ms"],
        "mean_ttft_ms": result.get("mean_ttft_ms"),
        "benchmark_seconds": result["duration"], "completed": result["completed"],
    }


def assert_free_ports(cfg):
    sockets = []
    try:
        for key, port in cfg.items():
            if key.endswith("_port"):
                sock = socket.socket()
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
    finally:
        for sock in sockets:
            sock.close()


def run_cell(cfg, cell, data_dir, folder):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + cfg["cell_timeout_seconds"] - CLEANUP_RESERVE
    startup_deadline = min(deadline, started + cfg["startup_timeout_seconds"])
    processes = {}
    result = cell | {"status": "running", "attempt": str(folder), "error": ""}
    save(folder / "status.json", result)
    try:
        env = runtime_env(cfg)
        launch, urls = commands(cfg, cell, data_dir, folder)
        save(folder / "launch.json", {
            "commands": launch,
            "shell_display_only": {k: shlex.join(v) for k, v in launch.items()},
            "controlled_environment": {
                k: env[k] for k in (
                    set(cfg["env"]) | {
                        "SGLANG_RAGGED_VERIFY_MODE", "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT",
                        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT",
                    }
                )
            },
            "removed_trace_environment": list(TRACE_ENV),
        })
        assert_free_ports(cfg)
        for role in ("prefill", "decode", "router"):
            remaining(startup_deadline)
            spawn(role, launch[role], folder, env, processes)
            wait_ready(urls[role], processes, startup_deadline)
        for role in ("prefill", "decode"):
            info = get_json(urls[role] + "/server_info", deadline)
            save(folder / f"{role}_server_info.json", info)
            validate_server_info(info, cfg, cell, role)
        remaining(deadline)
        spawn("benchmark", launch["benchmark"], folder, env, processes)
        bench = processes["benchmark"][0]
        while bench.poll() is None:
            check_processes({k: v for k, v in processes.items() if k != "benchmark"})
            time.sleep(min(0.25, remaining(deadline)))
        remaining(deadline)
        if bench.returncode != 0:
            raise RuntimeError(f"SGLang benchmark exited with {bench.returncode}; see benchmark.log")
        lines = (folder / "benchmark.jsonl").read_text().splitlines()
        if len(lines) != 1:
            raise ValueError("Expected exactly one fresh SGLang benchmark result")
        result.update(validate_result(json.loads(lines[0]), cell))
        result["status"] = "ok"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by user/signal")
        raise
    except TimeoutError as exc:
        result.update(status="timeout", error=str(exc))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup(processes)
        result["cell_seconds"] = time.monotonic() - started
        if result["cell_seconds"] > cfg["cell_timeout_seconds"] and result["status"] == "ok":
            result.update(status="timeout", error="Cell including cleanup exceeded its budget")
            for key in ("output_tps", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"):
                result.pop(key, None)
        save(folder / "status.json", result)
    return result


def summary(output):
    rows = []
    for cell_dir in sorted(output.glob("*/*/*")):
        attempts = sorted(cell_dir.glob("attempt-*/status.json"))
        if attempts:
            rows.append(json.loads(attempts[-1].read_text()))
    path = output / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def provenance():
    result = {"python": sys.version}
    for name, cmd in (
        ("git_head", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--porcelain", "--untracked-files=no"]),
        ("gpus", ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"]),
    ):
        try:
            result[name] = subprocess.check_output(cmd, cwd=ROOT, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = str(exc)
    try:
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT, timeout=5)
        result["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.SubprocessError) as exc:
        result["git_diff_sha256"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "overall.example.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--input-lens", nargs="+", type=int, choices=INPUT_LENS, default=list(INPUT_LENS))
    parser.add_argument("--concurrencies", nargs="+", type=int, choices=CONCURRENCIES, default=list(CONCURRENCIES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip only successful identical cells")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir, output = args.data_dir.resolve(), args.output.resolve()
    cells = matrix(args.methods, args.input_lens, args.concurrencies)
    preflight = source_preflight()
    if args.dry_run:
        print(json.dumps({
            "preflight": preflight, "config": cfg, "cells": [
                cell | {"commands": commands(cfg, cell, data_dir, output / cell["method"] / str(cell["input_len"]) / str(cell["max_concurrency"]) / "attempt-0001")[0]}
                for cell in cells
            ],
        }, indent=2))
        return
    if "lagom" in args.methods and preflight["lagom_pd_explicitly_blocked"]:
        raise RuntimeError(
            "This checkout explicitly rejects Lagom + PD in speculative_hook.py. "
            "Integrate and GPU-validate PD support first; do not merely remove the guard. "
            "See benchmark/lagom/OVERALL.md. --dry-run is available."
        )
    if set(args.methods) & {"dhisparse", "lagom"} and not preflight["oneshot_source_check_passed"]:
        raise RuntimeError("DHisparse oneshot verify source check failed; inspect this checkout")
    manifests = {}
    for path in sorted(data_dir.glob("*/manifest.json")):
        m = load_manifest(path)
        length = m["input_len"]
        if path.parent.name != str(length) or m["seed"] != cfg["seed"]:
            raise ValueError(f"Input manifest length/seed mismatch: {path}")
        if length in args.input_lens and len(m["order"]) < 4 * max(args.concurrencies):
            raise ValueError(f"Not enough prepared requests: {path}")
        manifests[str(length)] = sha(m)
    if any(str(length) not in manifests for length in args.input_lens):
        raise ValueError("Prepare a token-bank manifest for every selected input length")
    current_provenance = provenance()
    protocol = {
        "source": {
            **{key: current_provenance[key] for key in ("git_head", "git_status", "git_diff_sha256")},
            "harness_sha256": sha({name: (HERE / name).read_text() for name in ("run_overall.py", "longbench_v2_fixed.py")}),
        },
        "config": cfg, "data_dir": str(data_dir), "data_manifests": manifests,
        "output_len": OUTPUT_LEN, "requests_per_concurrency": 4,
    }
    identity = sha(protocol)
    # Exclusive lock prevents concurrent drivers sharing ports/results.
    output.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if not args.resume or existing["protocol_sha256"] != identity:
                raise ValueError("Use a new output directory, or --resume with an identical protocol")
        else:
            save(manifest_path, {
                "protocol": protocol, "protocol_sha256": identity,
                "provenance": current_provenance, "preflight": preflight,
            })
        runtime_env(cfg)
        failed = False
        try:
            for cell in cells:
                cell_dir = output / cell["method"] / str(cell["input_len"]) / str(cell["max_concurrency"])
                attempts = sorted(cell_dir.glob("attempt-*"))
                if args.resume and attempts:
                    status_file = attempts[-1] / "status.json"
                    if status_file.exists() and json.loads(status_file.read_text())["status"] == "ok":
                        print(f"Skip completed {cell}", flush=True)
                        continue
                folder = cell_dir / f"attempt-{len(attempts) + 1:04d}"
                print(f"Run {cell} -> {folder}", flush=True)
                result = run_cell(cfg, cell, data_dir, folder)
                failed |= result["status"] != "ok"
                summary(output)
                print(canonical(result), flush=True)
        finally:
            summary(output)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    main()
