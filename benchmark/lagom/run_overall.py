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

from analyze_nsys import step_breakdown
from h2d_trace import compare_cells, read_cell
from longbench_v2_fixed import load_manifest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ("hisparse", "dhisparse", "lagom")
INPUT_LENS = (32768, 65536, 131072, 204800)
CONCURRENCIES = (8, 16, 32, 64, 128, 256)
OUTPUT_LEN = 1024
CLEANUP_RESERVE = 20
TRACE_ENV = (
    "SGLANG_LAGOM_PERF_TRACE", "SGLANG_LAGOM_DECODE_PERF",
    "SGLANG_DSPARK_DEBUG_DUMP", "SGLANG_OASISKV_TRACE_CONFIG",
)
SUMMARY_FIELDS = (
    "method",
    "force_h2d_each_step",
    "input_len",
    "max_concurrency",
    "num_requests",
    "status",
    "output_tps",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_ttft_ms",
    "benchmark_seconds",
    "cell_seconds",
    "completed",
    "error",
    "attempt",
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
    if "/" in cfg["python"] and not Path(cfg["python"]).is_absolute():
        cfg["python"] = str(ROOT / cfg["python"])
    cfg["tokenizer"] = cfg["tokenizer"] or cfg["model"]
    positive = (
        "w", "p", "prefetch_slots", "top_k", "device_buffer_size", "chunked_prefill_size",
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
    if hot < cfg["top_k"]:
        raise ValueError("device_buffer_size must hold at least index_topk entries")
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
        "--disaggregation-dsv4-state-execution-width",
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
    if method == "lagom" and (cfg["w"] + cfg["p"] + 1) * cfg["top_k"] > 8192:
        raise ValueError(
            "Require (W+P+1)*index_topk <= verification staging capacity 8192"
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
        if role == "prefill" and method == "lagom":
            # Prepare the decoder's compressor-state layout only. P keeps
            # ordinary DSPARK, its W+1 public window and no HiSparse/prefetch.
            argv += [
                "--disaggregation-dsv4-state-execution-width",
                str(cfg["w"] + cfg["p"] + 1),
            ]
        if role == "decode":
            hisparse = {
                "top_k": cfg["top_k"],
                "device_buffer_size": cfg["device_buffer_size"],
                "host_to_device_ratio": cfg["host_to_device_ratio"],
                "force_h2d_each_step": cell.get("force_h2d_each_step", False),
            }
            if method == "lagom":
                hisparse.update(
                    prefetcher="lagom", prefetcher_config={
                        "prediction_steps": cfg["p"], "buffer_slots": cfg["prefetch_slots"],
                    }
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
    if cell.get("nsys"):
        launch["decode"] = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-event-trace=false",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o",
            str(folder / "prof"),
        ] + launch["decode"]
    sharegpt = cell.get("dataset") == "sharegpt"
    client = (
        ["-m", "sglang.benchmark.serving"]
        if sharegpt
        else [str(HERE / "longbench_v2_fixed.py"), "bench"]
    )
    dataset = (
        [
            "--dataset-name",
            "sharegpt",
            "--dataset-path",
            str(data_dir),
            "--sharegpt-input-len",
            str(length),
        ]
        if sharegpt
        else [
            "--dataset-name",
            "longbench_v2",
            "--dataset-path",
            str(data_dir / str(length) / "manifest.json"),
        ]
    )
    launch["benchmark"] = [
        cfg["python"],
        *client,
        "--backend",
        "sglang",
        "--base-url",
        urls["router"],
        "--model",
        cfg["model"],
        "--tokenizer",
        cfg["tokenizer"],
        *dataset,
        "--tokenize-prompt",
        "--num-prompts",
        str(cell["num_requests"]),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--sharegpt-output-len",
        str(OUTPUT_LEN),
        "--temperature",
        "0",
        "--seed",
        str(cfg["seed"]),
        "--warmup-requests",
        "1",
        "--output-file",
        str(folder / "benchmark.jsonl"),
        "--output-details",
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
        ("_run_swap_in_kernel", "_run_swap_in_verify_window"),
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
    layout_files = {
        "server_args": "python/sglang/srt/server_args.py",
        "pool": "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
        "budget": "python/sglang/srt/model_executor/pool_configurator.py",
        "transfer": "python/sglang/srt/disaggregation/mooncake/conn.py",
    }
    layout_sources = {key: (ROOT / path).read_text() for key, path in layout_files.items()}
    pd_layout_ready = (
        not blocked
        and "disaggregation_dsv4_state_execution_width" in layout_sources["server_args"]
        and "_validate_dsv4_pd_state_layout(server_args)" in hook_text
        and "dsv4_state_execution_width(" in layout_sources["pool"]
        and "dsv4_state_execution_width(" in layout_sources["budget"]
        and "validate_state_transfer_layout(" in layout_sources["transfer"]
        and "dst_state_item_lens=dst_item_lens" in layout_sources["transfer"]
    )
    return {
        "oneshot_source_evidence": evidence,
        "oneshot_source_check_passed": all(v["found"] for v in evidence.values()),
        "lagom_pd_explicitly_blocked": blocked,
        "lagom_pd_guard_file": hook,
        "lagom_pd_layout_ready": pd_layout_ready,
        "lagom_pd_layout_files": layout_files,
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
    expected_state_width = (
        cfg["w"] + cfg["p"] + 1
        if role == "prefill" and cell["method"] == "lagom"
        else None
    )
    if info.get("disaggregation_dsv4_state_execution_width") != expected_state_width:
        raise ValueError(f"{role} resolved a different PD compressor-state execution width")
    if role == "prefill" and info.get("is_lagom", False):
        raise ValueError("Prefill must remain ordinary DSPARK; only its state layout is widened")
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
        if hs.get("force_h2d_each_step", False) is not cell.get(
            "force_h2d_each_step", False
        ):
            raise ValueError(
                "Resolved force_h2d_each_step differs from the requested arm"
            )
        if cell["method"] == "lagom":
            if hs.get("prefetcher") != "lagom" or hs.get("prefetcher_config", {}).get("prediction_steps") != cfg["p"]:
                raise ValueError("Lagom prefetch did not resolve with the requested P")
            if hs.get("prefetcher_config", {}).get("buffer_slots") != cfg["prefetch_slots"]:
                raise ValueError("Lagom prefetch did not resolve with the requested N")
        elif hs.get("prefetcher"):
            raise ValueError("A baseline unexpectedly enabled prefetch")


def cell_directory(output, cell):
    method = cell["method"]
    if "force_h2d_each_step" in cell:
        method += "-force-h2d" if cell["force_h2d_each_step"] else "-baseline"
    return output / method / str(cell["input_len"]) / str(cell["max_concurrency"])


def trace_config(cell, folder):
    if "force_h2d_each_step" not in cell:
        return None
    return {
        "output_dir": str(folder / "steps"),
        "required_batch_size": cell["trace_batch_size"],
        "steps": cell["trace_steps"],
        "skip": cell["trace_skip"],
        "nsys": cell["nsys"],
        "min_output_tokens": OUTPUT_LEN - 1,  # PD can publish one token during prefill.
    }


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
        trace = trace_config(cell, folder)
        decode_env = env
        if trace:
            path = folder / "trace_config.json"
            save(path, trace)
            decode_env = env | {"SGLANG_LAGOM_PERF_TRACE": str(path)}
        save(
            folder / "launch.json",
            {
                "commands": launch,
                "shell_display_only": {k: shlex.join(v) for k, v in launch.items()},
                "controlled_environment": {
                    k: env[k]
                    for k in (
                        set(cfg["env"])
                        | {
                            "SGLANG_RAGGED_VERIFY_MODE",
                            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT",
                            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT",
                        }
                    )
                },
                "decode_trace_config": trace,
                "removed_trace_environment": list(TRACE_ENV),
            },
        )
        assert_free_ports(cfg)
        for role in ("prefill", "decode", "router"):
            remaining(startup_deadline)
            spawn(
                role,
                launch[role],
                folder,
                decode_env if role == "decode" else env,
                processes,
            )
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
        if trace:
            # All four TP processes must deliver their complete measured window.
            save(folder / "h2d.json", read_cell(folder, cell))
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
        if cell.get("nsys") and result["status"] == "ok":
            try:
                with (folder / "nsys-stats.txt").open("w") as log:
                    subprocess.run(
                        [
                            "nsys",
                            "stats",
                            "--force-export=true",
                            "--report",
                            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum",
                            str(folder / "prof.nsys-rep"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=remaining(deadline + CLEANUP_RESERVE),
                    )
                breakdown = step_breakdown(folder / "prof.sqlite")
                save(folder / "nsys-steps.json", breakdown)
            except Exception as exc:
                result.update(status="failed", error=f"Nsight export: {exc}")
        result["cell_seconds"] = time.monotonic() - started
        if result["cell_seconds"] > cfg["cell_timeout_seconds"] and result["status"] == "ok":
            result.update(status="timeout", error="Cell including cleanup exceeded its budget")
        if result["status"] != "ok":
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
    parser.add_argument("--config", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-dir", type=Path, help="Prepared LongBench-v2 token banks"
    )
    source.add_argument(
        "--sharegpt",
        type=Path,
        help="Pack a local ShareGPT JSON file into exact-length token IDs",
    )
    parser.add_argument(
        "--h2d-pair", action="store_true", help="Trace two DSPARK fused-swap-in arms"
    )
    parser.add_argument("--trace-steps", type=int, default=12)
    parser.add_argument("--trace-skip", type=int, default=2)
    parser.add_argument(
        "--trace-batch-size",
        type=int,
        help="Actual decode batch to match (default: concurrency)",
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Profile the paired decode windows with Nsight Systems",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--input-lens", nargs="+", type=int, choices=INPUT_LENS)
    parser.add_argument("--concurrencies", nargs="+", type=int, choices=CONCURRENCIES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip only successful identical cells")
    args = parser.parse_args()
    if args.nsys and not args.h2d_pair:
        parser.error("--nsys requires --h2d-pair")
    if args.h2d_pair and args.methods not in (None, ["dhisparse"]):
        parser.error("--h2d-pair compares DSPARK dhisparse with itself")
    args.methods = args.methods or (
        ["dhisparse"] if args.h2d_pair else ["hisparse", "dhisparse"]
    )
    args.input_lens = args.input_lens or (
        [32768] if args.h2d_pair else list(INPUT_LENS)
    )
    args.concurrencies = args.concurrencies or (
        [256] if args.h2d_pair else list(CONCURRENCIES)
    )
    cfg = load_config(
        args.config
        or HERE / ("h2d.example.json" if args.h2d_pair else "overall.example.json")
    )
    if "dhisparse" in args.methods and (cfg["w"] + 1) * cfg["top_k"] > min(
        8192, cfg["device_buffer_size"]
    ):
        parser.error(
            "Fused swap-in requires (W + 1) * top_k <= min(8192, device_buffer_size)"
        )
    data_dir, output = (args.sharegpt or args.data_dir).resolve(), args.output.resolve()
    cells = matrix(args.methods, args.input_lens, args.concurrencies)
    if args.sharegpt:
        cells = [cell | {"dataset": "sharegpt"} for cell in cells]
    if args.h2d_pair:
        if not 1 <= args.trace_steps <= 256 or args.trace_skip < 0:
            parser.error("Require 1..256 trace steps and a nonnegative skip")
        if args.trace_batch_size is not None and not 1 <= args.trace_batch_size <= min(
            args.concurrencies
        ):
            parser.error(
                "The matched decode batch size must fit every selected concurrency"
            )
        cells = [
            cell
            | {
                "force_h2d_each_step": force,
                "nsys": args.nsys,
                "trace_steps": args.trace_steps,
                "trace_skip": args.trace_skip,
                "trace_batch_size": args.trace_batch_size or cell["max_concurrency"],
            }
            for cell in cells
            for force in (False, True)
        ]
    preflight = source_preflight()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preflight": preflight,
                    "config": cfg,
                    "cells": [
                        cell
                        | {
                            "commands": commands(
                                cfg,
                                cell,
                                data_dir,
                                cell_directory(output, cell) / "attempt-0001",
                            )[0],
                            "decode_trace_config": trace_config(
                                cell, cell_directory(output, cell) / "attempt-0001"
                            ),
                        }
                        for cell in cells
                    ],
                },
                indent=2,
            )
        )
        return
    if "lagom" in args.methods and not preflight["lagom_pd_layout_ready"]:
        raise RuntimeError(
            "This checkout lacks Lagom PD state-layout compatibility. "
            "Use matching lagom/perf branches with the prefill state-width option "
            "and Mooncake layout validation; see benchmark/lagom/OVERALL.md."
        )
    if set(args.methods) & {"dhisparse", "lagom"} and not preflight["oneshot_source_check_passed"]:
        raise RuntimeError("DHisparse oneshot verify source check failed; inspect this checkout")
    manifests = {}
    if args.sharegpt:
        with data_dir.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        manifests["sharegpt_sha256"] = digest.hexdigest()
    else:
        for path in sorted(data_dir.glob("*/manifest.json")):
            m = load_manifest(path)
            length = m["input_len"]
            if path.parent.name != str(length) or m["seed"] != cfg["seed"]:
                raise ValueError(f"Input manifest length/seed mismatch: {path}")
            if length in args.input_lens and len(m["order"]) < 4 * max(
                args.concurrencies
            ):
                raise ValueError(f"Not enough prepared requests: {path}")
            manifests[str(length)] = sha(m)
        if any(str(length) not in manifests for length in args.input_lens):
            raise ValueError(
                "Prepare a token-bank manifest for every selected input length"
            )
    current_provenance = provenance()
    protocol = {
        "source": {
            **{
                key: current_provenance[key]
                for key in ("git_head", "git_status", "git_diff_sha256")
            },
            "harness_sha256": sha(
                {
                    name: (HERE / name).read_text()
                    for name in ("run_overall.py", "longbench_v2_fixed.py")
                }
            ),
        },
        "config": cfg,
        "data_dir": str(data_dir),
        "data_manifests": manifests,
        "h2d_pair": args.h2d_pair,
        "nsys": args.nsys,
        "trace_batch_size": args.trace_batch_size,
        "trace_steps": args.trace_steps,
        "trace_skip": args.trace_skip,
        "output_len": OUTPUT_LEN,
        "requests_per_concurrency": 4,
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
                cell_dir = cell_directory(output, cell)
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
            if args.h2d_pair and not failed:
                for baseline, force in zip(cells[::2], cells[1::2]):
                    attempts = [
                        sorted(cell_directory(output, c).glob("attempt-*"))[-1]
                        for c in (baseline, force)
                    ]
                    comparison = compare_cells(
                        *(json.loads((d / "h2d.json").read_text()) for d in attempts)
                    )
                    save(attempts[1] / "comparison.json", comparison)
                    print(canonical(comparison), flush=True)
                    failed |= not comparison["passed"]
        finally:
            summary(output)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    main()
