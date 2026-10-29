"""Run the isolated V4/DSPARK trace experiment on four local H100 GPUs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = "/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/"
DRAFT = "/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-DSpark"
DATASET = "/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def checkpoint_gamma(path):
    # Matches parse_dspark_draft_config: DSPARK's G is the checkpoint's
    # block_size; target verification has G+1 rows, including the anchor.
    config = json.loads((Path(path) / "config.json").read_text())
    gamma = config.get("dspark_block_size")
    if gamma is None:
        gamma = config.get("dflash_config", {}).get(
            "block_size", config.get("block_size")
        )
    if type(gamma) is not int or gamma < 1:
        raise ValueError(
            "Cannot resolve checkpoint proposal length; inspect draft config.json"
        )
    return gamma


def checkpoint_topk(path):
    config = json.loads((Path(path) / "config.json").read_text())
    text_config = config.get("text_config", config)
    # DeepSeekV4Config's default; the server uses the model index_topk rather
    # than treating HiSparse's compatibility field as a model override.
    value = text_config.get("index_topk", 512)
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid target index_topk")
    return value


def real_candidates(args):
    value = getattr(args, "verify_candidates", None)
    return value if value is not None else getattr(args, "verify_width", 6) - 1


def proposal_gamma(args, trained_gamma):
    mode = getattr(args, "collection_mode", getattr(args, "mode", "paired"))
    if mode in ("untraced", "full-control") or not hasattr(args, "probes"):
        return trained_gamma
    return max(trained_gamma, real_candidates(args) + max(args.probes))


def resident_verify_width(args, trained_gamma):
    mode = getattr(args, "mode", "paired")
    if mode in ("untraced", "full-control"):
        return trained_gamma + 1
    width = real_candidates(args) + 1
    if mode == "experiment" and not getattr(args, "skip_full_control", False):
        width = max(width, trained_gamma + 1)
    return width


def server_command(args, gamma):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.target,
        "--speculative-draft-model-path",
        args.draft,
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-dspark-block-size",
        str(proposal_gamma(args, gamma)),
        "--speculative-num-draft-tokens",
        str(proposal_gamma(args, gamma) + 1),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp",
        "4",
        "--attention-backend",
        "dsv4",
        "--moe-runner-backend",
        getattr(args, "moe_runner_backend", "flashinfer_mxfp4"),
        "--enable-hisparse",
        "--hisparse-config",
        json.dumps(
            {
                "top_k": args.top_k,
                "device_buffer_size": args.resident_units,
                "host_to_device_ratio": getattr(args, "host_to_device_ratio", 1),
            }
        ),
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--max-running-requests",
        "1",
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.context_length),
        "--chunked-prefill-size",
        "4096",
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--random-seed",
        str(args.seed),
        "--watchdog-timeout",
        str(getattr(args, "watchdog_timeout", args.request_timeout)),
    ]


def validate_args(args, gamma):
    if (
        getattr(args, "mode", None) == "untraced"
        and getattr(args, "topk_order", "native") != "native"
    ):
        raise ValueError(
            "Untraced mode has no ordering hook; use replay-check or control"
        )
    v = real_candidates(args)
    if v < 1 or not args.probes or any(p < 1 for p in args.probes):
        raise ValueError("Need W >= 2 and positive probe lengths")
    runtime_gamma = proposal_gamma(args, gamma)
    if len(set(args.probes)) != len(args.probes):
        raise ValueError("Duplicate probe lengths")
    if min(args.input_lengths) < 1 or len(set(args.input_lengths)) != len(
        args.input_lengths
    ):
        raise ValueError("Input lengths must be positive and unique")
    if (
        args.context_length
        < max(args.input_lengths) + args.output_length + runtime_gamma + 1
    ):
        raise ValueError("Context length must cover prompt, output and proposal window")
    if (
        min(args.requests, args.output_length, args.snapshot_gib) <= 0
        or args.max_windows < 2
    ):
        raise ValueError("Need positive requests/output/snapshot size and >=2 windows")
    baseline_width = resident_verify_width(args, gamma)
    if args.resident_units < baseline_width * args.top_k:
        raise ValueError(
            f"Real verify requires capacity >= W*index_topk = {baseline_width * args.top_k} C4 units"
        )
    if not 0 < args.mem_fraction < 1 or args.prefetch_units < 0:
        raise ValueError("Invalid memory fraction or prefetch cap")
    if len(args.gpus.split(",")) != 4:
        raise ValueError("This recipe requires four GPU IDs")
    ratio = getattr(args, "host_to_device_ratio", 1)
    if ratio < 1:
        raise ValueError("host_to_device_ratio must be >= 1")
    # V4 stages C4 to host after the LAST prefill chunk. All prompt C4s
    # coexist on device until then; chunking only reduces the SWA peak.
    prefill_capacity = (args.context_length // 256 * 256) // ratio
    if max(args.input_lengths) + 256 >= prefill_capacity:
        raise ValueError(
            "Prompt cannot fit the global prefill C4 pool (including scratch/page "
            "headroom). Use --host-to-device-ratio 1 or increase --context-length; "
            "--resident-units controls the separate per-request decode working set."
        )
    return v


def preflight(args):
    for path in (args.target, args.draft):
        root = Path(path)
        if not (root / "config.json").is_file():
            raise FileNotFoundError(root / "config.json")
        indices = list(root.glob("*.index.json"))
        for index in indices:
            for shard in set(
                json.loads(index.read_text()).get("weight_map", {}).values()
            ):
                if not (root / shard).is_file():
                    raise FileNotFoundError(root / shard)
        if not indices and not (
            list(root.glob("*.safetensors")) or list(root.glob("*.bin"))
        ):
            raise FileNotFoundError(f"No checkpoint weights in {root}")
    if not Path(args.dataset).is_file():
        raise FileNotFoundError(args.dataset)
    # Run in a child so this controller never creates a CUDA context that
    # consumes memory while the TP workers load their weights.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch,json; print(json.dumps([{'name':torch.cuda.get_device_name(i),'memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]))",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus},
        check=True,
        text=True,
        capture_output=True,
    )
    devices = json.loads(probe.stdout.strip().splitlines()[-1])
    if len(devices) != 4 or any("H100" not in d["name"] for d in devices):
        raise ValueError(f"Expected four H100s, found {devices}")
    return devices


def prepare_prompts(args, output):
    import numpy as np
    from transformers import AutoTokenizer

    # Reuse the performance harness's documented ShareGPT packing policy.
    sys.path.insert(0, str(HERE.parent))
    from prepare import packed_prompts, tokenizer_fingerprint

    tokenizer = AutoTokenizer.from_pretrained(
        args.target, local_files_only=True, trust_remote_code=False
    )
    records = json.loads(Path(args.dataset).read_text())
    if not isinstance(records, list):
        raise TypeError("ShareGPT must contain a JSON array")
    prompts = np.asarray(
        list(
            packed_prompts(
                records, tokenizer, args.requests, max(args.input_lengths), args.seed
            )
        ),
        dtype=np.int32,
    )
    np.save(output / "prompts.npy", prompts)
    write_json(
        output / "prompts.json",
        {
            "dataset": args.dataset,
            "dataset_sha256": digest(args.dataset),
            "tokenizer_files": tokenizer_fingerprint(args.target),
            "prompts_sha256": digest(output / "prompts.npy"),
            "seed": args.seed,
            "construction": "packed ShareGPT all turns; shorter inputs are prefixes; no chat template or BOS",
        },
    )
    return prompts


def request(url, payload=None, timeout=10):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited ({process.returncode}); see server.log")
        try:
            # /health may return an empty body; do not require JSON.
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("Server startup timeout; see server.log")


def process_table():
    """Linux process identity plus ancestry; start time guards against PID reuse."""
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm may contain spaces and parentheses.
            fields = path.read_text().rsplit(")", 1)[1].split()
            result[int(path.parent.name)] = (
                int(fields[19]),  # starttime, field 22
                int(fields[1]),  # ppid
                fields[0],  # state
            )
        except (OSError, ValueError, IndexError):
            continue  # Process exited during the scan.
    return result


class ServerProcessTree:
    """Own one server in this sequential CLI, including setsid()/orphan workers.

    Enable subreaping BEFORE spawn: scanning at teardown alone loses children
    already reparented to PID 1. Existing children (and their descendants) are
    excluded. The CLI must not launch unrelated subprocesses during this scope.
    """

    def __enter__(self):
        if sys.platform != "linux":
            raise RuntimeError("The V4 trace runner requires Linux process supervision")
        self.libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
        self.previous_subreaper = previous.value
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
        self.excluded = {
            pid: row[0] for pid, row in process_table().items() if row[1] == os.getpid()
        }
        self.owned = {}
        self.process = None
        self.stop_attempted = False
        return self

    def attach(self, process):
        self.process = process
        self.scan()

    def scan(self):
        table = process_table()
        # Remember descendants of pre-existing children before they can orphan.
        for identities in (self.excluded, self.owned):
            parents = {
                pid
                for pid, start in identities.items()
                if pid in table and table[pid][0] == start
            }
            if identities is self.owned:
                parents.update(
                    pid
                    for pid, row in table.items()
                    if row[1] == os.getpid() and self.excluded.get(pid) != row[0]
                )
            while parents:
                children = {
                    pid
                    for pid, row in table.items()
                    if row[1] in parents and identities.get(pid) != row[0]
                }
                for pid in parents | children:
                    identities[pid] = table[pid][0]
                parents = children
        return {pid: row for pid, row in table.items() if self.owned.get(pid) == row[0]}

    def stop(self, term_timeout=30, kill_timeout=30):
        self.stop_attempted = True
        if self.process is None:
            return
        for sig, timeout in (
            (signal.SIGTERM, term_timeout),
            (signal.SIGKILL, kill_timeout),
        ):
            deadline = time.monotonic() + timeout
            signalled = set()
            while True:
                self.process.poll()  # Reap the wrapper separately from adopted workers.
                live = {}
                for pid, (start, parent, state) in self.scan().items():
                    if (
                        state == "Z"
                        and parent == os.getpid()
                        and pid != self.process.pid
                    ):
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                    elif state != "Z":
                        live[pid] = start
                        if (pid, start) not in signalled:
                            try:
                                os.kill(pid, sig)
                            except ProcessLookupError:
                                pass
                            signalled.add((pid, start))
                if not live:
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        raise RuntimeError(f"Server workers survived teardown: {sorted(live)}")

    def __exit__(self, exc_type, exc, tb):
        try:
            if not self.stop_attempted:
                self.stop()
        finally:
            if self.libc.prctl(36, self.previous_subreaper, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "Restoring child subreaper failed")


def run_environment(args, config_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "SGLANG_RAGGED_VERIFY_MODE": "compact",
        "SGLANG_OPT_USE_ONLINE_COMPRESS": "0",
        "SGLANG_SIMULATE_ACC_LEN": "0",
        "SGLANG_DEBUG_HISPARSE_SKIP_IO": "0",
        "SGLANG_DSV4_FP4_DEQUANT": "0",
        "SGLANG_OASISKV_TRACE_DIAGNOSTICS": "1" if args.diagnostics else "0",
        "PYTHONUNBUFFERED": "1",
    }
    # Crucially remove any inherited trace configuration in the no-hook arm.
    env.pop("SGLANG_OASISKV_TRACE_CONFIG", None)
    if config_path is not None:
        env["SGLANG_OASISKV_TRACE_CONFIG"] = str(config_path)
    return env


def mode_plan(args):
    """One controller, separate servers; use a distinct declared port per mode."""
    modes = (
        ["control", "probes"] + ([] if args.skip_full_control else ["full-control"])
        if args.mode == "experiment"
        else ["probes"] if args.mode == "paired" else [args.mode]
    )
    if not 1 <= args.port <= 65536 - len(modes):
        raise ValueError("Base --port and all mode ports must be within 1..65535")
    return [
        (
            mode,
            argparse.Namespace(
                **(vars(args) | {"port": args.port + offset, "collection_mode": mode})
            ),
        )
        for offset, mode in enumerate(modes)
    ]


def check_listen_port(port):
    # Match the server's reuse semantics: TIME_WAIT after an old HTTP connection
    # must not be mistaken for a live listener. No SO_REUSEPORT: fail for a real
    # competing server. Closing before spawn cannot prevent an external race.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)


def run_one(args, directory, config, command, prompts):
    directory.mkdir()
    config_path = directory / "config.json"
    write_json(config_path, config)
    env = run_environment(args, config_path if config is not None else None)
    write_json(
        directory / "launch.json",
        {
            "command": command,
            "environment": {
                k: env.get(k)
                for k in (
                    "CUDA_VISIBLE_DEVICES",
                    "SGLANG_RAGGED_VERIFY_MODE",
                    "SGLANG_OPT_USE_ONLINE_COMPRESS",
                    "SGLANG_SIMULATE_ACC_LEN",
                    "SGLANG_DEBUG_HISPARSE_SKIP_IO",
                    "SGLANG_OASISKV_TRACE_CONFIG",
                    "SGLANG_OASISKV_TRACE_DIAGNOSTICS",
                    "SGLANG_DSV4_FP4_DEQUANT",
                    "CUDA_HOME",
                    "NVCC_PREPEND_FLAGS",
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
                )
            },
        },
    )
    url = f"http://127.0.0.1:{args.port}"
    check_listen_port(args.port)
    print(f"Starting {directory.name} on port {args.port}", flush=True)
    with (directory / "server.log").open("w") as log, ServerProcessTree() as tree:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        tree.attach(process)
        try:
            wait_ready(process, url, args.startup_timeout)
            write_json(directory / "server_info.json", request(url + "/server_info"))
            with (directory / "outputs.jsonl").open("x") as stream:
                for length in args.input_lengths:
                    for i, prompt in enumerate(prompts):
                        rid = f"trace-i{length}-r{i}"
                        write_json(
                            directory / "request-progress.json",
                            {
                                "request_id": rid,
                                "input_length": length,
                                "status": "submitted",
                                "submitted_unix_s": time.time(),
                            },
                        )
                        print(
                            f"{directory.name}: submitting {rid}, input={length}",
                            flush=True,
                        )
                        result = request(
                            url + "/generate",
                            {
                                "rid": rid,
                                "input_ids": prompt[:length].tolist(),
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": args.output_length,
                                    "ignore_eos": True,
                                },
                            },
                            timeout=args.request_timeout,
                        )
                        if (
                            not isinstance(result.get("output_ids"), list)
                            or not result["output_ids"]
                        ):
                            raise RuntimeError(
                                f"Request {rid} has no token output: {result}"
                            )
                        stream.write(
                            json.dumps(
                                {"request_id": rid, "input_length": length, **result},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        stream.flush()
                        write_json(
                            directory / "request-progress.json",
                            {
                                "request_id": rid,
                                "input_length": length,
                                "status": "completed",
                                "output_tokens": len(result["output_ids"]),
                            },
                        )
                        print(
                            f"{directory.name}: {rid}, output={len(result['output_ids'])}",
                            flush=True,
                        )
        except Exception as error:
            write_json(
                directory / "failure.json",
                {
                    "error": f"{type(error).__name__}: {error}",
                    "unix_s": time.time(),
                },
            )
            raise
        finally:
            try:
                tree.stop()
            except Exception as error:
                write_json(
                    directory / "teardown-failure.json",
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "unix_s": time.time(),
                    },
                )
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Otherwise print the resolved plan only (no GPU/model load)",
    )
    widths = parser.add_mutually_exclusive_group()
    widths.add_argument(
        "--verify-width",
        type=int,
        default=6,
        help="Real target verify rows including anchor (W); probes are appended",
    )
    widths.add_argument(
        "--verify-candidates",
        type=int,
        help="Equivalent to W-1; retained for old experiment recipes",
    )
    parser.add_argument("--probes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=[
            "experiment",
            "paired",
            "untraced",
            "control",
            "probes",
            "full-control",
            "replay-check",
        ],
        default="experiment",
        help="paired uses one probes server and its snapshot baseline; experiment adds independent control servers",
    )
    parser.add_argument(
        "--require-cross-server-match",
        action="store_true",
        help="For experiment only: also require independent control/probes to match exactly",
    )
    parser.add_argument(
        "--topk-order",
        choices=["native", "logical", "logical-all"],
        default="native",
        help="Offline diagnostic: logical sorts verify only; logical-all also sorts target prefill/decode",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Log CPU admission/forward stages, including in untraced mode",
    )
    parser.add_argument(
        "--moe-runner-backend",
        choices=["flashinfer_mxfp4", "marlin", "humming"],
        default="flashinfer_mxfp4",
    )
    parser.add_argument(
        "--host-to-device-ratio",
        type=int,
        default=1,
        help="Global C4 prefill pool shrink ratio, distinct from resident-units",
    )
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument(
        "--snapshot-gib",
        type=float,
        default=32,
        help="Host snapshot limit PER TP rank; four ranks allocate independently",
    )
    parser.add_argument(
        "--resident-units",
        type=int,
        help="Persistent C4 capacity; default W*index_topk (also covers native full-control if requested)",
    )
    parser.add_argument(
        "--prefetch-units",
        type=int,
        default=256,
        help="Offline admission cap in C4 units, not an online setting",
    )
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--mem-fraction", type=float, default=0.9)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Base port; experiment uses base/base+1/base+2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=21600)
    parser.add_argument(
        "--watchdog-timeout",
        type=int,
        default=600,
        help="Scheduler forward watchdog; independent of client request timeout",
    )
    parser.add_argument(
        "--skip-full-control",
        action="store_true",
        help="Smoke runs only: omit the full-G acceptance reference",
    )
    args = parser.parse_args()
    if args.require_cross_server_match and args.mode != "experiment":
        parser.error("--require-cross-server-match requires --mode experiment")
    gamma = checkpoint_gamma(args.draft)
    args.top_k = checkpoint_topk(args.target)
    if args.resident_units is None:
        args.resident_units = resident_verify_width(args, gamma) * args.top_k
    v = validate_args(args, gamma)
    plan = mode_plan(args)
    print(
        f"mode={args.mode}; checkpoint_G={gamma}, proposal_G={proposal_gamma(args, gamma)}, W={v+1}, P={args.probes}, index_topk={args.top_k}, resident_C4={args.resident_units}; B=1, TP=4; offline quality, graphs disabled"
    )
    for mode, mode_args in plan:
        print(f"{mode}: {shlex.join(server_command(mode_args, gamma))}")
    if not args.execute:
        return
    hardware = preflight(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True)
    (output / "source.diff").write_text(diff)
    write_json(
        output / "manifest.json",
        {
            "arguments": {
                k: str(x) if isinstance(x, Path) else x for k, x in vars(args).items()
            },
            "commit": commit,
            "source_diff_sha256": digest(output / "source.diff"),
            "hardware": hardware,
            "checkpoint_config_sha256": {
                p: digest(Path(p) / "config.json") for p in (args.target, args.draft)
            },
            "gamma": gamma,
            "checkpoint_gamma": gamma,
            "proposal_gamma": proposal_gamma(args, gamma),
            "verify_width": v + 1,
            "max_target_width": v + 1 + max(args.probes),
            "verify_candidates": v,
            "timing_valid": False,
        },
    )
    prompts = prepare_prompts(args, output)
    modes = [mode for mode, _ in plan]
    for mode, mode_args in plan:
        directory = output / mode
        config = {
            "run_id": mode,
            "output": str(directory),
            "verify_candidates": gamma if mode == "full-control" else v,
            "checkpoint_gamma": gamma,
            "proposal_gamma": proposal_gamma(mode_args, gamma),
            "probe_workspace": "row-copy",
            "probes": args.probes if mode == "probes" else [],
            "shadows": mode in ("probes", "replay-check"),
            "topk_order": args.topk_order,
            "max_windows": args.max_windows,
            "snapshot_gib": args.snapshot_gib,
            "request_ids": [
                f"trace-i{length}-r{i}"
                for length in args.input_lengths
                for i in range(args.requests)
            ],
        }
        run_one(
            mode_args,
            directory,
            None if mode == "untraced" else config,
            server_command(mode_args, gamma),
            prompts,
        )
    if args.mode not in ("experiment", "paired"):
        print(
            f"Collected {args.mode}; no cross-process audit or coverage report: {output}"
        )
        return
    from audit import acceptance_summary

    write_json(
        output / "acceptance.json",
        {mode: acceptance_summary(output / mode) for mode in modes},
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "audit.py"),
            str(output / "probes"),
            "--scope",
            "within-server",
            "--output",
            str(output / "audit.json"),
        ],
        check=True,
    )
    if args.mode == "experiment":
        subprocess.run(
            [
                sys.executable,
                str(HERE / "audit.py"),
                str(output / "control"),
                str(output / "probes"),
                "--output",
                str(output / "cross-server-audit.json"),
            ]
            + ([] if args.require_cross_server_match else ["--diagnostic-only"]),
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "analyze.py"),
            str(output / "probes" / "trace.jsonl.gz"),
            "--max-prefetch-units",
            str(args.prefetch_units),
            "--audit",
            str(output / "audit.json"),
            "--output",
            str(output / "analysis"),
        ],
        check=True,
    )
    print(f"Completed quality experiment: {output}")


if __name__ == "__main__":
    main()
