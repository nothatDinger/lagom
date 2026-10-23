"""Plan, preflight and run the Fig.11-shaped SGLang performance sweep."""

import argparse
import asyncio
import importlib.util
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import distributions
from pathlib import Path

from common import (
    HERE,
    ROOT,
    command_output,
    digest,
    load_config,
    matrix,
    point_name,
    read_json,
    server_command,
    sha256,
    summarize,
    write_json,
)
from prepare import tokenizer_fingerprint


def resolve_checkpoint(reference):
    path = Path(reference)
    if not path.is_dir():
        if path.is_absolute():
            raise FileNotFoundError(reference)
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(reference, local_files_only=True))
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json: {path}")
    indices = list(path.glob("*.index.json"))
    shards = set()
    for index in indices:
        shards.update(read_json(index).get("weight_map", {}).values())
    if shards:
        missing = [s for s in shards if not (path / s).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing weight shards in {path}: {missing[:5]}")
    elif not list(path.glob("*.safetensors")) and not list(
        path.glob("pytorch_model*.bin")
    ):
        raise FileNotFoundError(f"No model weights found in {path}")
    return str(path.resolve())


def checkpoint_metadata(path):
    path = Path(path)
    weights = sorted(path.glob("*.safetensors")) + sorted(
        path.glob("pytorch_model*.bin")
    )
    return {
        "resolved_path": str(path),
        "metadata_sha256": {f.name: sha256(f) for f in path.glob("*.json")},
        "weight_files": {
            f.name: {"size_bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns}
            for f in weights
        },
    }


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_diff": command_output(["git", "-C", str(ROOT), "diff", "HEAD"]),
        "git_status": command_output(["git", "-C", str(ROOT), "status", "--short"]),
        "packages": sorted((d.metadata["Name"], d.version) for d in distributions()),
        "gpu": command_output(["nvidia-smi", "-q"]),
        "gpu_identity": command_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "cpu": command_output(["lscpu"]),
        "memory": (
            Path("/proc/meminfo").read_text()
            if Path("/proc/meminfo").exists()
            else "unavailable"
        ),
    }


def preflight(config, models, methods, data_dir):
    errors, warnings, resolved = [], [], {}
    for package in (
        "aiohttp",
        "numpy",
        "msgspec",
        "zmq",
        "transformers",
        "torch",
        "sgl_kernel",
    ):
        if importlib.util.find_spec(package) is None:
            errors.append(f"Missing Python module: {package}")
    if not Path(config["dataset"]).is_file():
        errors.append(f"ShareGPT file missing: {config['dataset']}")
        source_hash = None
    else:
        source_hash = sha256(config["dataset"])
    for name in models:
        model = config["models"][name]
        record = {}
        resolved[name] = record
        for key in ["path", "draft"] if "oasiskv" in methods else ["path"]:
            try:
                record[key] = resolve_checkpoint(model[key])
                record[key + "_metadata"] = checkpoint_metadata(record[key])
            except Exception as error:
                errors.append(f"{name}/{key}: {error}")
        try:
            manifest = read_json(data_dir / name / "manifest.json")
            if (
                manifest["dataset_sha256"] != source_hash
                or manifest["seed"] != config["seed"]
            ):
                raise ValueError("Prepared data has a different source hash or seed")
            if (
                manifest["warmup_requests"] != config["warmup_requests"]
                or manifest["count"]
                < config["num_requests"] + config["warmup_requests"]
            ):
                raise ValueError(
                    "Prepared data has insufficient requests or different warmup split"
                )
            if manifest["max_input_length"] < max(config["input_lengths"]):
                raise ValueError("Prepared prompts are too short")
            if manifest["tokenizer_files"] != tokenizer_fingerprint(model["path"]):
                raise ValueError("Tokenizer changed since data preparation")
            if manifest["prompts_sha256"] != sha256(data_dir / name / "prompts.npy"):
                raise ValueError("Prepared prompt checksum mismatch")
            import numpy as np

            prompts = np.load(
                data_dir / name / "prompts.npy", mmap_mode="r", allow_pickle=False
            )
            if prompts.dtype != np.int32 or prompts.shape != (
                manifest["count"],
                manifest["max_input_length"],
            ):
                raise ValueError("Prepared prompt shape/dtype mismatch")
            record["data_manifest"] = manifest
        except Exception as error:
            errors.append(f"{name}/prepared data: {error}")
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,torch; print(json.dumps([{'name':torch.cuda.get_device_properties(i).name,'bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]))",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": model["cuda_visible_devices"]},
            text=True,
            capture_output=True,
            timeout=60,
        )
        try:
            devices = json.loads(probe.stdout)
            record["visible_gpus"] = devices
            if len(devices) != model["tp"] or any(
                "H100" not in d["name"]
                or not 75 * 1024**3 <= d["bytes"] <= 82 * 1024**3
                for d in devices
            ):
                errors.append(
                    f"{name}: expected {model['tp']} H100 80GB GPUs; observed {devices}"
                )
        except (ValueError, TypeError):
            errors.append(f"{name}/CUDA probe failed: {probe.stderr[-2000:]}")
    warnings.append(
        "Verify paper host topology manually: 2x Xeon 8480C, 2 TB RAM, NVLink/NVSwitch and PCIe Gen5 x16; captured in environment.json."
    )
    warnings.append(
        "Paper Fig.11 fetch cap is not explicit. max_new_blocks=null is uncapped in this port; capped prefetch does not cap real-query refill."
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "resolved": resolved,
    }


def ensure_port_free(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def validate_server_info(info, config, point):
    """Check the scheduler's resolved settings, including the lookahead-only gate."""
    states = info.get("internal_states") or [info]
    for state in states:
        expected = {
            "attention_backend": "oasiskv" if point["method"] == "oasiskv" else "fa3",
            "tp_size": config["models"][point["model"]]["tp"],
            "disable_radix_cache": True,
            "context_length": config["context_length"],
            "max_running_requests": point["concurrency"],
        }
        if point["method"] == "oasiskv":
            expected.update(is_oasiskv_lookahead=True, speculative_num_draft_tokens=2)
        elif state.get("speculative_algorithm") is not None:
            raise ValueError("Dense baseline unexpectedly enables speculative decoding")
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Resolved server {key}={state.get(key)!r}; expected {value!r}"
                )
        decode_graph = state.get("cuda_graph_config", {}).get("decode", {})
        if (
            state.get("disable_cuda_graph", False)
            or decode_graph.get("backend") != "full"
            or (decode_graph.get("max_bs") or 0) < point["concurrency"]
        ):
            raise ValueError(
                "Resolved decode CUDA graph must be full and cover the requested concurrency; "
                f"observed {decode_graph!r}"
            )


async def wait_ready(session, url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {process.returncode}")
        try:
            async with session.get(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(1)
    raise TimeoutError("Server startup timed out")


async def stop_server(process):
    # Terminate only the process group created for this point, including TP workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


async def run_point(config, point, folder, data_dir, resolved, port, fingerprint):
    import aiohttp
    import numpy as np
    from client import ForwardCollector, generate, workload

    folder.mkdir(parents=True, exist_ok=True)
    result = {**point, "fingerprint": fingerprint, "status": "failed"}
    process = collector = None
    model = config["models"][point["model"]]
    local = resolved[point["model"]]
    prompts = np.load(
        data_dir / point["model"] / "prompts.npy", mmap_mode="r", allow_pickle=False
    )
    # Short IPC names avoid the Unix-domain socket path limit.
    with (
        tempfile.TemporaryDirectory(prefix="okv-") as ipc_dir,
        open(folder / "server.log", "w") as log,
    ):
        endpoint = "ipc://" + ipc_dir + "/fpm"
        command = server_command(config, point, port, endpoint, local.get("draft"))
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": model["cuda_visible_devices"],
            "PYTHONPATH": str(ROOT / "python")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        write_json(
            folder / "launch.json",
            {
                "argv": command,
                "cuda_visible_devices": model["cuda_visible_devices"],
                "model_metadata": local,
                "point": point,
            },
        )
        try:
            ensure_port_free(port)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            collector = ForwardCollector(endpoint, folder / "forward.jsonl")
            timeout = aiohttp.ClientTimeout(total=config["request_timeout_s"])
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=0),
                read_bufsize=10 * 1024**2,
            ) as session:
                url = f"http://127.0.0.1:{port}"
                await wait_ready(session, url, process, config["startup_timeout_s"])
                async with session.get(url + "/server_info", timeout=30) as response:
                    response.raise_for_status()
                    info = await response.json()
                    write_json(folder / "server_info.json", info)
                    validate_server_info(info, config, point)
                warmups = []
                for i in range(config["warmup_requests"]):
                    row = await generate(
                        session,
                        url,
                        prompts[i, : point["input_length"]].tolist(),
                        config["warmup_output_length"],
                        f"warmup-{i}",
                    )
                    warmups.append(row)
                    if not row["success"]:
                        raise RuntimeError(f"Warmup failed: {row.get('error')}")
                write_json(folder / "warmup.json", warmups)
                await collector.wait_idle()
                collector.phase = "measurement"
                offset = config["warmup_requests"]
                requests, start, end = await asyncio.wait_for(
                    workload(
                        session,
                        url,
                        prompts[offset : offset + point["num_requests"]],
                        point["input_length"],
                        point["output_length"],
                        point["concurrency"],
                        folder / "requests.jsonl",
                    ),
                    timeout=config["point_timeout_s"],
                )
                await collector.wait_idle()
                collector.phase = "done"
                metrics = summarize(requests, collector.rows, start, end)
                result.update(metrics)
                reasons = []
                if metrics["failed_requests"]:
                    reasons.append("request failures or token-length mismatch")
                if (
                    metrics["decode_forward_samples"] == 0
                    or metrics["fpm_sequence_gaps"]
                    or collector.errors
                ):
                    reasons.append("missing/invalid forward telemetry")
                if metrics["tpot_samples"] != point["num_requests"]:
                    reasons.append(
                        "first stream chunk is not exactly one token; TPOT not comparable"
                    )
                if (
                    metrics["running_batch_max"]
                    and metrics["running_batch_max"] > point["concurrency"]
                ):
                    reasons.append(
                        "server running batch exceeds configured concurrency"
                    )
                if point["method"] == "oasiskv" and any(
                    r.get("spec_accept_length") is not None
                    and r["spec_accept_length"] > 1.001
                    for r in requests
                ):
                    reasons.append(
                        "draft acceptance observed; lookahead must force rejection"
                    )
                result.update(
                    status="invalid" if reasons else "ok",
                    invalid_reasons=reasons,
                    telemetry_errors=collector.errors,
                )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                if process is not None:
                    await stop_server(process)
            finally:
                try:
                    if collector is not None:
                        await collector.close()
                finally:
                    write_json(folder / "result.json", result)
    return result


async def run_sweep(args, config, points, checked):
    env_record = environment()
    identity = {
        "config": config,
        "protocol": "smoke" if args.smoke else "fig11-sglang-port",
        "resolved": checked["resolved"],
        "git_head": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_diff": command_output(["git", "-C", str(ROOT), "diff", "HEAD"]),
        "benchmark_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "software": {k: env_record[k] for k in ("python", "packages", "gpu_identity")},
    }
    fingerprint = digest(identity)
    manifest_path = args.output / "run.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path)["fingerprint"] != fingerprint:
            raise ValueError(
                "Result directory exists or fingerprint changed; use a new --output directory"
            )
    else:
        write_json(manifest_path, {"fingerprint": fingerprint, **identity})
        write_json(args.output / "environment.json", env_record)
    failed = 0
    for point in points:
        folder = args.output / point_name(point)
        result_path = folder / "result.json"
        if args.resume and result_path.exists():
            previous = read_json(result_path)
            if previous.get("fingerprint") != fingerprint:
                raise ValueError(f"Point fingerprint mismatch: {folder}")
            if previous["status"] == "ok":
                print(f"SKIP {folder.name}", flush=True)
                continue
        # Keep failed attempts rather than overwrite evidence on resume.
        if folder.exists():
            folder.rename(folder.with_name(folder.name + f".attempt-{time.time_ns()}"))
        print(f"RUN {folder.name}", flush=True)
        result = await run_point(
            config,
            point,
            folder,
            args.data,
            checked["resolved"],
            args.port,
            fingerprint,
        )
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in (
                        "status",
                        "output_tps",
                        "running_batch_mean",
                        "tpot_mean_ms",
                        "error",
                        "invalid_reasons",
                    )
                }
            ),
            flush=True,
        )
        failed += result["status"] != "ok"
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "preflight", "run"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--data", type=Path, default=Path("/tmp/oasiskv-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=["dense", "oasiskv"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    config = load_config(args.config)
    models = args.models or list(config["models"])
    methods = args.methods or config["methods"]
    points = list(matrix(config, models, methods, args.smoke))
    if args.action == "plan":
        planned = [
            {
                **point,
                "command": shlex.join(
                    server_command(config, point, args.port, "ipc:///tmp/oasiskv-fpm")
                ),
            }
            for point in points
        ]
        print(json.dumps({"count": len(points), "points": planned}, indent=2))
        return 0
    checked = preflight(config, models, methods, args.data)
    print(json.dumps(checked, indent=2))
    if args.action == "preflight":
        if args.output:
            write_json(args.output / "preflight.json", checked)
            write_json(args.output / "environment.json", environment())
        return 0 if checked["ok"] else 1
    if args.output is None:
        parser.error("run requires --output")
    if not checked["ok"]:
        return 1
    return asyncio.run(run_sweep(args, config, points, checked))


if __name__ == "__main__":
    raise SystemExit(main())
