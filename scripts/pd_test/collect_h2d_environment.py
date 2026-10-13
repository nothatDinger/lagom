"""Record non-secret versions, GPU/CPU topology and selected server settings."""

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path


def command(args):
    try:
        completed = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return dict(
            argv=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=args, error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--decode-url", default="http://127.0.0.1:30000")
    opts = parser.parse_args()
    report = dict(
        commands=[
            command(cmd)
            for cmd in [
                ["git", "rev-parse", "HEAD"],
                ["git", "status", "--short"],
                ["nvidia-smi", "topo", "-m"],
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
                    "--format=csv",
                ],
                ["nvidia-smi", "-q"],
                ["lscpu"],
                ["numactl", "--hardware"],
                ["nsys", "--version"],
                ["nvcc", "--version"],
            ]
        ],
        env={
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "NCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "SGLANG_HISPARSE_H2D_PROBE_CAPACITY",
                "SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH",
                "SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY",
            ]
        },
    )
    try:
        with urllib.request.urlopen(
            opts.decode_url.rstrip("/") + "/get_server_info", timeout=10
        ) as response:
            info = json.load(response)
        # Avoid exporting credentials or unrelated application settings.
        args = info.get("server_args", {})
        report["server_args"] = {
            key: args.get(key)
            for key in [
                "model_path",
                "tp_size",
                "dp_size",
                "pp_size",
                "enable_hisparse",
                "hisparse_config",
                "speculative_algorithm",
                "speculative_dspark_block_size",
                "disable_cuda_graph",
                "cuda_graph_max_bs_decode",
                "disaggregation_mode",
                "mem_fraction_static",
                "attention_backend",
                "moe_runner_backend",
            ]
        }
    except (OSError, ValueError) as exc:
        report["server_info_error"] = str(exc)
    opts.output_file.parent.mkdir(parents=True, exist_ok=True)
    opts.output_file.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
