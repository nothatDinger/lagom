"""Read current GPU/process NUMA evidence, without loading a model or modifying it."""

import argparse
import json
from pathlib import Path

from collect_h2d_environment import command

from sglang.benchmark.hisparse_h2d_diagnostics import process_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", nargs="+", type=int, default=[])
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    opts = parser.parse_args()
    if opts.output_file.exists():
        parser.error("Refusing to replace an existing output file")
    buffers = {pid: [] for pid in opts.pid}
    identities = []
    if opts.probe_dir:
        for path in sorted(opts.probe_dir.rglob("*.h2d.json")):
            probe = json.loads(path.read_text())
            pid = int(probe["pid"])
            saved = probe.get("host_numa_snapshot", {})
            buffers.setdefault(pid, []).extend(
                {k: b[k] for k in ("name", "address", "nbytes")}
                for b in saved.get("buffers", [])
            )
            identities.append(
                dict(
                    probe_file=str(path),
                    pid=pid,
                    saved_start_ticks=saved.get("start_ticks"),
                    cuda_device=probe.get("device"),
                    cuda_visible_devices=probe.get("cuda_visible_devices"),
                    gpu_uuid=probe.get("gpu_uuid"),
                )
            )
    if not buffers:
        parser.error("Provide --pid or a --probe-dir containing .h2d.json files")
    snapshots = []
    for pid in sorted(buffers):
        snapshot = process_snapshot(pid)
        historical = [i for i in identities if i["pid"] == pid]
        snapshot["probe_identity_status"] = (
            "verified_start_ticks"
            if historical
            and all(
                i["saved_start_ticks"] is not None
                and i["saved_start_ticks"] == snapshot["start_ticks"]
                for i in historical
            )
            and snapshot["pid_stable_during_snapshot"]
            else "unverified_or_stale"
        )
        # Addresses from an old process must never be interpreted as current KV.
        if snapshot["probe_identity_status"] == "verified_start_ticks":
            detailed = process_snapshot(pid, buffers[pid])
            if (
                detailed["pid_stable_during_snapshot"]
                and detailed["start_ticks"] == snapshot["start_ticks"]
            ):
                snapshot["buffer_snapshot"] = detailed
            else:
                snapshot["probe_identity_status"] = "unverified_or_stale"
        snapshots.append(snapshot)
    report = dict(
        schema_version=1,
        probe_identities=identities,
        processes=snapshots,
        commands=[
            command(cmd)
            for cmd in [
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,pci.bus_id,name",
                    "--format=csv",
                ],
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                    "--format=csv",
                ],
                ["nvidia-smi", "topo", "-m"],
                ["numactl", "--hardware"],
            ]
        ],
        interpretation=[
            "Process-wide node counts do not identify KV buffers in old captures.",
            "Mems_allowed_list is a permitted-node set, not an actual allocation policy.",
            "VMA node counts refer to the entire VMA; never prorate them to a buffer subrange.",
            "A historical PID without matching start_ticks is unverified; confirm live workers in the GPU inventory.",
            "Permission errors are retained; do not treat missing maps as local memory.",
        ],
    )
    opts.output_file.parent.mkdir(parents=True, exist_ok=True)
    opts.output_file.write_text(json.dumps(report, indent=2) + "\n")
    print(opts.output_file)


if __name__ == "__main__":
    main()
