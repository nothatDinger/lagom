"""CPU oracles and reporting for an opt-in dependent-read diagnostic."""

import hashlib
import json
import math
import statistics
from collections import Counter


def make_chains(order, chains):
    """Partition one address set into interleaved, disjoint rings.

    Each ring is traversed exactly once. The independent oracle is the sum of
    its node IDs, not a simulation using the generated successor table.
    """
    if chains <= 0 or not order or len(order) % chains:
        raise ValueError("Read count must be a positive multiple of chain count")
    if len(set(order)) != len(order) or min(order) < 0:
        raise ValueError("Node IDs must be distinct and nonnegative")
    if max(order) * len(order) >= 2**63:
        raise ValueError("Checksum exceeds signed int64")
    return dict(
        nodes=list(order),
        successors=list(order[chains:]) + list(order[:chains]),
        starts=list(order[:chains]),
        expected=[[order[c], sum(order[c::chains])] for c in range(chains)],
        steps=len(order) // chains,
        node_ids_sha256=hashlib.sha256(
            json.dumps(sorted(order), separators=(",", ":")).encode()
        ).hexdigest(),
    )


def check_result(rows, expected):
    """Check all chains, independent sums, and both sentinel rows."""
    if len(rows) != len(expected) + 2 or any(len(row) != 4 for row in rows):
        raise ValueError("Wrong result geometry")
    if rows[0] != [-1] * 4 or rows[-1] != [-1] * 4:
        raise ValueError("Read probe overwrote output guard rows")
    for chain, (actual, want) in enumerate(zip(rows[1:-1], expected)):
        if actual[:2] != want:
            raise ValueError(f"Chain {chain}: actual={actual[:2]}, expected={want}")
        if not 0 <= actual[2] < actual[3]:
            raise ValueError(f"Invalid SM ID on chain {chain}: {actual[2]}")
    return dict(
        all_chains_checked=True,
        guard_rows_intact=True,
        sm_ids=dict(Counter(row[2] for row in rows[1:-1])),
        nsmid=sorted({row[3] for row in rows[1:-1]}),
    )


def time_stats(samples, total_reads, chains):
    if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
        raise ValueError("Non-positive or non-finite CUDA Event time")
    if chains <= 0 or total_reads <= 0 or total_reads % chains:
        raise ValueError("Invalid read/chain counts")
    mean = statistics.fmean(samples)
    return dict(
        mean_event_us=mean,
        median_event_us=statistics.median(samples),
        # For N>1 this includes contention/scheduling; not single-read latency.
        ns_per_chain_hop=mean * 1000 / (total_reads // chains),
        aggregate_Mloads_s=total_reads / mean,
    )


def check_wave_result(samples, expected):
    if not samples:
        raise ValueError("No kernel outputs to validate")
    checks = [check_result(rows, expected) for rows in samples]
    return dict(
        samples_checked=len(checks),
        all_chains_checked=True,
        guard_rows_intact=True,
        sm_counts=[len(check["sm_ids"]) for check in checks],
        per_sample=checks,
    )


def compact_report(report):
    lines = [
        "Dependent SM read diagnostic: logical 8-byte loads; no PCIe GB/s estimate.",
        f'success={report["success"]} correctness={len(report["correctness"])}/{report["expected_correctness"]}',
        f'config={report["config"]}',
        f'code={report["code"]}',
        f'devices={report["devices"]}',
        f'host={report.get("host_buffers", [])}',
    ]
    if "error" in report:
        lines.append(f'FAILURE_CONTEXT {report.get("failure_context", {})}')
        for case in report["correctness"]:
            fields = {
                k: case[k] for k in ("stride", "blocks", "warps", "source", "device")
            }
            lines.append(f"COMPLETED {fields} eager=True graph=True")
        lines.append(f'ERROR {report["error"]}')
    captures = [case.get("capture", {}) for case in report["correctness"]]
    lines.append(
        f'CAPTURE_DEVICE_CHECK {sum(c.get("device_checked", False) for c in captures)}/{report["expected_correctness"]}'
    )
    keys = sorted(
        {(w["stride"], w["blocks"], w["warps"], w["source"]) for w in report["waves"]}
    )
    for stride, blocks, warps, source in keys:
        values = []
        detail = []
        for device in report["config"]["devices"]:
            waves = [
                w
                for w in report["waves"]
                if (w["stride"], w["blocks"], w["warps"], w["source"], w["device"])
                == (stride, blocks, warps, source, device)
            ]
            if not waves:
                continue
            samples = [t for w in waves for t in w["event_us"]]
            stats = time_stats(samples, report["config"]["total_reads"], blocks * warps)
            values.append(stats["mean_event_us"])
            means = [round(statistics.fmean(w["event_us"]), 3) for w in waves]
            sm_counts = sorted(
                {
                    n
                    for w in waves
                    for n in w.get("correctness", {}).get("sm_counts", [])
                }
            )
            detail.append(
                f'd{device} us={stats["mean_event_us"]:.3f} hop_ns={stats["ns_per_chain_hop"]:.2f} Mloads/s={stats["aggregate_Mloads_s"]:.3f} waves={means}'
                f" SMs={sm_counts}"
            )
        ratio = f" d1/d0={values[1]/values[0]:.4f}" if len(values) == 2 else ""
        lines.append(
            f"READ stride={stride} grid={blocks}x{warps} N={blocks*warps} source={source} "
            + " | ".join(detail)
            + ratio
        )
    samples = [s for w in report["waves"] for s in (w["before"], w["after"])]
    for device in report["config"]["devices"]:
        device_samples = [
            s
            for w in report["waves"]
            if w["device"] == device
            for s in (w["before"], w["after"])
        ]
        state = {
            k: sorted({s[k] for s in device_samples if k in s})
            for k in (
                "sm_MHz",
                "mem_MHz",
                "pstate",
                "pcie_gen",
                "pcie_width",
                "throttle_bits",
            )
        }
        lines.append(f"STATE d{device} {state}")
    lines.append(
        f'Telemetry errors={sum(bool(s.get("errors")) for s in samples)}; NVML endpoints do not prove constant state within waves.'
    )
    return "\n".join(lines) + "\n"
