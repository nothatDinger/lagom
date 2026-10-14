"""Evidence binding and paired summaries for the reviewed iterator-v2 kernels.

This is a diagnostic gate for the reviewed SM90a artifact, not a general SASS
dependency verifier or a model of live PCIe requests. No CUDA work here.
"""

import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_iterator_codegen import (
    LIBRARIES,
    decode_sm90_control,
)
from sglang.benchmark.hisparse_h2d_pair import (
    clock_text,
    correctness_summary,
    expected_case_keys,
    source_kind,
)

HOST_METHODS = ("sm", "sm_iter_d1", "sm_iter_d2")
HBM_METHODS = ("sm_hbm", "sm_hbm_iter_d1", "sm_hbm_iter_d2")
SOURCE_FILES = (
    "python/sglang/kernels/jit/csrc/hisparse_copy_iterator.cuh",
    "python/sglang/kernels/jit/csrc/hisparse.cuh",
    "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh",
    "python/sglang/kernels/ops/kvcache/hisparse.py",
)


def iterator_method_order(local_trial, reverse=False):
    """Six permutations PER DEVICE; source-group order alternates per wave."""
    index = local_trial % 6
    permutation = list(itertools.permutations(range(3)))[
        5 - index if reverse else index
    ]
    groups = [
        tuple(methods[i] for i in permutation)
        for methods in (HOST_METHODS, HBM_METHODS)
    ]
    if (local_trial % 2 == 1) != reverse:
        groups.reverse()
    return (*groups[0], *groups[1], "ce")


def _artifact_hashes(artifacts):
    result = {a["variant"]: a["sha256"] for a in artifacts}
    if len(artifacts) != 3 or set(result) != set(LIBRARIES):
        raise ValueError("Require exactly legacy/D1/D2 artifacts")
    return result


def reviewed_window(kernel):
    """Recheck necessary conditions of the manually reviewed v2 A/B path.

    0950's M0x02 is allowed: A writes barrier 5, not 1. The conditional branch
    selects the unpaired tail; the paired path falls through to the B loads.
    Changes in offsets/control flow require review rather than silent acceptance.
    """
    rows = kernel["instructions"]
    if kernel["arch"] != "sm_90a" or len(rows) != 272:
        raise ValueError("Expected reviewed v2 SM90a instruction stream")
    controls = [decode_sm90_control(r["raw"], kernel["arch"]) for r in rows]
    if any(c is None for c in controls):
        raise ValueError("Incomplete SASS control words")
    indexed = {int(r["offset"], 16): (r, c) for r, c in zip(rows, controls)}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate instruction offsets")
    loads = [r["offset"].lower() for r in rows if r["opcode"] == "LDG.E.64.CONSTANT"]
    if loads != ["08b0", "0900", "0920", "0a90", "0ad0", "0af0"]:
        raise ValueError("Source loads differ from reviewed v2")
    if [
        indexed[p][1]["write_barrier"]
        for p in (0x8B0, 0x900, 0x920, 0xA90, 0xAD0, 0xAF0)
    ] != [5, 5, 5, 2, 3, 4]:
        raise ValueError("Source dependency barriers differ from reviewed v2")
    window = [indexed[p] for p in range(0x8B0, 0xB00, 0x10)]
    for row, control in window:
        if control["wait_mask"] & 0x20:
            raise ValueError(f'A completion required before B at {row["offset"]}')
        if row["opcode"].startswith(("SHFL", "BSYNC", "DEPBAR", "WARPSYNC", "STG")):
            raise ValueError("Unexpected synchronization/consumer in A/B window")
        if row["opcode"].startswith("BRA") and (
            row["offset"].lower() != "0950"
            or " ".join(row["text"].split()) != "@P1 BRA 0xd70 ;"
        ):
            raise ValueError("Unexpected branch in A/B window")
    stores = [r for r in rows if r["opcode"].startswith("STG")]
    if not stores or stores[0]["offset"].lower() != "0d00":
        raise ValueError("Store path differs from reviewed v2")
    return dict(
        valid=True,
        arch=kernel["arch"],
        instructions=len(rows),
        first_a="08b0",
        last_b="0af0",
        a_write_barrier=5,
        intervening_waits=[
            dict(offset=r["offset"], mask=c["wait_mask"])
            for r, c in window
            if c["wait_mask"]
        ],
    )


def load_iterator_evidence(directory, repo):
    directory, repo = Path(directory), Path(repo)
    check_path, codegen_path = (
        directory / "check/iterator.json",
        directory / "codegen/codegen.json",
    )
    check = json.loads(check_path.read_text())
    codegen = json.loads(codegen_path.read_text())
    if not (check.get("success") is True and codegen.get("success") is True):
        raise ValueError("Require successful v2 correctness and codegen evidence")
    if codegen["input_sha256"] != sha256(check_path):
        raise ValueError("Codegen is not bound to this correctness report")
    expected = {(d, v, s) for d in (0, 1) for v in LIBRARIES for s in ("host", "hbm")}
    rows = check["correctness"]
    if (
        len(rows) != 12
        or {(r["device"], r["variant"], r["source"]) for r in rows} != expected
    ):
        raise ValueError("Incomplete iterator correctness coverage")
    for row in rows:
        if not (
            row["eager"] is True
            and row["all_bytes_and_padding"] is True
            and row["source_is_pinned"] is True
            and len(set(row["graph_phases"])) == len(row["graph_phases"]) == 6
            and row["capture_stream_device"] == row["device"]
        ):
            raise ValueError("Iterator correctness/capture check failed")
    hashes = check["code"]["sha256"]
    for name in SOURCE_FILES:
        if hashes.get(name) != sha256(repo / name):
            raise ValueError(f"Reviewed CUDA source/API changed: {name}")
    expected_artifacts = _artifact_hashes(check["jit_artifacts"])
    if _artifact_hashes(codegen["artifacts"]) != expected_artifacts:
        raise ValueError("Codegen/correctness artifacts differ")
    for artifact in codegen["artifacts"]:
        resources = artifact["resources"]
        if not (
            resources["complete"] is True and resources["no_local_or_stack"] is True
        ):
            raise ValueError("Incomplete resource data or unexpected stack/local use")
    d2 = next(a for a in codegen["artifacts"] if a["variant"] == "iterator_d2")
    if len(d2["kernels"]) != 1:
        raise ValueError("Expected one reviewed D2 kernel")
    return dict(
        valid=True,
        directory=str(directory),
        check_sha256=sha256(check_path),
        codegen_sha256=sha256(codegen_path),
        source_sha256={name: hashes[name] for name in SOURCE_FILES},
        expected_artifacts=expected_artifacts,
        window=reviewed_window(d2["kernels"][0]),
    )


def match_iterator_artifacts(evidence, artifacts):
    if _artifact_hashes(artifacts) != evidence["expected_artifacts"]:
        raise ValueError(
            "Loaded JIT artifacts differ from reviewed v2; retain report before timing"
        )
    return True


def iterator_bindings(cases):
    selected = [cases[name] for name in (*HOST_METHODS, *HBM_METHODS)]
    root = selected[0]
    checks = {
        key: all(
            c.sm_kwargs[key].data_ptr() == root.sm_kwargs[key].data_ptr()
            for c in selected
        )
        for key in ("miss_dst", "miss_count", "num_real_reqs", "device_buffer")
    }
    checks["same_source_plan"] = all(
        c.plan_src.data_ptr() == root.plan_src.data_ptr()
        and c.plan_src.shape == root.plan_src.shape
        and c.plan_src.stride() == root.plan_src.stride()
        for c in selected
    )
    checks["same_source_per_kind"] = all(
        cases[name].source.data_ptr() == cases[names[0]].source.data_ptr()
        for names in (HOST_METHODS, HBM_METHODS)
        for name in names
    )
    checks["same_geometry"] = all(
        (c.device, c.label, c.banks, c.nbytes)
        == (root.device, root.label, root.banks, root.nbytes)
        for c in selected
    )
    checks["same_launch_policy"] = all(
        c.sm_kwargs["num_blocks"] == 4
        and c.sm_kwargs["block_size"] == 1024
        and c.sm_kwargs["load_policy"] == "legacy"
        for c in selected
    )
    return checks


def timing_coverage(report):
    config = report["config"]
    expected = {
        (phase, case, method, device, trial)
        for case, method, device in expected_case_keys(report)
        for phase in config["phases"]
        for trial, current in enumerate(config["order"])
        if current == device
    }
    waves = report.get("waves", [])
    keys = [
        (w["phase"], w["case"], w["method"], w["device"], w["trial"]) for w in waves
    ]
    valid = len(keys) == len(expected) and set(keys) == expected
    shapes = defaultdict(set)
    for wave in waves:
        valid &= (
            len(wave["copy_us"]) == wave["banks"] >= 2
            and all(math.isfinite(t) and t > 0 for t in wave["copy_us"])
            and wave["h2d_bytes_per_copy"]
            == (0 if source_kind(wave["method"]) == "device" else wave["payload_bytes"])
        )
        shapes[(wave["case"], wave["device"])].add(
            (wave["banks"], wave["payload_bytes"])
        )
        if not wave["case"].startswith("bulk-ce-"):
            local = config["order"][: wave["trial"]].count(wave["device"])
            methods = iterator_method_order(
                local, config.get("reverse_method_order", False)
            )
            valid &= wave.get("local_trial") == local and wave.get(
                "method_position"
            ) == methods.index(wave["method"])
    valid &= all(len(shapes_for_case) == 1 for shapes_for_case in shapes.values())
    return dict(valid=bool(valid), recorded=len(keys), expected=len(expected))


def iterator_compact_report(report):
    config, check = report["config"], correctness_summary(report)
    timing = timing_coverage(report)
    labels = [f'B{config["batch"]}-miss{m}' for m in config["misses"]]
    expected_bindings = {(label, d) for label in labels for d in config["devices"]}
    bindings = report.get("iterator_bindings", [])
    bound = (
        len(bindings) == len(expected_bindings)
        and {(r["case"], r["device"]) for r in bindings} == expected_bindings
        and all(r["valid"] is True and all(r["checks"].values()) for r in bindings)
    )
    evidence = report.get("iterator_evidence", {})
    ready = (
        report.get("success") is True
        and check["complete"]
        and timing["valid"]
        and bound
        and evidence.get("valid") is True
        and report.get("iterator_artifacts_match") is True
        and (not config.get("profile") or report.get("profile_success") is True)
    )
    lines = [
        "Iterator v2: scattered C4, legacy scalar loads, CTA4/1024; no packing.",
        f'success={report["success"]} profile_success={report.get("profile_success", "not_requested")} '
        f'correctness={check["payload_checked"]}/{check["expected"]} graph={check["graph_replay_checked"]}/{check["expected"]} '
        f'untouched={check["untouched_checked"]}/{check["untouched_expected"]} results_valid={ready}',
        f'CONFIG batch={config["batch"]} misses={config["misses"]} phases={config["phases"]} '
        f'order={config["order"]} host_register_flags={report["host_register_flags"]}',
        f"TIMING_COVERAGE {timing} BINDINGS valid={bound} count={len(bindings)}/{len(expected_bindings)}",
        f'EVIDENCE valid={evidence.get("valid")} loaded_artifacts_match={report.get("iterator_artifacts_match")} window={evidence.get("window")}',
        f'CODE git={report.get("code", {}).get("git")} git_error={report.get("code", {}).get("git_error")}',
    ]
    if report.get("error"):
        lines.append("ERROR " + report["error"])
    if "profile_validation" in report:
        lines.append(f'PROFILE_COVERAGE {report["profile_validation"]}')
    for device, meta in sorted(report.get("devices", {}).items()):
        lines.append(
            f'IDENTITY d{device} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
        )
    for buf in report.get("host_snapshot", {}).get("buffers", []):
        placement = [
            (r.get("numa"), r["buffer_covers_whole_vma"]) for r in buf["regions"]
        ]
        lines.append(
            f'HOST bytes={buf["nbytes"]} mapped_bytes={buf["mapped_bytes"]} VMA={placement}'
        )
    if not timing["valid"]:
        return "\n".join(lines) + "\nTiming incomplete; keep pair.json and run.log.\n"
    groups = defaultdict(dict)
    for wave in report["waves"]:
        groups[(wave["phase"], wave["case"], wave["device"])].setdefault(
            wave["method"], []
        ).append(wave)
    for (phase, case, device), methods in sorted(groups.items()):
        for source, names in (("host", HOST_METHODS), ("hbm", HBM_METHODS)):
            if not all(name in methods for name in names):
                continue
            ordered = [
                sorted(methods[name], key=lambda w: w["trial"]) for name in names
            ]
            means = [
                statistics.fmean(t for w in rows for t in w["copy_us"])
                for rows in ordered
            ]
            wave_means = [
                [statistics.fmean(w["copy_us"]) for w in rows] for rows in ordered
            ]
            payload = ordered[0][0]["payload_bytes"]
            lines.append(
                f"RESULT {phase} {case} {source} d{device} L/D1/D2_us={_numbers(means, 3)} "
                f"GBps={_numbers([payload / t / 1000 for t in means], 2)} "
                f'n={sum(w["banks"] for w in ordered[0])} H2D_bytes/copy={payload if source == "host" else 0}'
            )
            lines.append(
                f"WAVES {case} {source} d{device} L/D1/D2_us={[[round(t, 3) for t in ts] for ts in wave_means]}"
            )
            ratio_fields = []
            for label, a, b in (("D1/L", 1, 0), ("D2/D1", 2, 1), ("D2/L", 2, 0)):
                ratios = [x / y for x, y in zip(wave_means[a], wave_means[b])]
                ratio_fields.append(
                    f"{label}={means[a] / means[b]:.4f} waves={[round(x, 4) for x in ratios]}"
                )
            lines.append(
                f"RATIOS {case} {source} d{device} " + " | ".join(ratio_fields)
            )
        waves = methods.get("ce", [])
        if waves:
            mean = statistics.fmean(t for w in waves for t in w["copy_us"])
            lines.append(
                f'CE {phase} {case} d{device} us={mean:.3f} GBps={waves[0]["payload_bytes"] / mean / 1000:.2f} '
                f'wave_us={[round(statistics.fmean(w["copy_us"]), 3) for w in waves]}'
            )
    profiles = {
        (w["case"], w["device"], w["method"]): w
        for w in report.get("profile_waves", [])
    }
    for case in labels:
        for device in config["devices"]:
            for source, names in (("host", HOST_METHODS), ("hbm", HBM_METHODS)):
                checks = [
                    profiles.get((case, device, name), {}).get("activity_check", {})
                    for name in names
                ]
                if not all(c.get("valid") is True for c in checks):
                    continue
                active = [c["activity_gpu_us"] / c["activities"] for c in checks]
                gap = [
                    c["event_gpu_us"] / c["activities"] - t
                    for c, t in zip(checks, active)
                ]
                lines.append(
                    f"PROFILE {case} {source} d{device} phase=profile_compute "
                    f"L/D1/D2_activity_us={_numbers(active, 3)} gap_us={_numbers(gap, 3)} "
                    f"D1/L={active[1]/active[0]:.4f} D2/D1={active[2]/active[1]:.4f} D2/L={active[2]/active[0]:.4f}"
                )
    for device in config["devices"]:
        for phase, waves in (
            ("unprofiled", report["waves"]),
            ("profiled", report.get("profile_waves", [])),
        ):
            lines.append(
                f"STATE {phase} d{device} "
                + clock_text([w for w in waves if w["device"] == device])
            )
    lines.append(
        "Unprofiled events are primary; profile activity is corroboration. HBM GBps is logical throughput. Waves, not graph nodes, are repeat blocks; NVML samples do not prove constant clocks."
    )
    return "\n".join(lines) + "\n"


def _numbers(values, digits):
    return "/".join(f"{value:.{digits}f}" for value in values)
