"""CPU checks and paired reports for legacy CTA4/CTA8; no CUDA work here."""

import math
import statistics
from collections import defaultdict
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_pair import (
    SM_GRID_BLOCKS,
    clock_text,
    correctness_summary,
    expected_case_keys,
    source_kind,
)

METHOD_PAIRS = (("host", ("sm", "sm_cta8")), ("hbm", ("sm_hbm", "sm_hbm_cta8")))
LEGACY_LIBRARY = "sgl_kernel_jit_sparse_copy_planned_1024_True_True_False_0.so"


def grid_method_order(local_trial, reverse=False):
    """Balance 4->8 / 8->4 and source-group order on EACH device's six waves."""
    pairs = [list(names) for _, names in METHOD_PAIRS]
    if (local_trial % 2 == 1) != reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
    if (local_trial % 6 in (2, 3, 5)) != reverse:
        pairs.reverse()
    return (*pairs[0], *pairs[1], "ce")


def grid_schedule(batch, misses, blocks):
    """Logical work distribution, not measured SM placement or transactions.

    Flat item k belongs to warp k % total_warps. Every warp still visits every
    real request in the legacy loop, including rows where it has no KV to copy.
    """
    warps = blocks * 32
    total = batch * misses
    per_warp = [len(range(w, total, warps)) for w in range(warps)]
    return dict(
        blocks=blocks,
        total_warps=warps,
        copying_warps=sum(n > 0 for n in per_warp),
        kv_per_warp_min_max=[min(per_warp), max(per_warp)],
        kv_per_cta=[sum(per_warp[i : i + 32]) for i in range(0, warps, 32)],
        total_kv=total,
        warp_request_visits=batch * warps,
    )


def grid_bindings(cases):
    selected = [cases[m] for _, names in METHOD_PAIRS for m in names]
    root = cases["sm"]
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
        cases[m].source.data_ptr() == cases[names[0]].source.data_ptr()
        for _, names in METHOD_PAIRS
        for m in names
    )
    checks["same_geometry"] = all(
        (c.device, c.label, c.banks, c.nbytes)
        == (root.device, root.label, root.banks, root.nbytes)
        for c in selected
    )
    checks["only_grid_changes"] = all(
        cases[m].sm_kwargs["num_blocks"] == blocks
        and cases[m].sm_kwargs["block_size"] == 1024
        and cases[m].sm_kwargs["load_policy"] == "legacy"
        and cases[m].sm_kwargs["is_dsv4_layout"] is True
        and cases[m].sm_kwargs["item_size_bytes"] == 584
        for m, blocks in SM_GRID_BLOCKS.items()
    )
    return checks


def loaded_grid_artifact(maps_text=None):
    maps_text = Path("/proc/self/maps").read_text() if maps_text is None else maps_text
    paths = {
        f[5]
        for line in maps_text.splitlines()
        if len(f := line.split(None, 5)) == 6
        and Path(f[5].removesuffix(" (deleted)")).name == LEGACY_LIBRARY
    }
    if len(paths) != 1:
        raise ValueError(
            f"Expected one shared legacy JIT artifact, found {sorted(paths)}"
        )
    path = Path(next(iter(paths)))
    return dict(variant="legacy", path=str(path), sha256=sha256(path))


def grid_timing_coverage(report):
    config = report["config"]
    expected = {
        (phase, case, method, device, trial)
        for case, method, device in expected_case_keys(report)
        for phase in config["phases"]
        for trial, d in enumerate(config["order"])
        if d == device
    }
    waves = report.get("waves", [])
    keys = [
        (w["phase"], w["case"], w["method"], w["device"], w["trial"]) for w in waves
    ]
    valid = len(keys) == len(expected) and set(keys) == expected
    shapes = defaultdict(set)
    for w in waves:
        valid &= (
            len(w["copy_us"]) == w["banks"] >= 2
            and all(math.isfinite(t) and t > 0 for t in w["copy_us"])
            and w.get("grid_control") is True
            and w["h2d_bytes_per_copy"]
            == (0 if source_kind(w["method"]) == "device" else w["payload_bytes"])
        )
        shapes[(w["case"], w["device"])].add((w["banks"], w["payload_bytes"]))
        if w["method"] in SM_GRID_BLOCKS:
            valid &= (
                w.get("num_blocks") == SM_GRID_BLOCKS[w["method"]]
                and w.get("block_size") == 1024
                and w.get("load_policy") == "legacy"
            )
        if not w["case"].startswith("bulk-ce-"):
            local = config["order"][: w["trial"]].count(w["device"])
            methods = grid_method_order(
                local, config.get("reverse_method_order", False)
            )
            valid &= w.get("local_trial") == local and w.get(
                "method_position"
            ) == methods.index(w["method"])
    valid &= all(len(values) == 1 for values in shapes.values())
    return dict(valid=bool(valid), recorded=len(keys), expected=len(expected))


def grid_compact_report(report):
    config = report["config"]
    check, timing = correctness_summary(report), grid_timing_coverage(report)
    labels = [f'B{config["batch"]}-miss{m}' for m in config["misses"]]
    expected_bindings = {(label, d) for label in labels for d in config["devices"]}
    bindings = report.get("grid_bindings", [])
    bound = (
        len(bindings) == len(expected_bindings)
        and {(r["case"], r["device"]) for r in bindings} == expected_bindings
        and all(r["valid"] is True and all(r["checks"].values()) for r in bindings)
    )
    artifacts = report.get("jit_artifacts", [])
    artifact_valid = (
        len(artifacts) == 1
        and artifacts[0].get("variant") == "legacy"
        and Path(artifacts[0]["path"]).name == LEGACY_LIBRARY
    )
    profiles = report.get("profile_waves", [])
    profile_keys = [(w["case"], w["method"], w["device"]) for w in profiles]
    sm_profiles = [w for w in profiles if w["method"] in SM_GRID_BLOCKS]
    grid_expected = sum(m in SM_GRID_BLOCKS for _, m, _ in expected_case_keys(report))
    grid_checked = sum(
        w.get("grid_control") is True
        and w.get("activity_check", {}).get("grid_verified") is True
        and w.get("activity_check", {}).get("valid") is True
        for w in sm_profiles
    )
    profile_valid = (
        report.get("profile_success") is True
        and len(profile_keys) == len(expected_case_keys(report))
        and set(profile_keys) == expected_case_keys(report)
        and grid_checked == grid_expected
    )
    ready = (
        report.get("success") is True
        and check["complete"]
        and timing["valid"]
        and bound
        and artifact_valid
        and (not config.get("profile") or profile_valid)
    )
    lines = [
        "Legacy CTA4/CTA8 control: same scalar loads, source/plan/output; only runtime grid changes.",
        f'success={report["success"]} profile_success={report.get("profile_success", "not_requested")} '
        f'correctness={check["payload_checked"]}/{check["expected"]} graph={check["graph_replay_checked"]}/{check["expected"]} '
        f'untouched={check["untouched_checked"]}/{check["untouched_expected"]} results_valid={ready}',
        f'CONFIG B={config["batch"]} misses={config["misses"]} phases={config["phases"]} '
        f'order={config["order"]} register_flags={report["host_register_flags"]}',
        f"TIMING_COVERAGE {timing} BINDINGS valid={bound} count={len(bindings)}/{len(expected_bindings)}",
        f"GRID_ACTIVITY valid={profile_valid} verified={grid_checked}/{grid_expected} "
        "(grid/block checked on every matched SM activity)",
        f'JIT single_legacy_artifact={artifact_valid} sha256={artifacts[0].get("sha256") if artifact_valid else None}',
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
    for miss in config["misses"]:
        for blocks in (4, 8):
            lines.append(
                f'SCHEDULE B{config["batch"]}-miss{miss} {grid_schedule(config["batch"], miss, blocks)}'
            )
    if not timing["valid"]:
        return "\n".join(lines) + "\nTiming incomplete; keep pair.json and run.log.\n"
    grouped = defaultdict(dict)
    for wave in report["waves"]:
        grouped[(wave["phase"], wave["case"], wave["device"])].setdefault(
            wave["method"], []
        ).append(wave)
    for (phase, case, device), methods in sorted(grouped.items()):
        for source, names in METHOD_PAIRS:
            if not all(name in methods for name in names):
                continue
            rows = [sorted(methods[name], key=lambda w: w["trial"]) for name in names]
            means = [
                statistics.fmean(t for w in group for t in w["copy_us"])
                for group in rows
            ]
            wave_means = [
                [statistics.fmean(w["copy_us"]) for w in group] for group in rows
            ]
            ratios = [b / a for a, b in zip(*wave_means)]
            payload = rows[0][0]["payload_bytes"]
            lines.append(
                f"RESULT {phase} {case} {source} d{device} CTA4/8_us={means[0]:.3f}/{means[1]:.3f} "
                f"GBps={payload/means[0]/1000:.2f}/{payload/means[1]/1000:.2f} "
                f'8/4={means[1]/means[0]:.4f} n={sum(w["banks"] for w in rows[0])} H2D_bytes/copy={payload if source == "host" else 0}'
            )
            lines.append(
                f"WAVES {case} {source} d{device} CTA4/8_us={[[round(t, 3) for t in ts] for ts in wave_means]} "
                f"ratios={[round(r, 4) for r in ratios]} faster={sum(r < 1 for r in ratios)}/{len(ratios)}"
            )
        ce = methods.get("ce", [])
        if ce:
            mean = statistics.fmean(t for w in ce for t in w["copy_us"])
            lines.append(
                f'CE {phase} {case} d{device} us={mean:.3f} GBps={ce[0]["payload_bytes"]/mean/1000:.2f} '
                f'wave_us={[round(statistics.fmean(w["copy_us"]), 3) for w in ce]}'
            )
    indexed_profiles = {(w["case"], w["device"], w["method"]): w for w in profiles}
    for case in labels:
        for device in config["devices"]:
            for source, names in METHOD_PAIRS:
                checks = [
                    indexed_profiles.get((case, device, name), {}).get(
                        "activity_check", {}
                    )
                    for name in names
                ]
                if not all(c.get("valid") is True for c in checks):
                    continue
                active = [c["activity_gpu_us"] / c["activities"] for c in checks]
                gaps = [
                    c["event_gpu_us"] / c["activities"] - t
                    for c, t in zip(checks, active)
                ]
                lines.append(
                    f"PROFILE {case} {source} d{device} phase=profile_compute "
                    f"CTA4/8_activity_us={active[0]:.3f}/{active[1]:.3f} 8/4={active[1]/active[0]:.4f} "
                    f'gap_us={gaps[0]:.3f}/{gaps[1]:.3f} grid_verified={all(c.get("grid_verified") is True for c in checks)}'
                )
    for w in profiles:
        c = w.get("activity_check", {})
        if c.get("valid") is not True:
            lines.append(f'PROFILE_ERROR {w["scope"]} {c}')
    for device in config["devices"]:
        for phase, waves in (("unprofiled", report["waves"]), ("profiled", profiles)):
            lines.append(
                f"STATE {phase} d{device} "
                + clock_text([w for w in waves if w["device"] == device])
            )
    lines.append(
        "Unprofiled events are primary; profile activity is corroboration. HBM GBps is logical, H2D=0. More CTA also duplicates traversal; launch grid does not prove actual SM placement or live requests."
    )
    return "\n".join(lines) + "\n"
