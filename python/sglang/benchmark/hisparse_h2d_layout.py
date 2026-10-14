"""Prepare an adjacent C4 source and summarize its paired diagnostic.

Tensor helpers receive torch explicitly. The compact report uses no tensor
operations and summarizes saved measurement fields without doing GPU work.
"""

import math
import statistics
from collections import defaultdict

from sglang.benchmark.hisparse_h2d_diagnostics import (
    ITEM_BYTES,
    PAGE_BYTES,
    PAGE_TOKENS,
    SCALE_BYTES,
    VALUE_BYTES,
)
from sglang.benchmark.hisparse_h2d_pair import (
    clock_text,
    correctness_summary,
    expected_case_keys,
)


def layout_geometries(batch, misses, source_bytes, max_banks):
    """Reserve disjoint page-aligned dense banks for ALL cases up front.

    Every random plan uses the remaining region. Packing a later case therefore
    cannot overwrite the original sources or an earlier packed case. Banks are
    common across cases and have unique source items within each case.
    """
    if not misses or len(set(misses)) != len(misses):
        raise ValueError("Layout control requires distinct, nonempty miss counts")
    if min(batch, source_bytes, max_banks, *misses) <= 0:
        raise ValueError("Layout geometry settings must be positive")
    host_pages = source_bytes // PAGE_BYTES
    entries = [batch * m for m in misses]
    pages_per_bank = [math.ceil(n / PAGE_TOKENS) for n in entries]
    # Leave enough random slots for all banks of the largest case as well as
    # enough dense pages for every case. No benchmark silently reuses a bank.
    banks = min(
        max_banks,
        host_pages // (sum(pages_per_bank) + max(pages_per_bank)),
    )
    if banks < 2:
        raise ValueError("Source must hold two disjoint random/adjacent banks per case")
    reserved_pages = banks * sum(pages_per_bank)
    cursor = 0
    result = []
    for count, pages in zip(entries, pages_per_bank):
        result.append(
            dict(
                entries=count,
                host_pages=host_pages,
                host_bytes=host_pages * PAGE_BYTES,
                host_tokens=host_pages * PAGE_TOKENS,
                banks=banks,
                payload_bytes=count * ITEM_BYTES,
                rotating_payload_bytes=banks * count * ITEM_BYTES,
                dense_token_start=cursor * PAGE_TOKENS,
                dense_bank_tokens=pages * PAGE_TOKENS,
                random_token_start=reserved_pages * PAGE_TOKENS,
                random_tokens=(host_pages - reserved_pages) * PAGE_TOKENS,
            )
        )
        cursor += banks * pages
    return result


def _addresses(torch, locations):
    page, offset = locations // PAGE_TOKENS, locations % PAGE_TOKENS
    value = page * PAGE_BYTES + offset * VALUE_BYTES
    scale = page * PAGE_BYTES + PAGE_TOKENS * VALUE_BYTES + offset * SCALE_BYTES
    return value, scale


def source_footprint(torch, sources, base_address=0):
    """Virtual 4KiB footprint, not physical pages, TLB misses or PCIe traffic."""
    rows = []
    for bank in sources:
        locations = bank.reshape(-1).to(torch.int64)
        value, scale = _addresses(torch, locations)
        ends = torch.cat(
            (
                value + base_address,
                value + base_address + VALUE_BYTES - 1,
                scale + base_address,
                scale + base_address + SCALE_BYTES - 1,
            )
        )
        # Each field is shorter than 4KiB, so first/last pages cover the field.
        rows.append(
            dict(
                c4_pages=int((locations // PAGE_TOKENS).unique().numel()),
                virtual_4k_pages=int((ends // 4096).unique().numel()),
                adjacent_pairs=int((locations[1:] - locations[:-1] == 1).sum()),
                items=int(locations.numel()),
            )
        )
    return rows


def prepare_adjacent_source(torch, host, plan, geometry):
    """Copy value/scale separately outside timing; keep the ORIGINAL oracle.

    The whole dense reservation is below every random source. Packing cannot
    corrupt baseline data. Chunks bound temporary storage; all banks/bytes are
    checked on CPU before any CUDA graph is captured.
    """
    if (
        host.device.type != "cpu"
        or host.dtype != torch.uint8
        or not host.is_contiguous()
        or host.shape != (geometry["host_pages"], PAGE_BYTES)
    ):
        raise ValueError("Packing requires the full contiguous CPU C4 allocation")
    original = plan["active_src"]
    banks, batch, misses = original.shape
    if not torch.equal(original, plan["src"][:, :, :misses]):
        raise ValueError("Active source oracle disagrees with the source plan")
    if (banks, batch * misses) != (geometry["banks"], geometry["entries"]):
        raise ValueError("Source plan disagrees with layout geometry")
    lo = geometry["random_token_start"]
    hi = lo + geometry["random_tokens"]
    if int(original.min()) < lo or int(original.max()) >= hi:
        raise ValueError("Original sources must be outside every dense reservation")
    if original.unique().numel() != original.numel():
        raise ValueError("Rotating random banks must have unique source slots")
    if plan["dst"][:, :misses].unique().numel() != batch * misses:
        raise ValueError("Destination slots must be unique")
    starts = geometry["dense_token_start"] + (
        torch.arange(banks, dtype=torch.int64) * geometry["dense_bank_tokens"]
    )
    active = (
        starts[:, None] + torch.arange(batch * misses, dtype=torch.int64)
    ).reshape(banks, batch, misses)
    if int(active.min()) < 0 or int(active.max()) >= lo:
        raise ValueError("Adjacent sources overlap random sources or leave the pool")
    src = plan["src"].clone()
    src[:, :, :misses] = active
    flat = host.view(-1)
    vwords, swords = torch.arange(VALUE_BYTES), torch.arange(SCALE_BYTES)
    for bank in range(banks):
        old, new = original[bank].reshape(-1), active[bank].reshape(-1)
        for begin in range(0, len(old), 1024):
            ov, os = _addresses(torch, old[begin : begin + 1024])
            nv, ns = _addresses(torch, new[begin : begin + 1024])
            values = flat[ov[:, None] + vwords]
            scales = flat[os[:, None] + swords]
            flat[nv[:, None] + vwords] = values
            flat[ns[:, None] + swords] = scales
            if not (
                torch.equal(flat[nv[:, None] + vwords], values)
                and torch.equal(flat[ns[:, None] + swords], scales)
                and torch.equal(flat[ov[:, None] + vwords], values)
                and torch.equal(flat[os[:, None] + swords], scales)
            ):
                raise RuntimeError("C4 packing changed KV contents or original sources")
    adjacent = dict(plan, src=src, active_src=active)
    evidence = dict(
        banks=banks,
        entries_per_bank=batch * misses,
        payload_bytes=geometry["payload_bytes"],
        all_banks_payload_checked=True,
        original_sources_preserved=True,
        destination_plan_shared=adjacent["dst"] is plan["dst"],
        source_row_stride=src.stride(1),
        dense_token_start=geometry["dense_token_start"],
        dense_bank_tokens=geometry["dense_bank_tokens"],
        random_token_start=lo,
        random_tokens=geometry["random_tokens"],
        random=source_footprint(torch, original, host.data_ptr()),
        adjacent=source_footprint(torch, active, host.data_ptr()),
    )
    return adjacent, evidence


def layout_method_pairs(methods, trial, reverse=False):
    """Alternate layout order within each GPU wave; CUDA ABBA stays unchanged."""
    required = ("sm", "sm_adjacent", "sm_hbm", "sm_hbm_adjacent", "ce")
    if tuple(methods) != required:
        raise ValueError("Unexpected methods for the standalone layout control")
    pairs = [("sm", "sm_adjacent"), ("sm_hbm", "sm_hbm_adjacent"), ("ce",)]
    if reverse:
        pairs.reverse()
    if (trial % 2 == 1) != reverse:
        pairs = [tuple(reversed(pair)) for pair in pairs]
    return [method for pair in pairs for method in pair]


def layout_compact_report(report):
    check = correctness_summary(report)
    config = report["config"]
    expected_plans = {f'B{config["batch"]}-miss{m}' for m in config["misses"]}
    plans = report.get("layout_plans", [])
    bindings = report.get("layout_bindings", [])
    expected_bindings = {
        (case, device) for case in expected_plans for device in config["devices"]
    }
    prepared = (
        len(plans) == len(expected_plans)
        and {row["case"] for row in plans} == expected_plans
        and all(
            all(
                row.get(key) is True
                for key in (
                    "all_banks_payload_checked",
                    "original_sources_preserved",
                    "destination_plan_shared",
                )
            )
            for row in plans
        )
    )
    bound = (
        len(bindings) == len(expected_bindings)
        and {(row["case"], row["device"]) for row in bindings} == expected_bindings
        and all(row["valid"] is True for row in bindings)
    )
    expected_waves = {
        (phase, case, method, device, trial)
        for case, method, device in expected_case_keys(report)
        for phase in config["phases"]
        for trial, current in enumerate(config["order"])
        if current == device
    }
    waves = report.get("waves", [])
    actual_waves = [
        (w["phase"], w["case"], w["method"], w["device"], w["trial"]) for w in waves
    ]
    timed = (
        len(actual_waves) == len(expected_waves)
        and set(actual_waves) == expected_waves
        and all(
            len(w["copy_us"]) == w["banks"] >= 2
            and all(math.isfinite(t) and t > 0 for t in w["copy_us"])
            for w in waves
        )
    )
    ready = (
        report["success"] is True
        and check["complete"]
        and prepared
        and bound
        and timed
        and (not config.get("profile") or report.get("profile_success") is True)
    )
    lines = [
        "Adjacent-source C4 diagnostic; packing excluded, same legacy CTA4/1024.",
        f'success={report["success"]} profile_success={report.get("profile_success", "not_requested")} '
        f'correctness={check["payload_checked"]}/{check["expected"]} '
        f'graph={check["graph_replay_checked"]}/{check["expected"]} '
        f'untouched={check["untouched_checked"]}/{check["untouched_expected"]} complete={check["complete"]} ready_for_main={ready}',
        f'CONFIG batch={report["config"]["batch"]} misses={report["config"]["misses"]} '
        f'phases={report["config"]["phases"]} order={report["config"]["order"]} '
        f'host_register_flags={report["host_register_flags"]} layout_prepare_us={report.get("layout_prepare_us")}',
        f'CODE git={report.get("code", {}).get("git")} git_error={report.get("code", {}).get("git_error")}',
    ]
    if report.get("error"):
        lines.append("ERROR " + report["error"])
    if "profile_validation" in report:
        lines.append(f'PROFILE_COVERAGE {report["profile_validation"]}')
    lines.append(
        f"TIMING_COVERAGE valid={timed} waves={len(actual_waves)}/{len(expected_waves)}"
    )
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
    for row in report.get("layout_plans", []):
        gates = all(
            row.get(key) is True
            for key in (
                "all_banks_payload_checked",
                "original_sources_preserved",
                "destination_plan_shared",
            )
        )
        lines.append(
            f'LAYOUT {row["case"]} prepared={gates} banks={row["banks"]} '
            f'bytes/copy={row["payload_bytes"]} stride={row["source_row_stride"]} '
            f'dense_start/bank_slots={row["dense_token_start"]}/{row["dense_bank_tokens"]} '
            f'random_start/slots={row["random_token_start"]}/{row["random_tokens"]}'
        )
        lines.append(
            f'FOOTPRINT {row["case"]} random/adjacent_mean_C4_pages='
            f'{statistics.fmean(r["c4_pages"] for r in row["random"]):.1f}/'
            f'{statistics.fmean(r["c4_pages"] for r in row["adjacent"]):.1f} '
            f'virtual_4K_pages={statistics.fmean(r["virtual_4k_pages"] for r in row["random"]):.1f}/'
            f'{statistics.fmean(r["virtual_4k_pages"] for r in row["adjacent"]):.1f}'
        )
    lines.append(
        f'LAYOUT_BINDINGS checked={sum(r["valid"] for r in bindings)}/{len(bindings)}'
    )
    groups = defaultdict(dict)
    for wave in report.get("waves", []):
        if wave["method"] != "ce":
            groups[(wave["phase"], wave["case"], wave["device"])].setdefault(
                wave["method"], []
            ).append(wave)
    for (phase, case, device), methods in sorted(groups.items()):
        for source, names in [
            ("host", ("sm", "sm_adjacent")),
            ("hbm", ("sm_hbm", "sm_hbm_adjacent")),
        ]:
            if not all(name in methods for name in names):
                continue
            random, adjacent = [
                {w["trial"]: w for w in methods[name]} for name in names
            ]
            if random.keys() != adjacent.keys():
                raise ValueError("Layout waves do not pair by trial")
            totals, counts = [], []
            ratios = []
            for trial in sorted(random):
                a, b = random[trial], adjacent[trial]
                if a["payload_bytes"] != b["payload_bytes"] or len(a["copy_us"]) != len(
                    b["copy_us"]
                ):
                    raise ValueError("Layout waves have different payloads or banks")
                if any(
                    not math.isfinite(t) or t <= 0 for w in (a, b) for t in w["copy_us"]
                ):
                    raise ValueError("Invalid layout timing")
                ratios.append(sum(b["copy_us"]) / sum(a["copy_us"]))
            for waves in (random.values(), adjacent.values()):
                totals.append(sum(sum(w["copy_us"]) for w in waves))
                counts.append(sum(len(w["copy_us"]) for w in waves))
            means = [total / count for total, count in zip(totals, counts)]
            payload = next(iter(random.values()))["payload_bytes"]
            lines.append(
                f"RESULT {phase} {case} {source} d{device} random/adjacent_us={means[0]:.2f}/{means[1]:.2f} "
                f"GBps={payload / means[0] / 1000:.2f}/{payload / means[1] / 1000:.2f} "
                f"adjacent/random={totals[1] / totals[0]:.4f} wave_ratios={[round(r, 4) for r in ratios]} "
                f'n={counts[0]} H2D_bytes/copy={payload if source == "host" else 0}'
            )
    ce_groups = defaultdict(list)
    for wave in report.get("waves", []):
        if wave["method"] == "ce":
            ce_groups[(wave["phase"], wave["case"], wave["device"])].append(wave)
    for (phase, case, device), waves in sorted(ce_groups.items()):
        times = [t for wave in waves for t in wave["copy_us"]]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("Invalid CE timing")
        mean = statistics.fmean(times)
        lines.append(
            f'CE {phase} {case} d{device} us={mean:.2f} GBps={waves[0]["payload_bytes"] / mean / 1000:.2f}'
        )
    profiles = {
        (w["case"], w["device"], w["method"]): w
        for w in report.get("profile_waves", [])
    }
    for case in sorted(expected_plans):
        for device in config["devices"]:
            for source, names in [
                ("host", ("sm", "sm_adjacent")),
                ("hbm", ("sm_hbm", "sm_hbm_adjacent")),
            ]:
                waves = [profiles.get((case, device, name)) for name in names]
                if not all(waves):
                    continue
                checks = [w.get("activity_check", {}) for w in waves]
                if not all(c.get("valid") for c in checks):
                    lines.append(
                        f"PROFILE_RESULT {case} {source} d{device} valid=False"
                    )
                    continue
                active = [c["activity_gpu_us"] / c["activities"] for c in checks]
                gaps = [
                    c["event_gpu_us"] / c["activities"] - a
                    for c, a in zip(checks, active)
                ]
                lines.append(
                    f"PROFILE_RESULT {case} {source} d{device} phase=profile_compute "
                    f"random/adjacent_activity_us={active[0]:.2f}/{active[1]:.2f} "
                    f"adjacent/random={active[1] / active[0]:.4f} gap_us={gaps[0]:.2f}/{gaps[1]:.2f}"
                )
    for device in report.get("config", {}).get("devices", []):
        lines.append(
            f"STATE d{device} "
            + clock_text([w for w in report.get("waves", []) if w["device"] == device])
        )
    lines.append(
        "Virtual footprint changes with packing; no claim about physical pages, TLB or PCIe transactions. HBM GBps is logical throughput."
    )
    return "\n".join(lines) + "\n"
