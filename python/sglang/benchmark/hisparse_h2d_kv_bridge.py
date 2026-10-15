"""CPU preparation/checks for scattered gather -> adjacent KV gather -> linear."""

import math

from sglang.benchmark.hisparse_h2d_diagnostics import (
    ITEM_BYTES,
    PAGE_BYTES,
    PAGE_TOKENS,
    SCALE_BYTES,
    VALUE_BYTES,
)
from sglang.benchmark.hisparse_h2d_grid_perf import LEGACY_LIBRARY
from sglang.benchmark.hisparse_h2d_layout import _addresses, prepare_adjacent_source
from sglang.benchmark.hisparse_h2d_linear import LINEAR_LIBRARY, PAYLOAD
from sglang.benchmark.hisparse_h2d_pair import clock_text

METHODS = ("gather4", "gather_adjacent4", "linear4")
BINDING_KEYS = {
    "same_host_allocation",
    "same_gather_output",
    "same_counts_and_real",
    "same_gather_options",
    "same_plan_shape_stride",
    "gpu_plans_match_cpu",
    "adjacent_indices",
    "linear_inputs_match_reservation",
    "same_bytes_banks_chunk",
    "host_pinned",
}
PACKING_KEYS = (
    "all_banks_payload_checked",
    "original_sources_preserved",
    "linear_payload_checked",
    "adjacent_destinations",
)


def bridge_geometry(batch=32, misses=100, source_bytes=512 * 2**20, banks=64):
    """Disjoint reservations in ONE allocation: C4, byte-linear, random C4.

    Never silently shrink the bank count. A contemporaneous random baseline is
    necessary: reserving the packed data reduces its address range vs round 17.
    """
    if min(batch, misses, source_bytes, banks) <= 0 or banks < 2:
        raise ValueError("Require positive geometry and at least two banks")
    entries = batch * misses
    pages = source_bytes // PAGE_BYTES
    bank_tokens = math.ceil(entries / PAGE_TOKENS) * PAGE_TOKENS
    c4_end = banks * bank_tokens // PAGE_TOKENS * PAGE_BYTES
    linear_start = math.ceil(c4_end / 128) * 128
    random_page = math.ceil((linear_start + banks * entries * ITEM_BYTES) / PAGE_BYTES)
    random_tokens = (pages - random_page) * PAGE_TOKENS
    if random_tokens < banks * entries:
        raise ValueError("Pool cannot hold disjoint C4, linear and random banks")
    return dict(
        entries=entries,
        banks=banks,
        host_pages=pages,
        host_bytes=pages * PAGE_BYTES,
        host_tokens=pages * PAGE_TOKENS,
        payload_bytes=entries * ITEM_BYTES,
        dense_token_start=0,
        dense_bank_tokens=bank_tokens,
        linear_byte_start=linear_start,
        random_token_start=random_page * PAGE_TOKENS,
        random_tokens=random_tokens,
    )


def prepare_bridge(torch, host, plan, geometry):
    """Preserve KV ordinal/content, remap both indices; packing is NOT timed."""
    banks, batch, misses = plan["active_src"].shape
    entries, payload = geometry["entries"], geometry["payload_bytes"]
    c4_end = (
        (geometry["dense_token_start"] + banks * geometry["dense_bank_tokens"])
        // PAGE_TOKENS
        * PAGE_BYTES
    )
    linear_end = geometry["linear_byte_start"] + banks * payload
    random_start = geometry["random_token_start"] // PAGE_TOKENS * PAGE_BYTES
    if not (
        entries == batch * misses
        and payload == entries * ITEM_BYTES
        and c4_end <= geometry["linear_byte_start"]
        and geometry["linear_byte_start"] % 128 == 0
        and linear_end <= random_start < host.numel()
        and entries <= plan["destination_pages"] * PAGE_TOKENS
    ):
        raise ValueError("Invalid or overlapping bridge reservations/destinations")
    adjacent, evidence = prepare_adjacent_source(torch, host, plan, geometry)
    dst = torch.full_like(plan["dst"], -1)
    dst[:, :misses] = torch.arange(entries, dtype=dst.dtype).reshape(batch, misses)
    adjacent = dict(adjacent, dst=dst)
    # An ORIGINAL-source oracle with the intentionally remapped destinations.
    oracle = dict(plan, dst=dst)
    flat = host.view(-1)
    inputs = []
    for bank in range(banks):
        linear = flat.narrow(0, geometry["linear_byte_start"] + bank * payload, payload)
        old = plan["active_src"][bank].reshape(-1)
        for begin in range(0, entries, 1024):
            value, scale = _addresses(torch, old[begin : begin + 1024])
            values = flat[value[:, None] + torch.arange(VALUE_BYTES)]
            scales = flat[scale[:, None] + torch.arange(SCALE_BYTES)]
            packed = torch.cat((values, scales), dim=1).reshape(-1)
            target = linear[begin * ITEM_BYTES : begin * ITEM_BYTES + packed.numel()]
            target.copy_(packed)
            if not (
                torch.equal(target, packed)
                and torch.equal(
                    flat[value[:, None] + torch.arange(VALUE_BYTES)], values
                )
                and torch.equal(
                    flat[scale[:, None] + torch.arange(SCALE_BYTES)], scales
                )
            ):
                raise ValueError(
                    "Linear preparation changed payload or original sources"
                )
        inputs.append(linear)
    evidence = dict(
        **{k: evidence[k] for k in PACKING_KEYS[:2]},
        linear_payload_checked=True,
        adjacent_destinations=torch.equal(
            dst[:, :misses].reshape(-1), torch.arange(entries, dtype=dst.dtype)
        ),
        random_source_footprint=evidence["random"],
        adjacent_source_footprint=evidence["adjacent"],
    )
    return adjacent, oracle, inputs, evidence


def method_order(local_trial):
    return METHODS if local_trial == 0 else tuple(reversed(METHODS))


def report_status(report):
    config = report["config"]
    expected = {(d, m) for d in (0, 1) for m in METHODS}
    checks = report.get("correctness", [])
    correct = (
        len(checks) == 6
        and {(c["device"], c["method"]) for c in checks} == expected
        and all(
            c["eager_banks"] == [0, 63]
            and c["all_bytes"] is True
            and c["eager"] is True
            for c in checks
        )
    )
    graphs = report.get("chunk_graphs", [])
    graph_ok = (
        len(graphs) == 6
        and {(g["device"], g["method"]) for g in graphs} == expected
        and all(
            g.get("correctness") is True
            and g["capture_device"] == g["device"]
            and g["per_copy_timing_events"] == 0
            and g["launches_per_chunk"] == 1
            and g["banks"] == 64
            and g["nbytes"] == PAYLOAD
            and (g["num_blocks"], g["block_size"]) == (4, 1024)
            and g["waves"] > 0
            and g["copy_nodes"] == g["waves"] * 64
            for g in graphs
        )
    )
    if graph_ok:
        graph_ok &= all(
            len({g["waves"] for g in graphs if g["device"] == d}) == 1 for d in (0, 1)
        )
    bindings = report.get("bindings", [])
    bound = (
        len(bindings) == 2
        and {b["device"] for b in bindings} == {0, 1}
        and all(
            b.get("valid") is True
            and set(b["checks"]) == BINDING_KEYS
            and all(v is True for v in b["checks"].values())
            for b in bindings
        )
    )
    packed = all(report.get("packing", {}).get(k) is True for k in PACKING_KEYS)
    artifacts = report.get("jit_artifacts", [])
    jit = (
        len(artifacts) == 2
        and {(a["variant"], a["name"]) for a in artifacts}
        == {("legacy", LEGACY_LIBRARY), ("linear", LINEAR_LIBRARY)}
        and all(
            isinstance(a.get("sha256"), str) and len(a["sha256"]) == 64
            for a in artifacts
        )
    )
    common = (
        report.get("success") is True
        and correct
        and graph_ok
        and bound
        and packed
        and jit
    )
    windows = report.get("windows", [])
    sequence = [
        (t, d, m, p)
        for t, d in enumerate([0, 1, 1, 0])
        for p, m in enumerate(method_order([0, 1, 1, 0][:t].count(d)))
    ]
    timing = config["mode"] == "main" and (
        len(windows) == 12
        and [(w["trial"], w["device"], w["method"], w["position"]) for w in windows]
        == sequence
        and all(
            w["sampler_enabled"] is False and w["metrics"].get("valid") is True
            for w in windows
        )
        and all(
            isinstance(w["metrics"].get("logical_wall_GBps"), (int, float))
            and math.isfinite(w["metrics"]["logical_wall_GBps"])
            and w["metrics"]["logical_wall_GBps"] > 0
            for w in windows
        )
    )
    drift = {}
    if timing:
        for d, m in sorted(expected):
            rows = [w for w in windows if (w["device"], w["method"]) == (d, m)]
            drift[d, m] = (
                rows[0]["metrics"]["logical_wall_GBps"]
                / rows[1]["metrics"]["logical_wall_GBps"]
            )
        timing &= all(0.9 <= ratio <= 1.1 for ratio in drift.values())
    return dict(
        ready=common and (timing if config["mode"] == "main" else not windows),
        correctness=correct,
        graphs=graph_ok,
        packing=packed,
        bindings=bound,
        jit=jit,
        timing=timing,
        drift=drift,
    )


def compact_report(report):
    status = report_status(report)
    lines = [
        "KV bridge: B32/miss100 CTA4/1024, 64 banks, same KV contents; no timed packing, CE, Rx sampler or service changes.",
        f'success={report["success"]} mode={report["config"]["mode"]} correctness={len(report.get("correctness",[]))}/6 correct={status["correctness"]} '
        f'graphs={len(report.get("chunk_graphs",[]))}/6 graph_valid={status["graphs"]} packing={status["packing"]} bindings={status["bindings"]} JIT={status["jit"]} results_valid={status["ready"]}',
    ]
    if report.get("error"):
        lines.append(f'ERROR active={report.get("active_case")} {report["error"]}')
    lines.append(f'CONFIG {report["config"]}')
    for d, meta in sorted(report.get("devices", {}).items()):
        lines.append(
            f'IDENTITY d{d} physical={meta["physical_index"]} BDF={meta["pci_bus"]} UUID={meta["uuid"]}'
        )
    for b in report.get("host_snapshot", {}).get("buffers", []):
        lines.append(
            f'HOST bytes={b["nbytes"]} NUMA={[r.get("numa") for r in b["regions"]]}'
        )
    lines.append(f'JIT {report.get("jit_artifacts",[])}')
    lines.append(
        f'RESERVATIONS {report.get("geometry")} packing_us={report.get("packing_us")}'
    )
    for b in report.get("bindings", []):
        lines.append(f'BINDING d{b["device"]} valid={b["valid"]} checks={b["checks"]}')
    if report["config"]["mode"] == "main":
        lines.append(
            f'TIMING valid={status["timing"]} windows={len(report.get("windows",[]))}/12'
        )
    index = {}
    for d in (0, 1):
        for m in METHODS:
            rows = [
                w
                for w in report.get("windows", [])
                if w["device"] == d and w["method"] == m
            ]
            if not rows:
                continue
            metrics = [w["metrics"] for w in rows]
            rates = [x.get("logical_wall_GBps") for x in metrics]
            index[d, m] = rates
            lines.append(
                f'RESULT d{d} {m} GBps={rates} valid={[x["valid"] for x in metrics]} '
                f'enqueue={[x.get("max_enqueue_fraction") for x in metrics]} chunkCV={[x.get("chunk_gpu_cv") for x in metrics]} late/early_us={status["drift"].get((d,m))}'
            )
        states = [
            s
            for w in report.get("windows", [])
            if w["device"] == d
            for s in w.get("states", [])
        ]
        lines.append(
            f"STATE d{d} "
            + (
                clock_text([dict(before=s) for s in states])
                if states
                else "not_collected"
            )
        )
    if status["ready"] and report["config"]["mode"] == "main":
        for d in (0, 1):

            def ratio(numerator, denominator):
                return [
                    round(a / b, 4)
                    for a, b in zip(index[d, denominator], index[d, numerator])
                ]

            lines.append(
                f'RATIOS d{d} adjacent/random_us={ratio("gather_adjacent4", "gather4")} '
                f'linear/adjacent_us={ratio("linear4", "gather_adjacent4")} linear/random_us={ratio("linear4", "gather4")}'
            )
    lines.append(
        "Adjacent gather changes BOTH address layouts and their footprints; residual vs linear combines plan, C4, lane utilization and compiler effects. No PCIe/TLB attribution or production packing benefit."
    )
    return "\n".join(lines) + "\n"
