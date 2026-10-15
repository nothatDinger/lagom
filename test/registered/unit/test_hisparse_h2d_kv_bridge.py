import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.benchmark.hisparse_h2d_diagnostics import PAGE_BYTES
from sglang.benchmark.hisparse_h2d_grid_perf import LEGACY_LIBRARY
from sglang.benchmark.hisparse_h2d_kv_bridge import (
    BINDING_KEYS,
    METHODS,
    PACKING_KEYS,
    bridge_geometry,
    compact_report,
    method_order,
    prepare_bridge,
    report_status,
)
from sglang.benchmark.hisparse_h2d_linear import LINEAR_LIBRARY, PAYLOAD
from sglang.benchmark.hisparse_h2d_pair import SM_LOAD_POLICIES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def cli(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3] / "scripts/pd_test/bench_h2d_kv_bridge.py"
    )
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("kv_bridge_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_cpu(cli):
    geometry = bridge_geometry(2, 65, 32 * PAGE_BYTES, 2)
    host = torch.randint(
        0,
        256,
        (32, PAGE_BYTES),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(8),
    )
    args = SimpleNamespace(batch=2, seed=9, plan_stride=96, buffer_slots=96)
    plan = cli.make_plan(torch, args, geometry, 0, 65)
    return geometry, host, plan


def test_three_reservations_and_minimum_working_set():
    g = bridge_geometry()
    assert (g["banks"], g["entries"], g["payload_bytes"]) == (64, 3200, 1868800)
    assert g["dense_bank_tokens"] == 3200
    assert g["linear_byte_start"] == 119808000
    assert g["random_tokens"] >= 64 * 3200
    assert (
        g["linear_byte_start"] + 64 * 1868800 <= g["random_token_start"] // 64 * 37440
    )
    with pytest.raises(ValueError, match="cannot hold"):
        bridge_geometry(source_bytes=128 * 2**20)


def test_bindings_accept_production_policy_names_and_detect_changed_options(cli):
    # Exercise the binding checks without CUDA, using the REAL policy names
    # passed by CopyCase ("legacy", not its numeric JIT code 0).
    host = SimpleNamespace(data_ptr=lambda: 4096, is_pinned=lambda: True)
    adjacent_src = torch.arange(64 * 3200, dtype=torch.int64).reshape(64, 32, 100)
    original_src = adjacent_src + 500000
    dst = torch.arange(3200, dtype=torch.int32).reshape(32, 100)
    original_dst = dst.flip(1)
    output = torch.empty(8, dtype=torch.uint8)
    kwargs = dict(
        device_buffer=output,
        miss_count=torch.full((32,), 100, dtype=torch.int32),
        num_real_reqs=torch.tensor([32], dtype=torch.int32),
        miss_dst=original_dst,
        num_blocks=4,
        block_size=1024,
        item_size_bytes=584,
        is_dsv4_layout=True,
        load_policy=SM_LOAD_POLICIES["sm"],
    )
    original = SimpleNamespace(
        source=host, sm_kwargs=kwargs, plan_src=original_src, nbytes=PAYLOAD, banks=64
    )
    packed = SimpleNamespace(
        source=host,
        sm_kwargs=dict(
            kwargs, miss_dst=dst, load_policy=SM_LOAD_POLICIES["sm_ordered"]
        ),
        plan_src=adjacent_src,
        nbytes=PAYLOAD,
        banks=64,
    )
    inputs = [
        SimpleNamespace(data_ptr=lambda b=b: 4096 + b * PAYLOAD, numel=lambda: PAYLOAD)
        for b in range(64)
    ]
    linear = SimpleNamespace(inputs=inputs, nbytes=PAYLOAD, banks=64)
    cases = dict(zip(METHODS, (original, packed, linear)))
    graphs = {m: SimpleNamespace(waves=2) for m in METHODS}
    plan = dict(src=original_src, dst=original_dst)
    adjacent = dict(src=adjacent_src, active_src=adjacent_src, dst=dst)

    def check():
        return cli.binding_record(torch, 0, host, cases, graphs, plan, adjacent, inputs)

    assert check()["valid"]
    packed.sm_kwargs["load_policy"] = "ca"
    assert not check()["checks"]["same_gather_options"]
    packed.sm_kwargs["load_policy"] = SM_LOAD_POLICIES["sm_ordered"]
    packed.sm_kwargs["device_buffer"] = output.clone()
    assert not check()["checks"]["same_gather_output"]


def token_bytes(flat, token):
    # Literal layout oracle, independent of preparation's address helper.
    page, offset = divmod(token, 64)
    return torch.cat(
        (
            flat[page * 37440 + offset * 576 : page * 37440 + (offset + 1) * 576],
            flat[
                page * 37440
                + 36864
                + offset * 8 : page * 37440
                + 36864
                + (offset + 1) * 8
            ],
        )
    )


def test_packing_preserves_ordinals_bytes_originals_and_every_untouched_byte(cli):
    from bench_h2d_copy import validate_sm

    geometry, host, plan = setup_cpu(cli)
    before, expected = host.clone(), host.clone()
    frozen_plan = copy.deepcopy(plan)
    adjacent, oracle, inputs, evidence = prepare_bridge(torch, host, plan, geometry)
    assert all(evidence[k] for k in PACKING_KEYS)
    assert adjacent["dst"].stride(0) == plan["dst"].stride(0) == 96
    assert adjacent["src"].stride() == plan["src"].stride()
    assert adjacent["destination_pages"] == plan["destination_pages"]
    assert oracle["active_src"] is plan["active_src"]
    assert oracle["dst"] is adjacent["dst"]
    assert adjacent["dst"][:, :65].flatten().tolist() == list(range(130))
    assert torch.equal(adjacent["dst"][:, 65:], plan["dst"][:, 65:])
    for bank in range(2):
        assert (adjacent["active_src"][bank].flatten().diff() == 1).all()
        expected_output = torch.zeros(
            plan["destination_pages"], 37440, dtype=torch.uint8
        )
        for ordinal, (old, new) in enumerate(
            zip(
                plan["active_src"][bank].flatten().tolist(),
                adjacent["active_src"][bank].flatten().tolist(),
            )
        ):
            payload = token_bytes(before.flatten(), old)
            assert torch.equal(token_bytes(host.flatten(), new), payload)
            assert torch.equal(
                inputs[bank][ordinal * 584 : (ordinal + 1) * 584], payload
            )
            page, offset = divmod(new, 64)
            expected.flatten()[
                page * 37440 + offset * 576 : page * 37440 + (offset + 1) * 576
            ] = payload[:576]
            expected.flatten()[
                page * 37440
                + 36864
                + offset * 8 : page * 37440
                + 36864
                + (offset + 1) * 8
            ] = payload[576:]
            linear = (
                geometry["linear_byte_start"]
                + bank * geometry["payload_bytes"]
                + ordinal * 584
            )
            expected.flatten()[linear : linear + 584] = payload
            # Build the remapped output independently of the packed source.
            dp, do = divmod(ordinal, 64)
            expected_output.flatten()[
                dp * 37440 + do * 576 : dp * 37440 + (do + 1) * 576
            ] = payload[:576]
            expected_output.flatten()[
                dp * 37440 + 36864 + do * 8 : dp * 37440 + 36864 + (do + 1) * 8
            ] = payload[576:]
        validate_sm(
            torch,
            host,
            expected_output,
            oracle["active_src"][bank],
            oracle["dst"][:, :65],
            check_untouched=True,
        )
    assert torch.equal(host, expected)
    assert all(
        torch.equal(plan[k], frozen_plan[k]) for k in ("src", "dst", "active_src")
    )


@pytest.mark.parametrize(
    "fault", ["linear_overlap", "random_overlap", "payload", "destination"]
)
def test_invalid_reservations_fail_before_any_host_write(cli, fault):
    geometry, host, plan = setup_cpu(cli)
    before = host.clone()
    if fault == "linear_overlap":
        geometry["linear_byte_start"] = 0
    elif fault == "random_overlap":
        geometry["random_token_start"] = 0
    elif fault == "payload":
        geometry["payload_bytes"] += 8
    else:
        plan["destination_pages"] = 1
    with pytest.raises(ValueError, match="reservations/destinations"):
        prepare_bridge(torch, host, plan, geometry)
    assert torch.equal(host, before)


def report_fixture(mode="main"):
    r = dict(
        success=True,
        config=dict(mode=mode),
        packing=dict.fromkeys(PACKING_KEYS, True),
        correctness=[],
        chunk_graphs=[],
        windows=[],
        bindings=[
            dict(device=d, valid=True, checks=dict.fromkeys(BINDING_KEYS, True))
            for d in (0, 1)
        ],
        jit_artifacts=[
            dict(variant=v, name=n, sha256="a" * 64)
            for v, n in (("legacy", LEGACY_LIBRARY), ("linear", LINEAR_LIBRARY))
        ],
    )
    for d in (0, 1):
        for m in METHODS:
            r["correctness"].append(
                dict(
                    device=d, method=m, eager_banks=[0, 63], eager=True, all_bytes=True
                )
            )
            r["chunk_graphs"].append(
                dict(
                    device=d,
                    method=m,
                    capture_device=d,
                    correctness=True,
                    per_copy_timing_events=0,
                    launches_per_chunk=1,
                    banks=64,
                    nbytes=PAYLOAD,
                    num_blocks=4,
                    block_size=1024,
                    waves=2,
                    copy_nodes=128,
                )
            )
    if mode == "main":
        for t, d in enumerate((0, 1, 1, 0)):
            for p, m in enumerate(method_order([0, 1, 1, 0][:t].count(d))):
                r["windows"].append(
                    dict(
                        device=d,
                        method=m,
                        trial=t,
                        position=p,
                        sampler_enabled=False,
                        metrics=dict(
                            valid=True,
                            logical_wall_GBps={
                                "gather4": 30,
                                "gather_adjacent4": 40,
                                "linear4": 50,
                            }[m],
                            max_enqueue_fraction=0.001,
                            chunk_gpu_cv=0.001,
                        ),
                    )
                )
    return r


@pytest.mark.parametrize("mode", ["check", "main"])
def test_complete_report_gates_ratios_and_uses_time_not_rate_ratio(mode):
    r = report_fixture(mode)
    assert report_status(r)["ready"]
    text = compact_report(r)
    assert ("RATIOS" in text) == (mode == "main")
    if mode == "main":
        assert "adjacent/random_us=[0.75, 0.75]" in text
        assert "linear/adjacent_us=[0.8, 0.8]" in text


@pytest.mark.parametrize(
    "fault",
    [
        "packing",
        "binding",
        "eager",
        "capture",
        "grid",
        "chunk",
        "jit",
        "missing",
        "duplicate",
        "order",
        "sampled",
        "quality",
        "drift",
    ],
)
def test_invalid_or_unbalanced_data_never_produces_ratios(fault):
    r = report_fixture()
    if fault == "packing":
        r["packing"]["original_sources_preserved"] = False
    elif fault == "binding":
        r["bindings"][0]["checks"]["same_gather_output"] = False
    elif fault == "eager":
        r["correctness"][0]["all_bytes"] = False
    elif fault == "capture":
        r["chunk_graphs"][0]["capture_device"] = 1
    elif fault == "grid":
        r["chunk_graphs"][0]["num_blocks"] = 8
    elif fault == "chunk":
        r["chunk_graphs"][0].update(waves=3, copy_nodes=192)
    elif fault == "jit":
        r["jit_artifacts"][0]["name"] = "another.so"
    elif fault == "missing":
        r["windows"].pop()
    elif fault == "duplicate":
        r["windows"].append(copy.deepcopy(r["windows"][0]))
    elif fault == "order":
        r["windows"][0]["position"] = 2
    elif fault == "sampled":
        r["windows"][0]["sampler_enabled"] = True
    elif fault == "quality":
        r["windows"][0]["metrics"]["valid"] = False
    else:
        r["windows"][-1]["metrics"]["logical_wall_GBps"] *= 0.7
    assert not report_status(r)["ready"]
    assert "RATIOS" not in compact_report(r)


def test_early_failure_produces_a_readable_compact():
    text = compact_report(
        dict(success=False, config=dict(mode="check"), error="no GPU")
    )
    assert "results_valid=False" in text and "no GPU" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
