import shutil

import pytest

from sglang.benchmark.hisparse_pr40670_check import (
    REFERENCE,
    launch_config,
    source_manifest,
    validate_profile,
)
from sglang.benchmark.hisparse_pr40670_layout import build_hisparse_spec_layout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_pinned_bodies_and_only_recorded_include_redirect():
    m = source_manifest()
    assert len(m["files"]) == 3
    assert sum(f["adaptation"] != "none" for f in m["files"]) == 1


def test_source_algorithm_change_rejected(tmp_path):
    shutil.copytree(REFERENCE, tmp_path / "reference")
    p = tmp_path / "reference" / "hisparse_spec.cuh"
    p.write_text(
        p.read_text().replace("BLOCK_SIZE, device)", "BLOCK_SIZE * 2, device)", 1)
    )
    with pytest.raises(ValueError):
        source_manifest(tmp_path / "reference")


@pytest.mark.parametrize(
    "w,k,h,t,scratch", [(2, 4, 16, 16, 14), (2, 4, 16, 2, 0), (6, 16, 128, 16, 13)]
)
def test_reference_scratch_formulas(w, k, h, t, scratch):
    layout = build_hisparse_spec_layout(
        hot_slots=h, tail_capacity_slots=t, compress_ratio=4, verify_width=w, top_k=k
    )
    assert layout.scratch_slots == scratch
    assert layout.scratch_offset + layout.scratch_slots == layout.total_buffer_slots
    assert layout.active_buffer_slots == layout.scratch_offset
    assert layout.state_width(3) >= max(12, 5 * w * k)
    assert layout.state_width(3) % 2 == 0


def trace():
    c = launch_config(4, 6, 16)
    return dict(
        traceEvents=[
            dict(
                cat="kernel",
                name=f"void sglang::load_cache_to_device_buffer_spec_{kind}_kernel<...>",
                dur=1,
                args=dict(
                    device=0, stream=7, grid=c[kind]["grid"], block=c[kind]["block"]
                ),
            )
            for kind in ("gather", "commit")
        ]
    )


def test_actual_launch_shape_and_two_kernel_contract():
    r = validate_profile(trace(), 4, 6, 16)
    assert r["valid"] and not r["independent_copy_kernel"]
    assert r["kernels"][0]["grid"] == [4, 2, 1]
    assert launch_config(32, 6, 16)["gather"]["grid"] == [32, 2, 1]
    assert launch_config(32, 6, 16)["commit"]["grid"] == [32, 1, 1]


@pytest.mark.parametrize(
    "kind", ["extra_copy", "wrong_block", "wrong_grid", "wrong_device"]
)
def test_incorrect_reference_launch_rejected(kind):
    t = trace()
    if kind == "extra_copy":
        t["traceEvents"].append(
            dict(cat="kernel", name="copy_cache_planned_kernel", dur=1, args={})
        )
    elif kind == "wrong_block":
        t["traceEvents"][0]["args"]["block"] = [1024, 1, 1]
    elif kind == "wrong_grid":
        t["traceEvents"][0]["args"]["grid"] = [4, 1, 1]
    else:
        t["traceEvents"][0]["args"]["device"] = 1
    with pytest.raises(ValueError):
        validate_profile(t, 4, 6, 16)


def test_wrapper_reference_has_no_extra_planned_copy():
    from sglang.benchmark.hisparse_pr40670_reference import _GATHER_BLOCK_SIZE

    assert _GATHER_BLOCK_SIZE == 64


def test_layout_rejects_unrepresentable_reference_window():
    with pytest.raises(ValueError):
        build_hisparse_spec_layout(
            hot_slots=32768,
            tail_capacity_slots=16,
            compress_ratio=4,
            verify_width=6,
            top_k=2048,
        )


def test_c4_oracle_checks_scale_and_untouched_bytes(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import torch

    root = Path(__file__).resolve().parents[3]
    scripts = root / "scripts/pd_test"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "reference_check_driver", scripts / "check_hisparse_pr40670.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    indices = driver.c4_indices(torch, torch.tensor([63, 64]))
    assert indices[0, 0].item() == 63 * 576
    assert indices[1, 0].item() == 37440
    assert indices[0, -1].item() == 36864 + 63 * 8 + 7
    assert indices[1, -1].item() == 37440 + 36864 + 7
    host = torch.arange(2 * 37440, dtype=torch.int64).to(torch.uint8).view(2, 37440)
    output = torch.full((1, 37440), 165, dtype=torch.uint8)
    before = output.clone()
    top = torch.full((4, 2, 4), -1, dtype=torch.int32)
    top[0, 0, 0] = 65
    loc = torch.full_like(top, -1)
    loc[0, 0, 0] = 7
    args = dict(
        top_k_tokens=top,
        top_k_device_locs=loc,
        seq_lens=torch.full((8,), 128, dtype=torch.int32),
        num_real_reqs=torch.tensor([1]),
        req_pool_indices=torch.tensor([0, 999, 999, 999]),
    )
    case = SimpleNamespace(
        args=args,
        layout=SimpleNamespace(verify_width=2, top_k=4),
        output=output,
        host=host,
        host_stride=128,
        locs=torch.tensor([[7]]),
        initial_locs=torch.tensor([[7]]),
    )
    src = driver.c4_indices(torch, torch.tensor([65]))
    dst = driver.c4_indices(torch, torch.tensor([7]))
    output.view(-1)[dst] = host.view(-1)[src]
    assert driver.oracle(torch, case, before)["valid_occurrences"] == 1
    good = output.clone()
    for idx in [dst[0, 0], dst[0, -1], torch.tensor(0)]:
        output.copy_(good)
        output.view(-1)[idx] ^= 1
        with pytest.raises(ValueError):
            driver.oracle(torch, case, before)
