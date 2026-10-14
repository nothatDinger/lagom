import sys

import pytest
import torch
from sglang.kernels.ops.kvcache import hisparse
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def reset_policy_cache(monkeypatch):
    monkeypatch.delenv("SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY", raising=False)
    hisparse._dsv4_copy_load_policy_from_env.cache_clear()
    hisparse._jit_copy_planned_module.cache_clear()
    yield
    hisparse._dsv4_copy_load_policy_from_env.cache_clear()
    hisparse._jit_copy_planned_module.cache_clear()


def test_production_default_and_explicit_benchmark_policy_are_independent(monkeypatch):
    assert hisparse._resolve_copy_load_policy(None, True) == 0
    monkeypatch.setenv("SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY", "cg")
    # Process-level policy stays fixed once resolved; restarting changes it.
    assert hisparse._resolve_copy_load_policy(None, True) == 0
    hisparse._dsv4_copy_load_policy_from_env.cache_clear()
    assert hisparse._resolve_copy_load_policy(None, True) == 2
    assert hisparse._resolve_copy_load_policy("legacy", True) == 0
    assert hisparse._resolve_copy_load_policy("ca", True) == 1
    assert hisparse._resolve_copy_load_policy(None, False) == 0


def test_invalid_policy_is_not_silently_benchmarked(monkeypatch):
    monkeypatch.setenv("SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY", "typo")
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy(None, True)
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy("typo", True)
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy("cg", False)


def test_rocm_rejects_explicit_ptx_policy(monkeypatch):
    monkeypatch.setattr(hisparse.torch.version, "hip", "test")
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy("cg", True)
    assert hisparse._resolve_copy_load_policy("legacy", True) == 0


@pytest.mark.parametrize("policy,code", [("ca_pair", 3), ("ca_v2", 4)])
def test_vector_diagnostics_require_explicit_cuda_layout_and_cannot_be_service_env(
    monkeypatch, policy, code
):
    assert hisparse._resolve_copy_load_policy(policy, True) == code
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy(policy, False)
    monkeypatch.setenv("SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY", policy)
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy(None, True)
    monkeypatch.setattr(hisparse.torch.version, "hip", "test")
    with pytest.raises(ValueError):
        hisparse._resolve_copy_load_policy(policy, True)


def test_vector_source_alignment_rejected_before_compilation(monkeypatch):
    source = torch.empty(128, dtype=torch.uint8)[8:]
    assert source.data_ptr() % 16 == 8
    monkeypatch.setattr(
        hisparse,
        "load_jit",
        lambda *args, **kwargs: pytest.fail(
            "An unaligned vector source must be rejected before JIT/CUDA"
        ),
    )
    with pytest.raises(ValueError, match="16-byte aligned"):
        hisparse.copy_cache_planned_mla(
            miss_src=torch.empty((1, 1), dtype=torch.int64),
            miss_dst=torch.empty((1, 1), dtype=torch.int32),
            miss_count=torch.ones(1, dtype=torch.int32),
            num_real_reqs=torch.ones(1, dtype=torch.int32),
            host_cache=source,
            device_buffer=torch.empty_like(source),
            item_size_bytes=584,
            is_dsv4_layout=True,
            load_policy="ca_v2",
        )


def test_jit_cache_does_not_alias_different_load_policies(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(hisparse, "load_jit", fake_load)
    modules = [
        hisparse._jit_copy_planned_module(1024, True, True, False, p) for p in range(5)
    ]
    assert len({id(m) for m in modules}) == 5
    assert len({c[0] for c in calls}) == 5
    assert len({c[1]["cuda_wrappers"][0][1] for c in calls}) == 5
    assert hisparse._jit_copy_planned_module(1024, True, True, False, 2) is modules[2]
    assert len(calls) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
