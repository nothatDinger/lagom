import pytest

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


def test_jit_cache_does_not_alias_different_load_policies(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(hisparse, "load_jit", fake_load)
    modules = [
        hisparse._jit_copy_planned_module(1024, True, True, False, p) for p in (0, 1, 2)
    ]
    assert len({id(m) for m in modules}) == 3
    assert len({c[0] for c in calls}) == 3
    assert len({c[1]["cuda_wrappers"][0][1] for c in calls}) == 3
    assert hisparse._jit_copy_planned_module(1024, True, True, False, 2) is modules[2]
    assert len(calls) == 3
