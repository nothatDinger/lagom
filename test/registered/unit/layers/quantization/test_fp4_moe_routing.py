"""Packed V4 experts must never silently fall through to CUDA FP8 MoE."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "backend",
    [MoeRunnerBackend.AUTO, MoeRunnerBackend.TRITON, MoeRunnerBackend.DEEP_GEMM],
)
def test_cuda_packed_experts_reject_fp8_runners(monkeypatch, backend):
    monkeypatch.setattr(fp8, "_is_cuda", True)
    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: backend)
    monkeypatch.setattr(fp8, "Fp8MoEMethod", lambda config: object())
    config = fp8.Fp8Config(is_fp4_experts=True)
    with pytest.raises(ValueError, match="Packed MXFP4.*ordinary FP8"):
        config.get_quant_method(MagicMock(spec=FusedMoE), "model.layers.0.mlp.experts")


@pytest.mark.parametrize(
    "packed,dequant,cuda",
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_existing_fp8_dequant_and_non_cuda_paths_remain(
    monkeypatch, packed, dequant, cuda
):
    sentinel = object()
    monkeypatch.setattr(fp8, "_is_cuda", cuda)
    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: MoeRunnerBackend.AUTO)
    monkeypatch.setattr(fp8, "Fp8MoEMethod", lambda config: sentinel)
    config = fp8.Fp8Config(is_fp4_experts=packed)
    config.dequant_fp4_to_fp8 = dequant
    assert config.get_quant_method(MagicMock(spec=FusedMoE), "experts") is sentinel


def test_h100_explicit_mxfp4_selects_cutlass_wrapper(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(fp8, "_is_cuda", True)
    monkeypatch.setattr(fp8, "is_sm90_supported", lambda: True)
    monkeypatch.setattr(
        fp8, "get_moe_runner_backend", lambda: MoeRunnerBackend.FLASHINFER_MXFP4
    )
    monkeypatch.setattr(fp8, "Fp8MoEMethod", lambda config: object())
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe",
        SimpleNamespace(
            Mxfp4FlashinferCutlassMoEMethod=lambda method, prefix: sentinel
        ),
    )
    config = fp8.Fp8Config(is_fp4_experts=True)
    assert config.get_quant_method(MagicMock(spec=FusedMoE), "experts") is sentinel
