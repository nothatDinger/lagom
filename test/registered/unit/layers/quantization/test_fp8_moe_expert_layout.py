"""Reject packed MXFP4 before the FP8 Triton runner reaches graph capture."""

from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "backend,fp4,dequant,a2a,reject",
    [
        ("auto", True, False, "none", True),
        ("triton", True, False, "none", True),
        ("auto", False, False, "none", False),
        ("triton", False, False, "none", False),
        ("auto", True, True, "none", False),
        ("deep_gemm", True, False, "none", False),
        ("auto", True, False, "megamoe", False),
    ],
)
def test_runner_rejects_only_unsupported_packed_layout(
    monkeypatch, backend, fp4, dequant, a2a, reject
):
    method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
    method.is_fp4_expert, method.dequant_fp4_to_fp8 = fp4, dequant
    monkeypatch.setattr(fp8, "_is_hip", False)
    monkeypatch.setattr(fp8, "_is_cuda", True)
    monkeypatch.setattr(
        fp8, "get_moe_runner_backend", lambda: MoeRunnerBackend(backend)
    )
    monkeypatch.setattr(fp8, "get_moe_a2a_backend", lambda: MoeA2ABackend(a2a))
    monkeypatch.setattr(moe_utils, "get_moe_a2a_backend", lambda: MoeA2ABackend(a2a))
    runner = Mock()
    monkeypatch.setattr(fp8, "MoeRunner", runner)
    config = Mock()
    if reject:
        with pytest.raises(ValueError, match="Packed MXFP4.*FP8 Triton"):
            method.create_moe_runner(torch.nn.Module(), config)
        runner.assert_not_called()
    else:
        method.create_moe_runner(torch.nn.Module(), config)
        expected = (
            MoeRunnerBackend.DEEP_GEMM
            if backend == "deep_gemm"
            else MoeRunnerBackend.TRITON
        )
        runner.assert_called_once_with(expected, config)


def test_fp4_fallback_cannot_be_reinterpreted_as_fp8():
    method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
    method.is_fp4_expert = True
    with pytest.raises(ValueError, match="speculative-moe-runner-backend"):
        method.get_triton_quant_info(torch.nn.Module())


def test_hopper_marlin_selects_fp4_adapter(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: MoeRunnerBackend.MARLIN)
    config = fp8.Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        is_fp4_experts=True,
    )
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")
    assert isinstance(method, Mxfp4MarlinMoEMethod)
