"""Byte-exact eager/graph checks for legacy and cross-request staging."""

import sys

import pytest
import torch
from sglang.benchmark.hisparse_h2d_iterator import (
    VARIANTS,
    make_iterator_case,
    run_iterator_case,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("source", ["host", "hbm"])
def test_iterator_eager_and_dynamic_graph(variant, source):
    with torch.cuda.device(0):
        result = run_iterator_case(
            torch, make_iterator_case(torch, "cuda:0"), variant, source
        )
    assert result["eager"] and result["all_bytes_and_padding"]
    assert len(result["graph_phases"]) == 6
    assert result["capture_stream_device"] == result["device"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
