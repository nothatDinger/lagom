import pytest
import torch
from sglang.benchmark.hisparse_h2d_linear import EDGE_BYTES, PAYLOAD, check_edge_case
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("blocks", [4, 8])
@pytest.mark.parametrize("size", EDGE_BYTES)
@pytest.mark.parametrize("source", ["host", "hbm"])
def test_linear_payload_guards_and_reset_graph(blocks, size, source):
    host = torch.randint(0, 256, (PAYLOAD + 8,), dtype=torch.uint8, pin_memory=True)
    result = check_edge_case(torch, host, 0, blocks, size, source)
    assert result["eager"] and result["graph"] and result["guards"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
