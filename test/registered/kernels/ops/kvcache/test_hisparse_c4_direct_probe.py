import pytest
import torch
from sglang.benchmark.hisparse_h2d_diagnostics import PAGE_BYTES
from sglang.benchmark.hisparse_h2d_direct import EDGE_COUNTS, check_edge
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("count", EDGE_COUNTS)
@pytest.mark.parametrize("source", ["host", "hbm"])
def test_direct_c4_payload_padding_and_reset_graph(count, source):
    # Reuse the independent full-output oracle used by the acquisition scripts.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[5] / "scripts/pd_test/bench_h2d_copy.py"
    spec = importlib.util.spec_from_file_location("direct_copy_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    host = torch.randint(0, 256, (53, PAGE_BYTES), dtype=torch.uint8, pin_memory=True)
    result = check_edge(torch, host, 0, count, source, module.validate_sm)
    assert result["eager"] and result["graph"] and result["all_bytes"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
