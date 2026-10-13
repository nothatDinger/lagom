"""Check the CUDA test's actual fixture and oracle without requiring a GPU."""

import importlib.util
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def harness():
    path = (
        Path(__file__).resolve().parents[1]
        / "kernels/ops/kvcache/test_hisparse_copy_load_policy.py"
    )
    spec = importlib.util.spec_from_file_location("copy_policy_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_with_kernel_addresses(case, active_counts, real_count):
    # Independent physical layout contract, not the fixture's row stride:
    # 36864 value + 512 scale + 64 padding = 37440 bytes per C4 page.
    src_flat, dst_flat = case.host.view(-1), case.output.view(-1)
    case.output.fill_(165)
    for request in range(real_count):
        for miss in range(active_counts[request]):
            source, destination = int(case.src[request, miss]), int(
                case.dst[request, miss]
            )
            sp, so, dp, do = (
                source >> 6,
                source & 63,
                destination >> 6,
                destination & 63,
            )
            for offset, width in ((0, 576), (36864, 8)):
                sb = sp * 37440 + offset + so * width
                db = dp * 37440 + offset + do * width
                dst_flat[db : db + width] = src_flat[sb : sb + width]


@pytest.mark.parametrize(
    "active_counts,real_count",
    [([0, 1, 65, 9], 3), ([0, 0, 0, 9], 3), ([0, 1, 65, 9], 2), ([0, 1, 65, 9], 0)],
)
def test_fixture_and_oracle_match_physical_pages(harness, active_counts, real_count):
    case = harness.make_copy_case("cpu")
    assert case.host.shape == (4, 37440)
    assert case.output.shape == (5, 37440)
    assert case.src.stride() == case.dst.stride() == (96, 1)
    assert case.dst_cpu[1, 0] == 63 and case.dst_cpu[2, 0] == 64
    for _ in range(2):
        case.host.bitwise_xor_(255)
        copy_with_kernel_addresses(case, active_counts, real_count)
        harness.assert_copy_matches(
            case, active_counts, real_count, policy="cpu_layout", phase="regression"
        )


def test_old_38016_byte_fixture_reproduces_reported_failure(harness):
    case = harness.make_copy_case("cpu", page_bytes=38016)
    copy_with_kernel_addresses(case, [0, 1, 65, 9], 3)
    expected = harness.expected_copy(case, [0, 1, 65, 9], 3)
    assert case.output[1, :3].tolist() == [251, 35, 250]
    assert expected[1, :3].tolist() == [181, 223, 108]
    with pytest.raises(AssertionError, match="page_bytes=38016") as error:
        harness.assert_copy_matches(
            case, [0, 1, 65, 9], 3, policy="cpu_layout", phase="old_fixture"
        )
    message = str(error.value)
    assert "src_stride=(96, 1) dst_stride=(96, 1)" in message
    assert "mismatched_bytes_per_tensor_row=" in message
    assert message.count("'flat_byte':") == 8
    assert len(message) < 2000  # Keep a useful error small enough to paste back.
