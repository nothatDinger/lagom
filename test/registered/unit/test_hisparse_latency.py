import gzip
import json

import pytest

from sglang.benchmark.hisparse_latency import (
    compare_skip_io,
    read_benchmark,
    summarize_trace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _kernel(name, ts, dur, device=0):
    return {
        "name": name,
        "ts": ts,
        "dur": dur,
        "ph": "X",
        "cat": "kernel",
        "args": {"device": device},
    }


def test_trace_deduplicates_overlap_and_normalizes_speculative_tokens(tmp_path):
    events = [
        _kernel("load_cache_to_device_buffer_spec_gather_kernel<64>", 0, 100),
        _kernel("load_cache_to_device_buffer_spec_commit_kernel<64>", 100, 100),
        _kernel("copy_cache_planned_kernel<256>", 150, 150),
        _kernel("initialize_hisparse_spec_state_kernel<8>", 300, 100),
        _kernel("attention", 0, 900),
        _kernel("copy_cache_planned_kernel<256>", 0, 900, device=1),
        {
            "name": "copy_cache_planned_kernel",
            "ph": "X",
            "cat": "cpu_op",
            "ts": 0,
            "dur": 900,
        },
    ]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    with pytest.raises(ValueError, match="one GPU"):
        summarize_trace(path, decode_steps=2, tokens_per_step=3, tpot_ms=1)
    report = summarize_trace(
        path, decode_steps=2, tokens_per_step=3, tpot_ms=1, device=0
    )
    assert report["plan_io_gpu_sum_us"] == 350
    assert report["plan_io_gpu_union_us"] == 300
    assert report["plan_io_gpu_ms_per_output_token"] == pytest.approx(0.05)
    assert report["plan_io_gpu_share_of_tpot_pct"] == pytest.approx(5)
    assert report["kernels"]["hash_initialization"]["gpu_us"] == 100
    sliced = summarize_trace(
        path,
        decode_steps=1,
        tokens_per_step=1,
        tpot_ms=1,
        device=0,
        start_us=100,
        end_us=250,
    )
    assert sliced["plan_io_gpu_union_us"] == 150


def test_fused_main_kernel_and_missing_graph_nodes(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    _kernel("load_cache_to_device_buffer_kernel<960>", 0, 100)
                ]
            }
        )
    )
    report = summarize_trace(path, decode_steps=1, tokens_per_step=1, tpot_ms=1)
    assert report["kernels"]["fused_plan_io"]["count"] == 1
    path.write_text(json.dumps({"traceEvents": [_kernel("cudaGraphLaunch", 0, 100)]}))
    with pytest.raises(ValueError):
        summarize_trace(path, decode_steps=1, tokens_per_step=1, tpot_ms=1, device=0)


def test_skip_io_is_separate_from_plan_io_and_keeps_negative_deltas():
    normal = {"mean_tpot_ms": 10, "random_input_len": 1024}
    skip = {"mean_tpot_ms": 8, "random_input_len": 1024}
    report = compare_skip_io(normal, skip)
    assert report["exposed_io_share_pct"] == 20
    skip["mean_tpot_ms"] = 11
    assert compare_skip_io(normal, skip)["exposed_io_share_pct"] == -10
    skip["random_input_len"] = 512
    with pytest.raises(ValueError, match="random_input_len"):
        compare_skip_io(normal, skip)


@pytest.mark.parametrize("tpot", [0, -1, float("nan"), float("inf")])
def test_invalid_tpot_cannot_produce_a_percentage(tmp_path, tpot):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"mean_tpot_ms": tpot}))
    with pytest.raises(ValueError, match="positive"):
        read_benchmark(path)


def test_result_selection_for_append_only_serving_jsonl(tmp_path):
    path = tmp_path / "result.jsonl"
    path.write_text('{"mean_tpot_ms": 3}\n{"mean_tpot_ms": 4}\n')
    assert read_benchmark(path)["mean_tpot_ms"] == 4
    assert read_benchmark(path, 0)["mean_tpot_ms"] == 3
