import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("deepseek_v4_csa_analyze", MODULE_PATH)
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


def test_commit_records_bind_to_repeated_decode_step_occurrences():
    traces = [
        {
            "decode_step": 0,
            "layer_id": 0,
            "h2d_ms": 1.0,
            "miss_tokens_per_request": [3],
        },
        {
            "event": "commit",
            "decode_step": 0,
            "cumulative_accepted_tokens": [2],
        },
        {
            "decode_step": 0,
            "layer_id": 0,
            "h2d_ms": 2.0,
            "miss_tokens_per_request": [5, 7],
        },
        {
            "event": "commit",
            "decode_step": 0,
            "cumulative_accepted_tokens": [1, 4],
        },
    ]

    cycles = ANALYZE.aggregate_trace_cycles(traces)

    assert [cycle["misses"] for cycle in cycles] == [[3], [5, 7]]
    assert [cycle["commit"]["cumulative_accepted_tokens"] for cycle in cycles] == [
        [2],
        [1, 4],
    ]


def test_layers_are_aggregated_before_their_commit():
    traces = [
        {
            "decode_step": 3,
            "layer_id": 0,
            "h2d_ms": 1.25,
            "miss_tokens_per_request": [2],
        },
        {
            "decode_step": 3,
            "layer_id": 1,
            "h2d_ms": 0.75,
            "miss_tokens_per_request": [4],
        },
        {
            "event": "commit",
            "decode_step": 3,
            "cumulative_accepted_tokens": [6],
        },
    ]

    cycle = ANALYZE.aggregate_trace_cycles(traces)[0]

    assert cycle["h2d_ms"] == 2.0
    assert cycle["layers"] == 2
    assert cycle["misses"] == [6]
    assert cycle["commit"]["cumulative_accepted_tokens"] == [6]


def test_legacy_padded_zero_request_rows_are_ignored():
    cycle = {
        "decode_step": 8,
        "misses": [11, 0, 0],
        "commit": {"cumulative_accepted_tokens": [5]},
    }

    assert ANALYZE.align_cycle_requests(cycle) == ([11], [5])


def test_request_ids_align_values_in_commit_order():
    cycle = {
        "decode_step": 2,
        "misses": [7, 3],
        "request_pool_indices": [20, 10],
        "commit": {
            "cumulative_accepted_tokens": [4, 9],
            "request_pool_indices": [10, 20],
        },
    }

    assert ANALYZE.align_cycle_requests(cycle) == ([3, 7], [4, 9])
