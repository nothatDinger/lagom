import pytest

from sglang.benchmark.hisparse_h2d_windows import FIELDS, validate_records
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def records(blocks, variant, total=3200, real=32):
    rows = [[-1] * 15]
    for w in range(blocks * 32):
        kv = len(range(w, total, blocks * 32))
        row = [
            w // 32,
            w % 32,
            w,
            w // 32,
            100,
            110 + kv * 8,
            kv * 4,
            kv,
            real,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        row[10 if variant == "issue" else 12] = kv * 4
        rows.append(row)
    return rows + [[-1] * 15]


@pytest.mark.parametrize("blocks", [4, 8])
@pytest.mark.parametrize("variant", ["issue", "tail"])
def test_independent_window_and_schedule(blocks, variant):
    result = validate_records(records(blocks, variant), blocks, [100] * 32, 32, variant)
    assert sum(r["kv"] for r in result) == 3200
    assert {r["kv"] for r in result} == ({25} if blocks == 4 else {12, 13})
    assert all(r["interval_ns"] == r[variant + "_window_ns"] for r in result)
    assert "copy_ns" not in FIELDS


@pytest.mark.parametrize(
    "column,value", [(10, -1), (12, 1), (9, 1), (13, 1), (14, 1), (7, 0), (5, 0)]
)
def test_invalid_data_not_repaired(column, value):
    rows = records(4, "issue")
    rows[1][column] = value
    with pytest.raises(ValueError):
        validate_records(rows, 4, [100] * 32, 32, "issue")


def test_no_cross_mode_interpretation():
    with pytest.raises(ValueError):
        validate_records(records(4, "tail"), 4, [100] * 32, 32, "issue")


@pytest.mark.parametrize("variant", ["legacy", "control"])
def test_uninstrumented_records_untouched(variant):
    rows = [[-1] * 15 for _ in range(130)]
    assert validate_records(rows, 4, [], 0, variant) == []
    rows[1][6] = 0
    with pytest.raises(ValueError):
        validate_records(rows, 4, [], 0, variant)


def test_guard_and_empty_warps():
    rows = records(4, "tail", total=0, real=0)
    assert all(r["interval_ns"] == 0 for r in validate_records(rows, 4, [], 0, "tail"))
    rows[0][0] = 0
    with pytest.raises(ValueError):
        validate_records(rows, 4, [], 0, "tail")
