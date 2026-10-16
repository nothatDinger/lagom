import pytest

from sglang.benchmark.hisparse_h2d_segments import validate_records
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def records(blocks, total=3200, real=32):
    rows = [[-1] * 15]
    for w in range(blocks * 32):
        kv = len(range(w, total, blocks * 32))
        rows.append(
            [
                w // 32,
                w % 32,
                w,
                w // 32,
                100,
                110 + kv * 4,
                kv * 4,
                kv,
                real,
                kv,
                kv,
                kv,
                kv,
                0,
                0,
            ]
        )
    return rows + [[-1] * 15]


@pytest.mark.parametrize("blocks", [4, 8])
def test_schedule_and_segment_accounting(blocks):
    parsed = validate_records(records(blocks), blocks, [100] * 32, 32)
    assert sum(r["kv"] for r in parsed) == 3200
    assert all(
        r["copy_ns"]
        == sum(r[k] for k in ("prep_ns", "issue_ns", "residual_ns", "store_ns"))
        for r in parsed
    )


@pytest.mark.parametrize("col,value", [(9, -1), (10, 999), (14, 1), (7, 0)])
def test_invalid_segment_or_schedule_rejected(col, value):
    rows = records(4)
    rows[1][col] = value
    with pytest.raises(ValueError):
        validate_records(rows, 4, [100] * 32, 32)


def test_guards_and_control_records():
    rows = [[-1] * 15 for _ in range(130)]
    assert validate_records(rows, 4, [], 0, False) == []
    rows[0][14] = 0
    with pytest.raises(ValueError):
        validate_records(rows, 4, [], 0, False)


def test_empty_warps_have_no_copy_data():
    rows = records(4, total=0, real=0)
    assert all(r["kv"] == 0 for r in validate_records(rows, 4, [], 0))
    rows[1][13] = 1
    with pytest.raises(ValueError):
        validate_records(rows, 4, [], 0)
