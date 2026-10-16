import pytest
from collections import Counter
from sglang.benchmark.hisparse_h2d_blockshape import (
    CONFIGS,
    ORDERS,
    schedule,
    summarize,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_equal_warp_work_and_request_visits():
    a, b, c = [schedule(*v) for v in CONFIGS.values()]
    assert a["total_warps"] == b["total_warps"] == 128
    assert a["visits"] == b["visits"] == 4096
    assert a["kv_per_warp"] == b["kv_per_warp"] == [25]
    assert a["kv_per_cta"] == [800] * 4 and b["kv_per_cta"] == [400] * 8
    assert c["total_warps"] == 256 and c["visits"] == 8192
    assert c["kv_per_warp"] == [12, 13]
    assert all(s["total_kv"] == 3200 for s in (a, b, c))
    # Original request traversal arithmetic vs flat ownership for equal-warps shapes.
    for blocks, threads in CONFIGS.values():
        W = blocks * threads // 32
        for w in range(W):
            actual = [
                r * 100 + m for r in range(32) for m in range((w - r * 100) % W, 100, W)
            ]
            assert actual == list(range(w, 3200, W))


def test_balanced_orders():
    for label in CONFIGS:
        assert Counter(order.index(label) for order in ORDERS) == {0: 2, 1: 2, 2: 2}


def waves():
    return [
        dict(trial=t, label=k, copy_us=[60.0] * 64) for t in range(6) for k in CONFIGS
    ]


def test_ratios():
    data = waves()
    for w in data:
        if w["label"] == "c8_t512":
            w["copy_us"] = [48.0] * 64
    r = summarize(data)
    assert r["ratios"]["c8_t512/c4_t1024"]["median"] == 0.8
    assert r["ratios"]["c8_t512/c4_t1024"]["faster"] == 6


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nan", "short"])
def test_bad_coverage(kind):
    data = waves()
    if kind == "missing":
        data.pop()
    elif kind == "duplicate":
        data[-1] = data[0]
    elif kind == "nan":
        data[0]["copy_us"][0] = float("nan")
    else:
        data[0]["copy_us"].pop()
    with pytest.raises(ValueError):
        summarize(data)
