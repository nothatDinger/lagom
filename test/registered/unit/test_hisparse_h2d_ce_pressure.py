import pytest

from sglang.benchmark.hisparse_h2d_ce_pressure import core_metrics
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_same_core_exact_sm_bounded_ce():
    sm = [(i * 10, (i + 1) * 10) for i in range(10)]
    r = core_metrics(sm, [(15, 25), (30, 40), (75, 85)], 1_000_000, 1_000_000)
    assert (r["core_start_us"], r["core_end_us"]) == (20, 80)
    assert r["sm_GBps"] == 100
    assert r["ce_complete"] == 1
    assert r["ce_intersecting"] == 3
    assert r["ce_GBps_lower"] == pytest.approx(1000 / 60)
    assert r["ce_GBps_upper"] == 50
    assert r["ce_event_duty"] == pytest.approx(1 / 3)


def test_no_ce_and_sm_gaps_are_counted_in_wall_interval():
    sm = [(i * 10, i * 10 + 8) for i in range(10)]
    r = core_metrics(sm, [], 1000, 1000)
    assert r["ce_GBps_lower"] == r["ce_GBps_upper"] == r["ce_event_duty"] == 0
    assert r["sm_event_coverage"] < 0.9


@pytest.mark.parametrize(
    "sm,ce",
    [([], []), ([(0, 0)] * 10, []), ([(i, i + 1) for i in range(10)], [(3, 2)])],
)
def test_bad_intervals_rejected(sm, ce):
    with pytest.raises(ValueError):
        core_metrics(sm, ce, 1000, 1000)
