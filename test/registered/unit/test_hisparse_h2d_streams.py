import pytest
from sglang.benchmark.hisparse_h2d_streams import activity_overlap, merge
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def kernel(stream, start, duration=10):
    return dict(
        cat="kernel",
        ph="X",
        name="copy_cache_planned_kernel",
        ts=start,
        dur=duration,
        args=dict(device=0, stream=stream, grid=[4, 1, 1], block=[1024, 1, 1]),
    )


def test_real_activity_intersection_not_envelope_overlap():
    trace = dict(
        traceEvents=[kernel(1, 0), kernel(1, 20), kernel(2, 10), kernel(2, 30)]
    )
    r = activity_overlap(trace, 2, True)
    assert r["intersection_us"] == 0
    assert r["envelope_us"] == 40
    assert r["overlap_fraction_shorter"] == 0


def test_partial_overlap():
    trace = dict(traceEvents=[kernel(1, 0), kernel(1, 20), kernel(2, 5), kernel(2, 25)])
    r = activity_overlap(trace, 2, True)
    assert r["intersection_us"] == 10
    assert r["overlap_fraction_shorter"] == 0.5


def test_serial_and_missing_records():
    trace = dict(traceEvents=[kernel(1, 0), kernel(1, 10)])
    assert activity_overlap(trace, 1, False)["intersection_us"] == 0
    with pytest.raises(ValueError):
        activity_overlap(trace, 1, True)
    with pytest.raises(ValueError):
        activity_overlap(trace, 2, False)


@pytest.mark.parametrize(
    "field,value", [("grid", [8, 1, 1]), ("block", [512, 1, 1]), ("device", 1)]
)
def test_wrong_launch_rejected(field, value):
    a = kernel(1, 0)
    a["args"][field] = value
    with pytest.raises(ValueError):
        activity_overlap(dict(traceEvents=[a, kernel(2, 0)]), 1, True)


def test_union_and_invalid_duration():
    assert merge([(0, 10), (5, 20), (30, 40)]) == [[0, 20], [30, 40]]
    with pytest.raises(ValueError):
        merge([(1, 1)])


def memcpy(stream, start, duration=20):
    return dict(
        cat="gpu_memcpy",
        ph="X",
        name="Memcpy HtoD (Pinned -> Device)",
        ts=start,
        dur=duration,
        args=dict(device=0, stream=stream, bytes=16 * 2**20),
    )


def test_mixed_overlap_reports_both_denominators():
    from sglang.benchmark.hisparse_h2d_sm_ce import mixed_overlap

    r = mixed_overlap(
        dict(traceEvents=[kernel(1, 0), memcpy(2, 5)]), 1, 1, 16 * 2**20, True
    )
    assert r["intersection_us"] == 5
    assert r["overlap_fraction"] == {"SM": 0.5, "CE": 0.25}


def test_mixed_serial_and_bad_coverage():
    from sglang.benchmark.hisparse_h2d_sm_ce import mixed_overlap

    trace = dict(traceEvents=[kernel(1, 0), memcpy(1, 10)])
    assert mixed_overlap(trace, 1, 1, 16 * 2**20, False)["intersection_us"] == 0
    for banks, copies, size, parallel in [
        (2, 1, 16 * 2**20, False),
        (1, 2, 16 * 2**20, False),
        (1, 1, 32, False),
        (1, 1, 16 * 2**20, True),
    ]:
        with pytest.raises(ValueError):
            mixed_overlap(trace, banks, copies, size, parallel)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
