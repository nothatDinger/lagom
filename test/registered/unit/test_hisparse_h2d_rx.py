import copy
import threading

import pytest
from sglang.benchmark.hisparse_h2d_pair import abba_order
from sglang.benchmark.hisparse_h2d_rx import (
    RxSampler,
    ce_calibration,
    rx_compact_report,
    sampled_rates,
    window_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_weighting_handles_irregular_sampling_and_clipped_edges():
    rows = [dict(t_ns=t, rx_kbps=2 * t + 100, tx_kbps=20) for t in (0, 50, 200, 1000)]
    m = sampled_rates(rows, 40, 800, min_samples=2, max_gap_ns=2000)
    assert m["valid"] and m["coverage"] == 1
    assert m["rx_kbps"] == pytest.approx(940)
    assert m["tx_kbps"] == 20
    # An unweighted point mean would be 350; do not average sample points.


@pytest.mark.parametrize(
    "fault", ["gap", "missing_edge", "error", "negative", "duplicate"]
)
def test_missing_counter_data_is_never_extrapolated_or_treated_as_zero(fault):
    rows = [dict(t_ns=t, rx_kbps=100, tx_kbps=10) for t in range(0, 1001, 20)]
    if fault == "gap":
        rows = [r for r in rows if r["t_ns"] < 200 or r["t_ns"] > 800]
    elif fault == "missing_edge":
        rows = rows[15:]
    elif fault == "error":
        rows[20]["error"] = "NVMLError_NotSupported"
    elif fault == "negative":
        rows[20]["rx_kbps"] = -1
    else:
        rows[20]["t_ns"] = rows[19]["t_ns"]
    assert not sampled_rates(rows, 100, 900, min_samples=20, max_gap_ns=80)["valid"]


def synthetic_window(logical_GBps=55, rx_GBps=55):
    """Analysis fixture, not GPU or PCIe measurements."""
    chunks = [
        dict(
            begin_ns=i * 100_000_000,
            end_ns=(i + 1) * 100_000_000,
            replays=10,
            enqueue_ns=100_000,
            gpu_us=99_000,
        )
        for i in range(30)
    ]
    return dict(
        chunks=chunks,
        bytes_per_graph=int(logical_GBps * 1e9 * 0.1 / 10),
        samples=[
            dict(t_ns=t, rx_kbps=rx_GBps * 1e9 / 1024, tx_kbps=100)
            for t in range(0, 3_000_000_001, 20_000_000)
        ],
        idle_samples=[
            dict(t_ns=t, rx_kbps=0, tx_kbps=0)
            for t in range(0, 500_000_001, 20_000_000)
        ],
    )


def test_logical_wall_rate_and_rx_share_the_exact_core_interval():
    window = synthetic_window()
    m = window_metrics(window, 0.02, 0.3)
    assert m["valid"] and m["core_chunks"] == 24
    assert m["core_seconds"] == 2.4
    assert m["logical_wall_GBps"] == pytest.approx(55)
    assert m["logical_gpu_GBps"] == pytest.approx(55 / 0.99)
    assert m["rx_GBps_1024"] == pytest.approx(55)
    assert ce_calibration(m)["valid"]


def test_large_rx_to_logical_ratio_is_preserved_not_clamped():
    m = window_metrics(synthetic_window(23, 55), 0.02, 0.3)
    assert m["valid"] and m["rx_GBps_1024"] / m["logical_wall_GBps"] == pytest.approx(
        55 / 23
    )
    assert not ce_calibration(m)["valid"]  # Only CE may be used for calibration.


@pytest.mark.parametrize(
    "fault",
    [
        "host_gaps",
        "slow_enqueue",
        "unstable",
        "background",
        "too_short",
        "overlap",
        "zero_rx",
    ],
)
def test_unreliable_windows_fail_before_causal_comparison(fault):
    w = synthetic_window()
    if fault == "host_gaps":
        for c in w["chunks"]:
            c["gpu_us"] = 80_000
    elif fault == "slow_enqueue":
        for c in w["chunks"]:
            c["enqueue_ns"] = 20_000_000
    elif fault == "unstable":
        for i, r in enumerate(w["samples"]):
            r["rx_kbps"] *= 0.5 if i % 2 else 1.5
    elif fault == "background":
        for r in w["idle_samples"]:
            r["rx_kbps"] = w["samples"][0]["rx_kbps"] * 0.1
    elif fault == "too_short":
        w["chunks"] = w["chunks"][:3]
    elif fault == "overlap":
        w["chunks"][10]["begin_ns"] -= 10_000_000
    else:
        for r in w["samples"]:
            r["rx_kbps"] = 0
    assert not window_metrics(w, 0.02, 0.3)["valid"]


@pytest.mark.parametrize("logical,rx", [(55, 0), (55, 110), (30, 30)])
def test_ce_must_match_counters_and_reach_a_useful_reference_rate(logical, rx):
    m = window_metrics(synthetic_window(logical, rx), 0.02, 0.3)
    assert not ce_calibration(m)["valid"]


def test_sampler_stops_and_keeps_reader_failures():
    sampler = RxSampler(lambda: dict(rx_kbps=100, tx_kbps=10), 0.001).start()
    threading.Event().wait(0.02)
    sampler.stop()
    assert sampler.rows and not sampler.thread.is_alive()

    def unavailable():
        raise RuntimeError("NVML counter not supported")

    failed = RxSampler(unavailable, 0.001).start()
    failed.thread.join(timeout=1)
    failed.stop()
    assert failed.rows and "not supported" in failed.rows[0]["error"]


def synthetic_report():
    order = abba_order([0, 1], 1)
    report = dict(
        success=True,
        counters_available=True,
        config=dict(
            devices=[0, 1], order=order, duration_s=3, trim_s=0.3, sample_ms=20
        ),
        correctness=[
            dict(device=d, method=m, payload=True, graph=True, untouched=m == "sm")
            for d in (0, 1)
            for m in ("sm", "ce_bulk")
        ],
        windows=[],
    )
    for trial, device in enumerate(order):
        for method in ("ce_bulk", "sm"):
            w = synthetic_window(
                55 if method == "ce_bulk" else 23, 55 if method == "ce_bulk" else 50
            )
            w.update(device=device, trial=trial, method=method)
            w["metrics"] = window_metrics(w, 0.02, 0.3)
            if method == "ce_bulk":
                w["calibration"] = ce_calibration(w["metrics"])
            report["windows"].append(w)
    return report


def test_compact_reports_supported_calibration_and_raw_ratios():
    text = rx_compact_report(synthetic_report())
    assert "CE_calibration=4/4 results_valid=True" in text
    assert "SM/CE_rx=[0.9091, 0.9091]" in text
    assert "SM/CE_logical=[0.4182, 0.4182]" in text
    assert "rx/logical=2.174" in text


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "calibration", "counter", "correctness"]
)
def test_incomplete_or_failed_measurements_have_no_comparison(fault):
    report = synthetic_report()
    if fault == "missing":
        report["windows"].pop()
    elif fault == "duplicate":
        report["windows"].append(copy.deepcopy(report["windows"][0]))
    elif fault == "calibration":
        report["windows"][0]["calibration"]["valid"] = False
    elif fault == "counter":
        report["counters_available"] = False
    else:
        report["correctness"][0]["graph"] = False
    text = rx_compact_report(report)
    assert "results_valid=False" in text and "PAIRS " not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
