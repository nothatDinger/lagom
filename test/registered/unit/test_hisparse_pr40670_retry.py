from copy import deepcopy

import pytest

from sglang.benchmark.hisparse_pr40670_retry import (
    compare_results,
    corrected_source,
    simulate_insert,
    validate_candidate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_lost_cas_consumes_original_budget_before_last_free_slot():
    occupied = set(range(96)) - {35, 36}
    original = simulate_insert(96, occupied, 36, {36}, False)
    fixed = simulate_insert(96, occupied, 36, {36}, True)
    assert not original["inserted"] and original["distinct_visited"] == 95
    assert fixed["inserted"] and fixed["slot"] == 35
    assert fixed["distinct_visited"] == 96


@pytest.mark.parametrize("corrected", [False, True])
def test_full_table_terminates(corrected):
    result = simulate_insert(96, set(range(96)), 36, set(), corrected)
    assert not result["inserted"] and result["iterations"] == 96


def test_multiple_lost_cas_are_bounded_by_monotonic_fills():
    result = simulate_insert(4, set(), 0, {0, 1, 2, 3}, True)
    assert not result["inserted"] and result["iterations"] == 8


def test_no_contention_preserves_result():
    args = (96, set(range(50)), 36, set())
    assert simulate_insert(*args, False) == simulate_insert(*args, True)


def test_candidate_is_exact_declared_patch():
    assert validate_candidate()["include_redirect_only"]
    with pytest.raises(ValueError, match="not unique"):
        corrected_source("different source")


def cases():
    rows = []
    for variant in ("original", "retry_fixed"):
        for unique in (8, 96):
            check = dict(
                valid=True,
                admitted_occurrences=288,
                unresolved_valid_occurrences=0,
                missing=[],
                other_errors=[],
                diagnostics=[
                    dict(
                        committed_miss_count=unique,
                        final_table_unique=unique,
                        requested_missing_from_table=[],
                    )
                    for _ in range(3)
                ],
            )
            rows.append(
                dict(
                    variant=variant,
                    unique_misses=unique,
                    success=True,
                    eager=deepcopy(check),
                    checks=[deepcopy(check) for _ in range(6)],
                    profile_check=deepcopy(check),
                    profile=dict(valid=True),
                )
            )
    rows[1]["eager"].update(valid=False, missing=[{"token": 196}])
    rows[1]["success"] = False
    return rows


def test_comparison_preserves_original_failure():
    result = compare_results(cases())
    assert result["candidate_validated"]
    assert not result["original_reference_all_pass"]


@pytest.mark.parametrize(
    "fault", ["diagnostics", "graphs", "profile", "count", "missing"]
)
def test_incomplete_or_failed_fixed_checks_are_rejected(fault):
    rows = cases()
    row = rows[3]
    if fault == "diagnostics":
        row["checks"][0]["diagnostics"] = []
    elif fault == "graphs":
        row["checks"].pop()
    elif fault == "profile":
        row["profile"]["valid"] = False
    elif fault == "count":
        row["profile_check"]["diagnostics"][0]["committed_miss_count"] = 95
    else:
        row["checks"][0]["missing"] = [{"token": 196}]
    assert not compare_results(rows)["candidate_validated"]


def test_original_not_reproduced_is_inconclusive():
    rows = cases()
    rows[1]["eager"].update(valid=True, missing=[])
    rows[1]["success"] = True
    result = compare_results(rows)
    assert result["original_reference_all_pass"]
    assert not result["candidate_validated"]
