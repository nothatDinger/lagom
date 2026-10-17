"""Candidate retry-accounting correction; original pinned reference is untouched."""

import hashlib

from sglang.benchmark.hisparse_pr40670_check import REFERENCE, source_manifest

PATCH_OLD = """            }
            continue;
          }
          if (static_cast<uint32_t>(old) == token_bits) {"""
PATCH_NEW = """            }
            // A lost CAS retries this same slot; it did not probe a new slot.
            --attempt;
            continue;
          }
          if (static_cast<uint32_t>(old) == token_bits) {"""
CANDIDATE = REFERENCE.parent / "pr40670_retry_candidate/hisparse_spec.cuh"


def corrected_source(text):
    if text.count(PATCH_OLD) != 1:
        raise ValueError("Pinned insertion pattern not unique")
    return text.replace(PATCH_OLD, PATCH_NEW).replace(
        '#include "hisparse_transfer.cuh"',
        '#include "../pr40670_reference/hisparse_transfer.cuh"',
    )


def validate_candidate():
    source_manifest()
    original = (REFERENCE / "hisparse_spec.cuh").read_text()
    actual = CANDIDATE.read_text()
    if actual != corrected_source(original):
        raise ValueError("Candidate contains changes beyond declared retry correction")
    return dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(actual.encode()).hexdigest(),
        functional_change="CAS failure retries same slot without consuming a probe",
        include_redirect_only=True,
    )


def simulate_insert(capacity, occupied, start, competing_wins, corrected):
    """Adversarial CPU model, not a CUDA scheduler model.

    A competing win fills a previously empty slot once; the table is monotonic
    within one gather epoch. A lost CAS then retries at that slot.
    """
    if capacity <= 0 or not 0 <= start < capacity:
        raise ValueError("Invalid model geometry")
    table = set(occupied)
    races = set(competing_wins)
    if any(not 0 <= s < capacity for s in table | races) or table & races:
        raise ValueError("Race slots must initially be empty")
    attempt = 0
    position = start
    visited = []
    iterations = 0
    while attempt < capacity:
        visited.append(position)
        iterations += 1
        if position not in table:
            if position in races:
                table.add(position)
                races.remove(position)
                if not corrected:
                    attempt += 1
                continue
            return dict(
                inserted=True,
                slot=position,
                distinct_visited=len(set(visited)),
                iterations=iterations,
            )
        position = (position + 1) % capacity
        attempt += 1
    return dict(
        inserted=False,
        slot=None,
        distinct_visited=len(set(visited)),
        iterations=iterations,
    )


def compare_results(cases):
    expected = {(v, n) for v in ("original", "retry_fixed") for n in (8, 96)}
    keys = [(c["variant"], c["unique_misses"]) for c in cases]
    if len(keys) != 4 or set(keys) != expected:
        raise ValueError("Require four unique original/candidate cases")
    lookup = dict(zip(keys, cases))

    def passed(c):
        checks = [c.get("eager", {}), *c.get("checks", []), c.get("profile_check", {})]
        return (
            c.get("success") is True
            and len(c.get("checks", [])) == 6
            and c.get("profile", {}).get("valid") is True
            and all(
                x.get("valid") is True
                and len(x.get("diagnostics", [])) == 3
                and x.get("admitted_occurrences") == 3 * 96
                and x.get("unresolved_valid_occurrences") == 0
                and not x.get("missing")
                and not x.get("other_errors")
                for x in checks
            )
            and all(
                d.get("committed_miss_count") == c["unique_misses"]
                and d.get("final_table_unique") == c["unique_misses"]
                and not d.get("requested_missing_from_table")
                for x in checks
                for d in x.get("diagnostics", [])
            )
        )

    old = lookup["original", 96].get("eager", {})
    reproduced = (
        old.get("valid") is False
        and bool(old.get("missing"))
        and not old.get("other_errors")
    )
    original_control = passed(lookup["original", 8])
    fixed_pass = all(passed(lookup["retry_fixed", n]) for n in (8, 96))
    return dict(
        original_control_pass=original_control,
        original_96_failure_reproduced=reproduced,
        retry_candidate_pass=fixed_pass,
        candidate_validated=original_control and reproduced and fixed_pass,
        original_reference_all_pass=all(passed(lookup["original", n]) for n in (8, 96)),
        note="Correctness evidence for the candidate; no timing or service claim",
    )
