import pytest

from sglang.benchmark.hisparse_pr40670_w6 import decode_request, input_tokens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("unique", [8, 96])
def test_same_geometry_changes_only_unique_tokens(unique):
    x = input_tokens(unique)
    assert len(x) == 96 and len(set(x)) == unique
    assert min(x) == 128 and max(x) < 576
    if unique == 96:
        assert x[4 * 16 + 4] == 196


def data(index_value=64, index_epoch=1):
    metadata = [[0] * 480 for _ in range(4)]
    counters = metadata[0]
    rid = 2
    counters[rid] = 96
    counters[9 + rid] = 1
    row = metadata[rid + 1]
    slot = 37
    row[2 * slot] = 196
    row[2 * slot + 1] = 1
    row[2 * (96 + slot)] = index_value
    row[2 * (96 + slot) + 1] = index_epoch
    row[384] = slot
    policy = [[0] * 128 for _ in range(4)]
    policy[0][rid] = 1 << 7
    policy[rid + 1][:96] = [1] * 96
    tokens = list(range(128))
    tokens[7] = 196
    return (
        metadata,
        counters,
        policy,
        tokens,
        list(range(64, 208)),
        rid,
        3,
        [196],
        [196],
    )


def test_resolved_index_with_missing_output_is_distinguished():
    r = decode_request(*data())
    assert r["protected_hot_slots"] == 96
    assert r["requested_missing_from_table"] == []
    m = r["missing"][0]
    assert m["hot_slots"] == [7]
    assert (
        m["table_matches"][0]["status"]
        == "final_index_is_physical_location_but_output_missing"
    )
    assert 0 in m["table_matches"][0]["compact_ordinals"]


@pytest.mark.parametrize(
    "value,epoch,status",
    [
        (-1, 1, "commit_marked_no_location"),
        (3, 1, "index_is_possible_scratch_ordinal"),
        (64, 0, "index_epoch_mismatch"),
        (30, 1, "unresolved_or_unclassified_index"),
    ],
)
def test_final_states_do_not_share_a_single_cause(value, epoch, status):
    r = decode_request(*data(value, epoch))
    assert r["missing"][0]["table_matches"][0]["status"] == status


def test_token_absent_from_union_table():
    args = list(data())
    args[0][3][2 * 37] = 195
    r = decode_request(*args)
    assert r["missing"][0]["classification"] == "missing_from_final_union_table"
    assert r["requested_missing_from_table"] == [196]
