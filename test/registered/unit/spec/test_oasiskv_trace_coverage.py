"""Coverage uses the complete P-query union, independent of a transfer budget."""

import csv
import gzip
import json
import subprocess
import sys

import pytest
from test_oasiskv_trace import analyzer, auditor, records, runner, write_paired_run

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("width", [6, 7])
@pytest.mark.parametrize("mode", ["exact", "resident"])
def test_full_p_axis_scales_candidates_beyond_the_old_256_cap(width, mode):
    source, _, probe = records()
    k, history, resident_count = 512, 8192, 64
    source.update(
        prefix_len=32768,
        prompt_len=32768,
        history_units=history,
        verify_candidates=width,
        total_width=12,
        token_counting="generated",
        resident_after=list(range(resident_count)),
        input_ids=list(range(width + 1)),
        topk=[[0] for _ in range(width + 1)],
        miss_events=[{"row": row, "ids": []} for row in range(width + 1)],
    )
    needed = [list(range(row * k, (row + 1) * k)) for row in range(width)]
    following = dict(
        source,
        window_id=1,
        snapshot_id="snapshot-1",
        prefix_len=source["prefix_len"] + source["commit_len"],
        topk=[[7999]] + needed,
        miss_events=[{"row": 0, "ids": []}]
        + [
            {"row": row + 1, "ids": [unit for unit in ids if unit >= resident_count]}
            for row, ids in enumerate(needed)
        ],
    )
    probes = [
        dict(
            probe,
            verify_candidates=width,
            token_counting="generated",
            p=p,
            mode=mode,
            input_ids=list(range(width + 1 + p)),
            topk=needed[:p],
        )
        for p in range(13 - width)
    ]
    report = analyzer.analyze([source, following, *probes])
    assert report["prefetch_cap_units"] is None
    assert report["prefetch_policy"] == "full-probe-union"
    for row, group, p in zip(report["samples"], report["groups"], range(13 - width)):
        candidates = max(0, p * k - resident_count)
        assert row["prediction_units"] == p * k
        assert row["prediction_resident_units"] == (resident_count if p else 0)
        assert row["prefetch_candidate_units"] == candidates
        assert row["prefetch_candidate_bytes"] == candidates * 584
        assert row["history_coverage"] == pytest.approx(p / width)
        assert row["next_step_coverage"] == pytest.approx(p / width)
        assert row["next_verify_width"] == width
        assert row["total_width"] == 12
        assert row["potential_unique_miss_coverage"] == pytest.approx(
            candidates / (width * k - resident_count)
        )
        assert row["candidate_precision"] == (1 if p else None)
        assert row["prefetch_policy"] == group["prefetch_policy"] == "full-probe-union"
        assert group["prefetch_candidate_units"] == candidates
        assert (
            group["potential_unique_miss_coverage"]
            == row["potential_unique_miss_coverage"]
        )
    # P=1 already exceeds the old fixed budget; W=6/P=6 retains all 3008
    # nonresident units instead of truncating to 256 or counting 3072 transfers.
    assert report["samples"][1]["prefetch_candidate_units"] > 256


@pytest.mark.parametrize("cap", [-1, 0, 1, 256, 3072])
def test_old_fixed_caps_cannot_silently_change_the_coverage_question(cap, tmp_path):
    with pytest.raises(ValueError, match="Fixed prefetch caps"):
        analyzer.analyze(records(), cap)
    with pytest.raises(SystemExit) as exc:
        runner.build_parser().parse_args(
            ["--output", str(tmp_path), "--prefetch-units", str(cap)]
        )
    assert exc.value.code == 2
    result = subprocess.run(
        [
            sys.executable,
            analyzer.__file__,
            str(tmp_path / "unused.jsonl"),
            "--max-prefetch-units",
            str(cap),
            "--output",
            str(tmp_path / "unused-analysis"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "unused-analysis").exists()


def test_run_rejects_a_fixed_cap_even_when_cli_parser_is_bypassed(tmp_path):
    args = runner.build_parser().parse_args(["--output", str(tmp_path)])
    args.top_k, args.prefetch_units = 512, 256
    with pytest.raises(ValueError, match="Fixed prefetch caps"):
        runner.resolve_runs(args, 5)


@pytest.mark.parametrize("explicit_all", [False, True])
def test_existing_trace_can_be_reanalyzed_without_recollecting(explicit_all, tmp_path):
    source, following, probe = records()
    # Schema-v1 raw records already contain every Top-K, even if their first
    # analysis used a fixed cap. No record migration or model forward is needed.
    source.update(prefix_len=4000, prompt_len=4000, history_units=1000)
    following.update(prefix_len=4001, prompt_len=4000, history_units=1000)
    following.update(
        topk=[[999], list(range(512))],
        miss_events=[
            {"row": 0, "ids": []},
            {"row": 1, "ids": [unit for unit in range(512) if unit not in (1, 2)]},
        ],
    )
    probe["topk"] = [list(range(512)) + [3, 3, -1, 1000, 1001]]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in (source, following, probe))
    )
    output = tmp_path / "analysis-all"
    subprocess.run(
        [sys.executable, analyzer.__file__, str(trace), "--output", str(output)]
        + (["--max-prefetch-units", "all"] if explicit_all else []),
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "report.json").read_text())
    assert report["analysis_version"] == 3
    assert report["prefetch_cap_units"] is None
    assert report["samples"][0]["prefetch_candidate_units"] == 510
    assert report["samples"][0]["potential_unique_miss_coverage"] == 1
    assert report["samples"][0]["history_coverage"] == 1
    for name in ("samples", "groups", "layers"):
        with (output / f"{name}.csv").open(newline="") as stream:
            (row,) = list(csv.DictReader(stream))
        assert row["prefetch_policy"] == "full-probe-union"
        assert int(row["prefetch_candidate_units"]) == 510


def test_next_step_w_union_is_per_csa_layer_not_misses_anchor_or_accepts():
    data = []
    for layer, model_layer, forecast in ((0, 3, [3, 4]), (1, 8, [3, 9])):
        source, following, probe = records()
        for record in (source, following, probe):
            record.update(layer=layer, model_layer=model_layer, verify_candidates=2)
        source.update(
            input_ids=[20, 21, 22],
            topk=[[3, 4], [8], [8]],
            resident_after=[3, 4, 9],
            miss_events=[{"row": r, "ids": []} for r in range(3)],
        )
        following.update(
            input_ids=[30, 31, 32],
            topk=[[0, 9], [3], [4]],
            accepted_drafts=0,
            commit_len=1,
            miss_events=[{"row": r, "ids": []} for r in range(3)],
        )
        probe.update(input_ids=[20, 21, 22, 23], topk=[forecast])
        data.extend((source, following, probe))
    report = analyzer.analyze(data)
    assert report["primary_metric"] == "next_step_coverage"
    assert [r["next_step_coverage"] for r in report["layers"]] == [1, 0.5]
    assert [r["model_layer"] for r in report["layers"]] == [3, 8]
    for sample in report["samples"]:
        assert sample["next_all_units"] == 2  # Two draft rows, exclude anchor.
        assert sample["next_window_id"] == sample["window_id"] + 1
        assert sample["prefetch_candidate_units"] == 0  # All already resident.
        assert sample["potential_unique_miss_coverage"] is None


def test_probe_must_reuse_the_entire_original_proposal():
    source, following, probe = records()
    source.update(total_width=4, proposal_input_ids=[20, 21, 25, 26])
    probe.update(total_width=4, proposal_input_ids=[20, 21, 25, 27])
    with pytest.raises(ValueError, match="same complete draft proposal"):
        analyzer.analyze([source, following, probe])


def test_layer_curve_combines_acceptance_strata_but_preserves_step_ratios():
    source, following, probe = records()
    probe["topk"] = [[3]]
    following.update(accepted_drafts=1, commit_len=2)
    last = dict(
        following,
        window_id=2,
        snapshot_id="snapshot-2",
        prefix_len=43,
        topk=[[], [3]],
        miss_events=[{"row": 0, "ids": []}, {"row": 1, "ids": [3]}],
    )
    second = dict(
        probe,
        window_id=1,
        snapshot_id=following["snapshot_id"],
        input_ids=following["input_ids"] + [25],
    )
    report = analyzer.analyze([source, following, last, probe, second])
    assert [r["next_step_coverage"] for r in report["samples"]] == [0.5, 1]
    assert len(report["groups"]) == 2
    (layer,) = report["layers"]
    assert layer["mean_step_coverage"] == 0.75
    assert layer["next_step_coverage"] == pytest.approx(2 / 3)
    assert layer["steps_with_nonempty_demand"] == 2


@pytest.mark.parametrize("tamper_suffix", [False, True])
def test_generated_protocol_audit_binds_every_p_to_the_full_proposal(
    tmp_path, tamper_suffix
):
    directory = tmp_path / "probes"
    write_paired_run(directory)
    protocol = {"token_counting": "generated", "proposal_gamma": 3, "total_width": 3}
    for name in ("config.json", "runtime.json"):
        path = directory / name
        value = json.loads(path.read_text())
        path.write_text(json.dumps(dict(value, **protocol)))
    for name in ("trace.jsonl.gz", "paired-baseline.jsonl.gz", "replay-audit.jsonl"):
        path = directory / name
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            row.update(protocol)
            row["proposal_input_ids"] = row["input_ids"][:2] + [25, 26]
            if tamper_suffix and row.get("kind") == "probe":
                row["proposal_input_ids"][-1] = 27  # Unused suffix still must agree.
        with opener(path, "wt") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in rows))
    if tamper_suffix:
        with pytest.raises(ValueError, match="different snapshot/proposal"):
            auditor.audit_within_server(directory)
    else:
        audit = auditor.audit_within_server(directory)
        assert audit["passed"] and audit["verify_width"] == 1
        assert audit["target_verify_width"] == 2
