"""Coverage uses the complete P-query union, independent of a transfer budget."""

import csv
import json
import subprocess
import sys

import pytest
from test_oasiskv_trace import analyzer, records, runner

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
        verify_candidates=width - 1,
        total_width=12,
        resident_after=list(range(resident_count)),
        input_ids=list(range(width)),
        topk=[[0] for _ in range(width)],
        miss_events=[{"row": row, "ids": []} for row in range(width)],
    )
    needed = [list(range(row * k, (row + 1) * k)) for row in range(width)]
    following = dict(
        source,
        window_id=1,
        snapshot_id="snapshot-1",
        prefix_len=source["prefix_len"] + source["commit_len"],
        topk=needed,
        miss_events=[
            {"row": row, "ids": [unit for unit in ids if unit >= resident_count]}
            for row, ids in enumerate(needed)
        ],
    )
    probes = [
        dict(
            probe,
            verify_candidates=width - 1,
            p=p,
            mode=mode,
            input_ids=list(range(width + p)),
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
        topk=[list(range(512)), []],
        miss_events=[
            {"row": 0, "ids": [unit for unit in range(512) if unit not in (1, 2)]},
            {"row": 1, "ids": []},
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
    assert report["analysis_version"] == 2
    assert report["prefetch_cap_units"] is None
    assert report["samples"][0]["prefetch_candidate_units"] == 510
    assert report["samples"][0]["potential_unique_miss_coverage"] == 1
    assert report["samples"][0]["history_coverage"] == 1
    for name in ("samples", "groups"):
        with (output / f"{name}.csv").open(newline="") as stream:
            (row,) = list(csv.DictReader(stream))
        assert row["prefetch_policy"] == "full-probe-union"
        assert int(row["prefetch_candidate_units"]) == 510
