"""Fixed proposal geometry, long-context sweep and explicit P=0 baseline."""

import json
import subprocess
import sys

import pytest
from test_oasiskv_trace import analyzer, records, runner

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def args_for(tmp_path, *extra):
    args = runner.build_parser().parse_args(
        ["--mode", "paired", "--output", str(tmp_path / "results"), *extra]
    )
    args.top_k = 512
    return args


def test_matrix_uses_fixed_g12_and_common_capacity(tmp_path):
    args = args_for(tmp_path, "--verify-widths", "6", "7")
    runs = runner.resolve_runs(args, 5)
    assert len(runs) == 2
    for w, run in zip((6, 7), runs):
        assert runner.real_candidates(run) == w - 1
        assert run.probes == list(range(13 - w))
        assert run.input_lengths == [16384, 32768, 65536, 131072]
        assert run.context_length == 139264
        assert run.resident_units == 3584
        assert run.prefetch_units == "all"
        assert run.output == args.output / f"w{w}"
        command = runner.server_command(run, 5)
        assert command[command.index("--speculative-num-draft-tokens") + 1] == "12"
        assert command[command.index("--speculative-dspark-block-size") + 1] == "11"
    assert args.context_length is None and args.resident_units is None


@pytest.mark.parametrize("probes", [[0], [1], [0, 2, 4, 6]])
def test_p_subset_does_not_shrink_proposal(tmp_path, probes):
    args = args_for(tmp_path, "--probes", *map(str, probes))
    (run,) = runner.resolve_runs(args, 5)
    assert run.probes == probes
    assert runner.proposal_gamma(run, 5) == 11


@pytest.mark.parametrize(
    "flags,reason",
    [
        (["--verify-width", "7", "--probes", "0", "6"], "W \\+ max"),
        (
            ["--verify-widths", "6", "7", "--resident-units", "3072"],
            "Real verify requires",
        ),
        (["--context-length", "40960"], "Context length must cover"),
        (["--host-to-device-ratio", "2"], "global prefill C4 pool"),
        (["--total-width", "5"], "must cover every real"),
        (["--probes", "-1"], "nonnegative"),
    ],
)
def test_rejects_invalid_sweep_before_launch(tmp_path, flags, reason):
    args = args_for(tmp_path, *flags)
    with pytest.raises(ValueError, match=reason):
        runner.resolve_runs(args, 5)
    assert not args.output.exists()


def test_cli_dry_run_delivers_both_twelve_row_server_commands(tmp_path):
    for name, config in (
        ("target", {"index_topk": 512}),
        ("draft", {"dspark_block_size": 5}),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            str(runner.HERE / "run.py"),
            "--mode",
            "paired",
            "--verify-widths",
            "6",
            "7",
            "--target",
            str(tmp_path / "target"),
            "--draft",
            str(tmp_path / "draft"),
            "--output",
            str(tmp_path / "results"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.count("--speculative-num-draft-tokens 12") == 2
    assert "W=6, P=[0, 1, 2, 3, 4, 5, 6]" in result.stdout
    assert "W=7, P=[0, 1, 2, 3, 4, 5]" in result.stdout
    assert result.stdout.count("--context-length 139264") == 2
    assert result.stdout.count("prefetch=all (full probe union, no cap)") == 2
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("fail_second", [False, True])
def test_sweep_runs_each_w_once_and_records_failures(
    tmp_path, monkeypatch, fail_second
):
    monkeypatch.setattr(runner, "checkpoint_gamma", lambda _: 5)
    monkeypatch.setattr(runner, "checkpoint_topk", lambda _: 512)
    output = tmp_path / "sweep"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--mode",
            "paired",
            "--verify-widths",
            "6",
            "7",
            "--execute",
            "--output",
            str(output),
        ],
    )
    widths = []

    def collect(args, gamma):
        width = runner.real_candidates(args) + 1
        widths.append(width)
        assert len(runner.mode_plan(args)) == 1
        assert runner.mode_plan(args)[0][0] == "probes"
        assert args.probes == list(range(13 - width))
        if fail_second and width == 7:
            raise RuntimeError("synthetic collection failure")

    monkeypatch.setattr(runner, "run_experiment", collect)
    if fail_second:
        with pytest.raises(RuntimeError, match="synthetic collection failure"):
            runner.main()
    else:
        runner.main()
    assert widths == [6, 7]
    manifest = json.loads((output / "sweep.json").read_text())
    assert [run["status"] for run in manifest["runs"]] == [
        "completed",
        "failed" if fail_second else "completed",
    ]
    assert manifest["total_width"] == 12


@pytest.mark.parametrize("mode", ["exact", "resident"])
@pytest.mark.parametrize("empty_denominator", [False, True])
def test_p0_measures_no_new_prefetch_without_inventing_hits(mode, empty_denominator):
    source, following, probe = records()
    probe.update(
        p=0,
        mode=mode,
        input_ids=source["input_ids"],
        topk=[],
        probe_demand_h2d_bytes=0,
        probe_workspace_d2d_bytes=0,
    )
    if empty_denominator:
        following.update(
            topk=[[], []], miss_events=[{"row": 0, "ids": []}, {"row": 1, "ids": []}]
        )
    report = analyzer.analyze([source, following, probe])
    (row,) = report["samples"]
    assert row["p"] == 0
    assert (
        row["prediction_units"]
        == row["prefetch_candidate_units"]
        == row["potential_miss_units"]
        == 0
    )
    assert row["history_coverage"] == (None if empty_denominator else 0)
    assert row["potential_unique_miss_coverage"] == (None if empty_denominator else 0)
    assert row["candidate_precision"] is None
    assert row["verify_width"] == source["verify_candidates"] + 1


def test_p0_rejects_fake_probe_io():
    _, _, probe = records()
    probe.update(
        p=0, mode="exact", input_ids=[20, 21], topk=[], probe_demand_h2d_bytes=584
    )
    with pytest.raises(ValueError, match="P=0 must have no probe IO"):
        analyzer.validate(probe)
