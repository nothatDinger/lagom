import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def cli():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/pd_test/summarize_h2d_ncu_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("ncu_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_csv(cli, *, corrupt=None):
    rows = [
        [
            "ID",
            "Process ID",
            "Kernel Name",
            "Device",
            "Metric Name",
            "Metric Unit",
            "Metric Value",
        ]
    ]
    for metric, value, unit in zip(
        cli.REQUIRED,
        ["10,000", "1,500", "4", "1024"],
        ["nsecond", "cycle", "block", "thread"],
    ):
        rows.append(
            [
                "0",
                "123",
                "void sglang::copy_cache_planned_kernel<1024, true, true, false, 0>()",
                "0",
                metric,
                unit,
                value,
            ]
        )
    if corrupt == "missing":
        rows.pop()
    elif corrupt == "unavailable":
        rows[2][-1] = "n/a"
    elif corrupt == "nan":
        rows[2][-1] = "nan"
    elif corrupt == "second_launch":
        rows[-1][0] = "1"
    elif corrupt == "wrong_kernel":
        rows[1][2] = "unrelated_kernel"
    elif corrupt == "wrong_grid":
        rows[3][-1] = "8"
    result = io.StringIO()
    result.write("==PROF== Disconnected from process 123\n")
    csv.writer(result).writerows(rows)
    return result.getvalue()


def pair_report():
    config = dict(
        batch=2,
        misses=[2],
        devices=[0, 1],
        bulk_ce_mib=1,
        device_source_control=True,
        profile=False,
    )
    rows = [
        dict(
            case=case,
            method=method,
            device=device,
            all_payload_bytes_checked=True,
            graph_replay_checked=True,
        )
        for case, method in [
            ("B2-miss2", "sm"),
            ("B2-miss2", "sm_hbm"),
            ("B2-miss2", "ce"),
            ("bulk-ce-1MiB", "ce"),
        ]
        for device in [0, 1]
    ]
    return dict(
        success=True,
        host_register_flags=1,
        config=config,
        correctness=rows,
        devices={"0": {"physical_index": 4}, "1": {"physical_index": 5}},
    )


def test_ncu_csv_keeps_units_and_accepts_quoted_thousands_and_kernel_commas(cli):
    target = cli.read_target(raw_csv(cli))
    assert target["metrics"]["gpu__time_duration.sum"] == dict(
        value=10000, unit="nsecond"
    )
    assert target["metrics"]["launch__block_size"]["value"] == 1024


@pytest.mark.parametrize(
    "corrupt",
    ["missing", "unavailable", "nan", "second_launch", "wrong_kernel", "wrong_grid"],
)
def test_incomplete_or_wrong_capture_cannot_pass(cli, corrupt):
    with pytest.raises(ValueError):
        cli.read_target(raw_csv(cli, corrupt=corrupt))


def test_missing_copy_and_torch_profiler_fail_pair_gate(cli):
    report = pair_report()
    assert cli.pair_gate(report)["expected"] == 8
    report["correctness"].pop()
    with pytest.raises(ValueError):
        cli.pair_gate(report)
    report = pair_report()
    report["config"]["profile"] = True
    with pytest.raises(ValueError):
        cli.pair_gate(report)


def test_metric_availability_does_not_infer_suffixes(cli):
    name = cli.CANDIDATES[0]
    assert cli.metric_availability(name)["available"] == [name]
    assert not cli.metric_availability(name.removesuffix(".sum"))["available"]


def test_offline_cli_reports_compact_gate(tmp_path, monkeypatch, capsys, cli):
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps(pair_report()))
    raw = tmp_path / "raw.csv"
    raw.write_text(raw_csv(cli))
    monkeypatch.setattr(
        sys, "argv", ["summary", "--pair-json", str(pair), "--ncu-csv", str(raw)]
    )
    assert cli.main() == 0
    assert "compatibility_ok=True" in capsys.readouterr().out
    isolated = subprocess.run(
        [sys.executable, "-I", "-S", cli.__file__, *sys.argv[1:]],
        text=True,
        capture_output=True,
    )
    assert isolated.returncode == 0, isolated.stderr + isolated.stdout
    assert "compatibility_ok=True" in isolated.stdout
    raw.write_text(raw_csv(cli, corrupt="unavailable"))
    assert cli.main() == 1
    result = capsys.readouterr().out
    assert "compatibility_ok=False" in result and "compatibility_ok=True" not in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
