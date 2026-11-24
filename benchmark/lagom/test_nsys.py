"""CPU fixtures for Nsight exports with overlapping streams and TP workers."""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import analyze_nsys as nsys
import pytest


def make_capture(path, ranks=4, steps=2):
    with closing(sqlite3.connect(path)) as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER, deviceId INTEGER, demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER, copyKind INTEGER, bytes INTEGER);
        """)
        symbols = ["stamp", *nsys.FUSED, "nvjet_sm90", "other::copy_kernel<true>"]
        db.executemany("INSERT INTO StringIds VALUES (?,?)", enumerate(symbols))
        for step in range(steps):
            t = step * 100
            for i, name in enumerate(("step", "step_begin", "step_end")):
                db.execute(
                    "INSERT INTO StringIds VALUES (?,?)",
                    (10 + 3 * step + i, f"lagom/{name}/step={step}"),
                )
            for rank in range(ranks):
                pid, dev = (30 + rank) << 24, 4 + rank
                tid = pid + 3
                # Correlation IDs are reused by different processes.
                for i, (a, b) in enumerate(((0, 95), (1, 5), (85, 89))):
                    db.execute(
                        "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)",
                        (t + a, t + b, tid, None, 10 + 3 * step + i),
                    )
                db.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
                    [
                        (t + 2, t + 4, tid, t + 1),
                        (t + 86, t + 88, tid, t + 2),
                        (t + 58, t + 67, tid, t + 3),
                    ],
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(?,?,?,?)",
                    (t + 60, t + 65, tid, t + 4),
                )
                for a, b, cid, name in (
                    (9, 10, 1, 0),
                    (90, 91, 2, 0),
                    (15, 25, 5, 1),
                    (25, 35, 6, 2),
                    (35, 45, 7, 3),
                    (30, 50 + rank, 8, 4),
                ):
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
                        (t + a, t + b, pid, t + cid, dev, name),
                    )
                db.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?)",
                    [
                        (t + 40, t + 55, pid, dev, 1, 584),
                        (t + 52, t + 58, pid, dev, 8, 80),
                    ],
                )
                if step == 0:
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(84,111,?,9,?,5)",
                        (pid, dev),
                    )
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(87,114,?,?,1,999)",
                        (pid, dev),
                    )
        db.commit()


def test_step_partition_and_kernel_name_attribution_per_rank(tmp_path):
    path = tmp_path / "profile.sqlite"
    make_capture(path)
    rank_map = {30 + rank: rank for rank in range(4)}
    report = nsys.step_report(
        path, ranks_by_pid=rank_map, steps=2, required_families=nsys.FUSED.values()
    )
    assert len(report["decode_steps"]) == 8
    for row in report["decode_steps"]:
        assert row["device"] == 4 + row["rank"]
        assert sum(
            row[k]
            for k in (
                "kernel_only_ms",
                "explicit_h2d_only_ms",
                "other_gpu_only_ms",
                "gpu_overlap_ms",
                "cpu_api_during_gpu_idle_ms",
                "idle_without_observed_api_ms",
            )
        ) == pytest.approx(row["gpu_step_ms"])
        assert row["cpu_api_during_gpu_idle_ms"] == pytest.approx(
            (9 if row["step"] == 0 else 11) / 1e6
        )
        assert row["cpu_cuda_api_union_ms"] == pytest.approx(13 / 1e6)
        assert (
            row["explicit_h2d_bytes"],
            row["explicit_h2d_copies"],
            row["h2d_copies_crossing_boundary"],
        ) == (584, 1, 1)
    for row in report["kernel_attribution"]:
        families = {f["family"]: f for f in row["families"]}
        assert families["other"]["instances"] == 1
        assert families["other"]["kernel_ms"] == pytest.approx(7 / 1e6)
        assert families["gemm"]["kernel_ms"] == pytest.approx(
            2 * (20 + row["rank"]) / 1e6
        )
        assert sum(
            f["pct_of_summed_kernel_time"] for f in families.values()
        ) == pytest.approx(100)
        assert {k["name"] for k in row["kernels"]} >= nsys.FUSED.keys()
    with pytest.raises(ValueError, match="devices/processes"):
        nsys.step_report(path, ranks_by_pid={30: 0, 31: 1, 32: 2})
    with pytest.raises(ValueError, match="Incomplete Nsight step"):
        nsys.step_report(path, steps=3)
    with pytest.raises(ValueError, match="Missing kernel families"):
        nsys.step_report(path, required_families=["promote"])


def test_entire_legacy_capture_resolves_names_without_step_markers(tmp_path):
    path = tmp_path / "old.sqlite"
    make_capture(path, ranks=2, steps=1)
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TABLE NVTX_EVENTS")
        names = {
            1: "void sglang::staging::copy_kernel<(bool)1>(foo)",
            2: "void sglang::staging::copy_kernel<false>(foo)",
            3: "sglang::staging::promote_kernel<2048>(foo)",
        }
        db.executemany(
            "UPDATE StringIds SET value=? WHERE id=?",
            [(v, k) for k, v in names.items()],
        )
        db.commit()
    for row in nsys.capture_kernels(path):
        families = {f["family"]: f for f in row["families"]}
        assert {"promote", "refill", "stage_copy", "gemm", "other"} == families.keys()
        assert families["refill"]["instances"] == 1
        assert families["gemm"]["kernel_ms"] == pytest.approx(
            (row["device"] - 4 + 20) / 1e6
        )


def test_missing_and_ambiguous_boundaries_fail(tmp_path):
    path = tmp_path / "broken.sqlite"
    make_capture(path, ranks=1, steps=1)
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1"
        )
        db.commit()
    with pytest.raises(ValueError, match="Ambiguous GPU stamp"):
        nsys.step_report(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1")
        db.commit()
    with pytest.raises(ValueError, match="Missing correlated GPU stamps"):
        nsys.step_report(path)


def test_export_uses_fresh_writable_copy_and_checks_data_even_on_exit_zero(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "archive", tmp_path / "results"
    source.mkdir()
    output.mkdir()
    report = source / "prof.nsys-rep"
    database, log = output / "prof.sqlite", output / "stats.txt"
    stale = report.with_suffix(".sqlite")
    stale.write_bytes(b"stale")
    for exists in (False, True):
        if exists:
            report.touch()
        with pytest.raises(ValueError, match="Missing or empty"):
            nsys.export_report(report, database, log)
    report.write_bytes(b"rep payload")
    fixture = tmp_path / "fixture.sqlite"
    make_capture(fixture)

    def fake_stats(argv, **kwargs):
        copied = Path(argv[-1])
        assert argv[:3] == ["nsys", "stats", "--force-export=true"]
        assert (
            copied.parent.parent == output
            and copied.read_bytes() == report.read_bytes()
        )
        fresh = copied.with_suffix(".sqlite")
        assert not fresh.exists()
        shutil.copyfile(fixture, fresh)

    monkeypatch.setattr(nsys.subprocess, "run", fake_stats)
    nsys.export_report(report, database, log)
    valid = database.read_bytes()
    assert valid == fixture.read_bytes() and stale.read_bytes() == b"stale"
    with closing(sqlite3.connect(fixture)) as db:
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        db.commit()
    with pytest.raises(ValueError, match="no CUDA kernel activities"):
        nsys.export_report(report, database, log)
    monkeypatch.setattr(nsys.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="no SQLite export"):
        nsys.export_report(report, database, log)
    assert database.read_bytes() == valid


def test_adaptive_kernel_names_from_cuda_13_offline_h100_build():
    assert nsys.kernel_family("sglang::dhisparse_plan_kernel(...)") == "swap_in_gather"
    assert nsys.kernel_family("sglang::dhisparse_admit_kernel(...)") == "swap_in_commit"
    for number, family in enumerate(
        ("swap_in_gather", "swap_in_reread", "swap_in_commit")
    ):
        name = f"sglang::dhisparse_copy_kernel<584, true, (sglang::DHiSparseCopy){number}>(...)"
        assert nsys.kernel_family(name) == family
