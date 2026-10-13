import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.benchmark.hisparse_h2d_diagnostics import (
    ITEM_BYTES,
    PAGE_BYTES,
    copy_geometry,
    describe_buffer,
    parse_maps,
    parse_numa_maps,
    process_snapshot,
    timing_summary,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_numa_counts_for_a_shared_vma_are_not_attributed_to_a_buffer():
    maps = parse_maps("1000-9000 rw-s 00000000 00:01 42 /dev/zero (deleted)\n")
    numa = parse_numa_maps(
        "1000 bind:1 dirty=8 mapped=8 N0=2 N1=6 kernelpagesize_kB=4\n"
    )
    row = describe_buffer(dict(name="kv", address=0x2000, nbytes=0x2000), maps, numa)
    assert row["mapped_bytes"] == 0x2000
    region = row["regions"][0]
    assert region["numa"]["pages_by_node"] == {"0": 2, "1": 6}
    assert region["counts_scope"] == "whole_vma"
    assert not region["buffer_covers_whole_vma"]
    exact = describe_buffer(dict(name="kv", address=0x1000, nbytes=0x8000), maps, numa)
    assert exact["regions"][0]["buffer_covers_whole_vma"]


def test_numa_missing_vmas_and_hugetlb_page_units_are_visible():
    maps = parse_maps("1000-2000 rw-s 0 0:0 0\n2000-4000 rw-s 0 0:0 0\n")
    numa = parse_numa_maps("2000 default huge dirty=1 N1=1 kernelpagesize_kB=2048\n")
    row = describe_buffer(dict(name="kv", address=0x1000, nbytes=0x4000), maps, numa)
    assert row["mapped_bytes"] == 0x3000
    assert row["regions"][0]["numa"] is None
    assert row["regions"][1]["numa"]["kernelpagesize_kB"] == 2048


def test_process_snapshot_retains_missing_proc_errors(tmp_path):
    row = process_snapshot(123, proc_root=tmp_path)
    assert not row["pid_stable_during_snapshot"]
    assert row["errors"]
    assert row["status"] == {}
    assert row["process_pages_by_node"] == {}


def test_process_identity_and_task_affinity_are_read_without_cmdline(tmp_path):
    root = tmp_path / "123"
    (root / "task" / "123").mkdir(parents=True)
    (root / "stat").write_text(
        "123 (worker (TP 0)) S " + " ".join(["0"] * 18 + ["777"] + ["0"] * 5)
    )
    status = (
        "Name:\tworker\nPid:\t123\nCpus_allowed_list:\t2-4\nMems_allowed_list:\t0-1\n"
    )
    (root / "status").write_text(status)
    (root / "task" / "123" / "status").write_text(status)
    (root / "maps").write_text("1000-2000 rw-s 0 0:0 0\n")
    (root / "numa_maps").write_text("1000 prefer:1 N1=1 kernelpagesize_kB=4\n")
    row = process_snapshot(123, proc_root=tmp_path)
    assert row["start_ticks"] == 777
    assert row["pid_stable_during_snapshot"]
    assert row["threads_by_cpu_affinity"] == {"2-4": 1}
    assert not row["errors"]


@pytest.mark.parametrize("misses", [100, 1272])
def test_rotating_plans_fit_host_source_and_preserve_effective_bytes(misses):
    row = copy_geometry(32, misses, 512 * 2**20, 64)
    assert row["banks"] * row["entries"] <= row["host_tokens"]
    assert row["payload_bytes"] == 32 * misses * 584
    assert row["rotating_payload_bytes"] > 100 * 2**20
    assert row["destination_pages"] * 64 >= row["entries"]


def test_timing_bandwidth_is_total_bytes_divided_by_total_time():
    row = timing_summary([10, 90], 100000)
    assert row["GBps"] == 2
    assert row["p95_us"] == 90
    with pytest.raises(ValueError, match="positive"):
        timing_summary([0], 100000)
    with pytest.raises(ValueError, match="two disjoint"):
        copy_geometry(32, 1272, PAGE_BYTES, 64)


@pytest.mark.parametrize("corrupt", [None, "tail_value", "scale"])
@pytest.mark.parametrize("scatter", [False, True])
def test_cpu_copy_reference_covers_page_boundaries_values_and_scales(corrupt, scatter):
    path = Path(__file__).resolve().parents[3] / "scripts/pd_test/bench_h2d_copy.py"
    spec = importlib.util.spec_from_file_location("bench_h2d_copy", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    host = torch.randint(0, 256, (4, PAGE_BYTES), dtype=torch.uint8)
    device = torch.zeros((3, PAGE_BYTES), dtype=torch.uint8)
    source = torch.arange(65, dtype=torch.int64) * 3
    destination = torch.arange(65, dtype=torch.int64) * (2 if scatter else 1)
    for dest, src in zip(destination.tolist(), source.tolist()):
        sp, so, dp, do = src // 64, src % 64, dest // 64, dest % 64
        device[dp, do * 576 : (do + 1) * 576] = host[sp, so * 576 : (so + 1) * 576]
        device[dp, 64 * 576 + do * 8 : 64 * 576 + (do + 1) * 8] = host[
            sp, 64 * 576 + so * 8 : 64 * 576 + (so + 1) * 8
        ]
    if corrupt:
        dp, do = int(destination[-1]) // 64, int(destination[-1]) % 64
        col = do * 576 + 575 if corrupt == "tail_value" else 64 * 576 + do * 8 + 7
        device[dp, col] ^= 1
        with pytest.raises(RuntimeError, match="mismatch"):
            benchmark.validate_sm(torch, host, device, source, destination)
    else:
        benchmark.validate_sm(torch, host, device, source, destination)
    assert ITEM_BYTES == 584
