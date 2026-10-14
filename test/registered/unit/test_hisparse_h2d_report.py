import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.benchmark.hisparse_h2d import summarize
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

path = Path(__file__).resolve().parents[3] / "scripts/pd_test/summarize_h2d_report.py"
spec = importlib.util.spec_from_file_location("summarize_h2d_report", path)
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


def analysis(device=0, slowdown=1, batches=(32, 32), widths=None):
    rows, events = [], []
    widths = widths or [6] * len(batches)
    for batch, width in zip(batches, widths):
        for layer, misses, us in [(0, 0, 10), (1, 1, 20), (2, 320, 60)]:
            index = len(rows)
            start = index * 1000
            rows.append(
                dict(
                    index=index,
                    layer=layer,
                    verify_width=width,
                    real_batch=batch,
                    captured_batch=32,
                    item_bytes=584,
                    copy_blocks=4,
                    copy_threads=1024,
                    skip_io=False,
                    gpu_stamp_ns=start + 1,
                    misses_per_request=[misses] * batch,
                    unique_misses=misses * batch,
                    h2d_bytes=misses * batch * 584,
                )
            )
            for name, ts, dur in [
                ("copy_cache_planned_kernel", start, us * slowdown),
                ("hisparse_h2d_record_kernel", start + us * slowdown, 2),
            ]:
                events.append(
                    dict(
                        ph="X",
                        cat="kernel",
                        name=name,
                        ts=ts,
                        dur=dur,
                        args=dict(device=device, stream=7),
                    )
                )
    return summarize(
        dict(
            schema_version=1,
            pid=123 + device,
            gpu="test",
            total_records=len(rows),
            overflow=False,
            samples=rows,
        ),
        events,
    )


def test_local_report_keeps_zero_byte_time_and_does_not_sum_rank_costs(tmp_path):
    for device, slowdown in [(0, 1), (1, 1.5)]:
        (tmp_path / f"TP{device}.json").write_text(
            json.dumps(analysis(device, slowdown))
        )
    text = "\n".join(compact.analysis_text(tmp_path, 21))
    assert "3/2 0.090" in text
    assert "3/2 0.135" in text
    assert "zero 0.00,11.1 0.00,11.1" in text
    assert "n=2 min/median/max=1.500/1.500/1.500" in text
    assert "selected=[0, 1, 2]" in text


def test_cycle_cost_is_not_guessed_from_partial_or_interleaved_layers():
    rows = analysis()["samples"]
    assert compact.cycles(rows)["iterations"] == 2
    assert compact.cycles(rows[:-1]) is None
    altered = copy.deepcopy(rows)
    altered[4]["layer"] = 2
    assert compact.cycles(altered) is None
    altered = copy.deepcopy(rows)
    altered[3]["index"] += 1
    assert compact.cycles(altered) is None


def test_geometry_subset_recomputes_totals_and_keeps_zero_payload_and_source():
    report = analysis(batches=(1, 14, 32, 32, 32), widths=(6, 6, 6, 6, 5))
    original = copy.deepcopy(report)
    selected = compact.select_analysis(report, batch_size=32, verify_width=6)
    assert selected["samples"] == report["samples"][6:12]
    assert compact.validate_analysis(selected)
    summary = selected["summary"]
    assert summary["samples"] == 6
    assert summary["h2d_bytes"] == 2 * 321 * 32 * 584
    assert summary["copy_gpu_us"] == summary["copy_gpu_union_us"] == 180
    assert summary["copy_gpu_overlap_us"] == 0
    assert summary["recorder_gpu_us"] == 12
    assert summary["recorder_over_copy_pct"] == pytest.approx(100 * 12 / 180)
    assert compact.cycles(selected["samples"])["iterations"] == 2
    assert report == original
    assert "retained=6/15" in selected["selection"]
    assert len(compact.select_analysis(report, batch_size=32)["samples"]) == 9
    assert len(compact.select_analysis(report, verify_width=6)["samples"]) == 12


def test_geometry_subset_does_not_stitch_disjoint_cycles():
    report = analysis(batches=(32, 1, 32))
    selected = compact.select_analysis(report, batch_size=32)
    assert [s["index"] for s in selected["samples"]] == [0, 1, 2, 6, 7, 8]
    assert compact.cycles(selected["samples"]) is None


def test_geometry_filter_rejects_unassigned_empty_and_corrupt_inputs(tmp_path):
    report = analysis()
    report["pairing"]["mode"] = "capture_totals_only"
    with pytest.raises(ValueError, match="serialized recorder"):
        compact.select_analysis(report, batch_size=32)
    with pytest.raises(ValueError, match="No samples"):
        compact.select_analysis(analysis(), batch_size=1)
    report = analysis(batches=(1, 32, 32))
    # A filter must not conceal a corrupt excluded sample from validation.
    report["samples"][1]["h2d_bytes"] += 1
    (tmp_path / "TP0.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="byte sum"):
        compact.analysis_text(tmp_path, 21, batch_size=32)


def test_geometry_filter_cli_needs_no_torch_or_raw_trace(tmp_path):
    source = tmp_path / "TP0.json"
    original = json.dumps(analysis(batches=(1, 14, 32, 32, 32), widths=(6, 6, 6, 6, 5)))
    source.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(path),
            "--analysis-dir",
            str(tmp_path),
            "--batch-size",
            "32",
            "--verify-width",
            "6",
            "--max-layers",
            "21",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "retained=6/15" in result.stdout
    assert "3/2 0.090" in result.stdout
    assert "32/32/6 n=6" in result.stdout
    assert "zero 0.00,11.1" in result.stdout
    assert "d0:GB/s,time%,mean_MB,n" in result.stdout
    assert "0 0.00,11.1,0.000,2" in result.stdout
    assert source.read_text() == original


def test_aggregate_only_captures_do_not_invent_layer_timing(tmp_path):
    report = analysis()
    report["pairing"]["mode"] = "capture_totals_only"
    for row in report["samples"]:
        del row["copy_us"]
    (tmp_path / "TP0.json").write_text(json.dumps(report))
    text = "\n".join(compact.analysis_text(tmp_path, 8))
    assert "unverified NA" in text
    assert "Per-layer/payload-bin comparison omitted" in text
    assert "Payload bins (decimal MB)" not in text


def test_mismatched_summaries_and_mixed_captures_are_rejected(tmp_path):
    report = analysis()
    report["summary"]["h2d_bytes"] += 584
    with pytest.raises(ValueError, match="byte sum"):
        compact.validate_analysis(report)
    report = analysis()
    report["summary"]["copy_gpu_us"] += 1
    with pytest.raises(ValueError, match="time sum"):
        compact.validate_analysis(report)
    for name in ("capture1.json", "capture2.json"):
        (tmp_path / name).write_text(json.dumps(analysis()))
    with pytest.raises(ValueError, match="one capture"):
        compact.analysis_text(tmp_path, 8)


def test_acceptance_extracts_only_five_b32_features_without_prompt_text(tmp_path):
    path = tmp_path / "decode.log"
    lines = [
        f"[2026-10-13 12:00:0{i}] Decode batch, #running-req: 32, accept len: {i}.00, accept rate: .5, private prompt text\n"
        for i in range(7)
    ]
    lines.append("#running-req: 320, accept len: 99.0\n")
    path.write_text("".join(lines))
    result = compact.acceptance_text(path)
    assert len(result) == 6
    assert "accept len: 2.00" in result[1]
    assert "accept len: 6.00" in result[-1]
    assert not any("prompt" in line or "99.0" in line for line in result)


def test_copy_report_uses_raw_total_time_and_retains_allocation_scope(tmp_path):
    (tmp_path / "rank0.json").write_text(
        json.dumps(
            dict(
                success=True,
                rank=0,
                world_size=1,
                gpu_uuid="test-uuid",
                cuda_visible_devices="4",
                cuda_device=0,
                config=dict(
                    source_mib=512,
                    seed=0,
                    rounds=5,
                    blocks=[4],
                    plan_stride=3072,
                    buffer_slots=3072,
                ),
                host_allocation=dict(
                    buffers=[
                        dict(
                            regions=[
                                dict(
                                    numa=dict(
                                        policy="bind:1", pages_by_node={"1": 100}
                                    ),
                                    buffer_covers_whole_vma=False,
                                )
                            ]
                        )
                    ]
                ),
                cases=[
                    dict(
                        label="B32-miss100",
                        payload_bytes=100000,
                        results=[
                            dict(
                                method="sm-cta4",
                                bytes_per_copy=100000,
                                raw_copy_us=[10, 90],
                                correctness=dict(all_payload_bytes_checked=True),
                            )
                        ],
                    )
                ],
            )
        )
    )
    text = "\n".join(compact.copy_text(tmp_path))
    assert "- 2.00/-/-/-/-" in text
    assert "('bind:1', {'1': 100}, False)" in text


def test_numa_report_preserves_mixed_nodes_and_partial_vma_scope(tmp_path):
    path = tmp_path / "numa.json"
    path.write_text(
        json.dumps(
            dict(
                commands=[],
                processes=[
                    dict(
                        pid=123,
                        status={},
                        pid_stable_during_snapshot=True,
                        probe_identity_status="verified_start_ticks",
                        process_pages_by_node={"1": 999},
                        errors=[],
                        buffer_snapshot=dict(
                            buffers=[
                                dict(
                                    name="layer17",
                                    mapped_bytes=4096,
                                    nbytes=4096,
                                    regions=[
                                        dict(
                                            numa=dict(
                                                policy="prefer:1",
                                                pages_by_node={"0": 3, "1": 7},
                                                kernelpagesize_kB=4,
                                            ),
                                            buffer_covers_whole_vma=False,
                                        )
                                    ],
                                )
                            ]
                        ),
                    )
                ],
            )
        )
    )
    text = "\n".join(compact.numa_text(path))
    assert '"whole_vma": false' in text
    assert '"nodes": {"0": 3, "1": 7}' in text
    assert "KV=layer17" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
