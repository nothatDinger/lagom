"""CPU-only protocol tests. Run: python -m unittest discover -s benchmark/lagom -p test_overall.py"""
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import longbench_v2_fixed as data
import run_overall as overall


class OverallProtocolTest(unittest.TestCase):
    def setUp(self):
        self.cfg = overall.load_config(overall.HERE / "overall.example.json")
        self.cell = overall.matrix(["lagom"], [32768], [8])[0]

    def test_matrix_order_counts_and_concurrency_is_only_a_cap(self):
        cells = overall.matrix(overall.METHODS, overall.INPUT_LENS, overall.CONCURRENCIES)
        self.assertEqual(len(cells), 72)
        self.assertEqual(
            [cells[i]["method"] for i in (0, 24, 48)], list(overall.METHODS)
        )
        self.assertTrue(all(c["num_requests"] == 4 * c["max_concurrency"] for c in cells))
        self.assertEqual(cells[-1]["num_requests"], 1024)

    def test_methods_share_buffers_and_pd_topology(self):
        configs = []
        for method in overall.METHODS:
            cell = self.cell | {"method": method}
            cmds, _ = overall.commands(self.cfg, cell, Path("/data"), Path("/results"))
            p, d, bench = (cmds[k] for k in ("prefill", "decode", "benchmark"))
            self.assertNotIn("--enable-hisparse", p)
            state_flag = "--disaggregation-dsv4-state-execution-width"
            self.assertNotIn(state_flag, d)
            if method == "lagom":
                self.assertEqual(
                    int(p[p.index(state_flag) + 1]),
                    self.cfg["w"] + self.cfg["p"] + 1,
                )
            else:
                self.assertNotIn(state_flag, p)
            self.assertIn("--enable-hisparse", d)
            self.assertEqual(p[p.index("--base-gpu-id") + 1], "0")
            self.assertEqual(d[d.index("--base-gpu-id") + 1], "4")
            for cmd in (p, d):
                self.assertEqual("--speculative-algorithm" in cmd, method != "hisparse")
                self.assertEqual(cmd[cmd.index("--tp-size") + 1], "4")
                self.assertIn("--disable-radix-cache", cmd)
                if method != "hisparse":
                    self.assertEqual(cmd[cmd.index("--speculative-num-draft-tokens") + 1], "8")
            hs = json.loads(d[d.index("--hisparse-config") + 1])
            configs.append(hs)
            if method == "lagom":
                self.assertEqual(hs["prefetcher_config"]["prediction_steps"], self.cfg["p"])
                self.assertEqual(hs["prefetcher_config"]["buffer_slots"], self.cfg["prefetch_slots"])
            else:
                self.assertNotIn("prefetcher", hs)
            self.assertEqual(bench[bench.index("--max-concurrency") + 1], "8")
            self.assertEqual(bench[bench.index("--num-prompts") + 1], "32")
            self.assertNotIn("--disable-ignore-eos", bench)
            self.assertNotIn("--disable-stream", bench)
            self.assertEqual(
                json.loads(d[d.index("--cuda-graph-config") + 1]),
                {"decode": {"backend": "full", "bs": [1, 2, 4, 8]}},
            )
        for key in ("device_buffer_size", "host_to_device_ratio", "force_h2d_each_step"):
            self.assertEqual(len({config[key] for config in configs}), 1)
        self.assertFalse(configs[0]["force_h2d_each_step"])

    def test_max_batch_graphs_and_no_protocol_override(self):
        cell = self.cell | {"max_concurrency": 256, "num_requests": 1024}
        cmds, _ = overall.commands(self.cfg, cell, Path("/data"), Path("/out"))
        d = cmds["decode"]
        self.assertEqual(
            json.loads(d[d.index("--cuda-graph-config") + 1])["decode"]["bs"],
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
        for extra in (["--disable-cuda-graph"], ["--hisparse-config={}"], ["--max-running-requests", "1"]):
            cfg = self.cfg | {"decode_extra_args": extra}
            with self.assertRaises(ValueError):
                overall.commands(cfg, self.cell, Path("/data"), Path("/out"))

    def test_short_inputs_are_rejected_and_question_is_preserved(self):
        context, suffix = list(range(100)), [1001, 1002, 1003, 1004]
        wrap = lambda ids: [-1] + ids + [-2]
        prompt = data.fixed_prompt(context, suffix, 16, wrap)
        self.assertEqual(len(prompt), 16)
        self.assertEqual(prompt, [-1] + list(range(5)) + list(range(95, 100)) + suffix + [-2])
        self.assertIsNone(data.fixed_prompt([1], suffix, 16, wrap))
        self.assertIsNone(data.fixed_prompt(context, suffix, 6, wrap))
        row = {"question": "Q?", **{"choice_" + c: c.lower() for c in "ABCD"}}
        self.assertIn("D. d\nAnswer:", data.question_suffix(row))

    def test_request_order_repeats_deterministically_without_padding(self):
        order = data.request_order(3, 32, 42)
        self.assertEqual(order, data.request_order(3, 32, 42))
        self.assertEqual(len(order), 32)
        self.assertEqual(set(order), {0, 1, 2})
        self.assertEqual(set(order[:3]), {0, 1, 2})
        with self.assertRaises(ValueError):
            data.request_order(0, 32, 42)

    def test_token_bank_checksum_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.u32"
            checksum = data.write_ids(path, [1, 100000, 5])
            self.assertEqual(data.read_ids(path, checksum, 3), [1, 100000, 5])
            path.write_bytes(b"\x00" * 12)
            with self.assertRaises(ValueError):
                data.read_ids(path, checksum, 3)

    def valid_result(self):
        n = self.cell["num_requests"]
        return {
            "completed": n, "max_concurrency": 8, "duration": 64,
            "input_lens": [32768] * n, "output_lens": [1024] * n,
            "total_input_tokens": n * 32768, "total_output_tokens": n * 1024,
            "errors": [""] * n, "output_throughput": n * 1024 / 64,
            "mean_tpot_ms": 10, "median_tpot_ms": 9, "p99_tpot_ms": 20,
        }

    def test_only_complete_fixed_length_results_are_valid(self):
        valid = self.valid_result()
        self.assertEqual(overall.validate_result(valid, self.cell)["output_tps"], 512)
        for key, value in (
            ("completed", 31), ("duration", 0), ("mean_tpot_ms", float("nan")),
            ("output_lens", [1023] * 32), ("input_lens", [32000] * 32),
            ("output_throughput", 123), ("max_concurrency", 16),
            ("errors", ["failure"] * 32),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                overall.validate_result(valid | {key: value}, self.cell)

    def server_info(self, method):
        info = {
            "disaggregation_mode": "decode", "disaggregation_transfer_backend": "mooncake",
            "tp_size": 4, "max_running_requests": 8, "enable_hisparse": True,
            "moe_runner_backend": "marlin", "speculative_moe_runner_backend": "marlin",
            "speculative_algorithm": None if method == "hisparse" else "DSPARK",
            "speculative_num_draft_tokens": 8, "disable_cuda_graph": False,
            "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 4, 8]}},
            "hisparse_config": {
                "device_buffer_size": 8192, "host_to_device_ratio": 2,
                "force_h2d_each_step": False,
            },
        }
        if method == "dhisparse":
            info["hisparse_config"].update(
                adaptive=True, resident_first=True, refill_pages_per_step=16
            )
        if method == "lagom":
            info["hisparse_config"].update(
                prefetcher="lagom", prefetcher_config={
                    "prediction_steps": self.cfg["p"], "buffer_slots": self.cfg["prefetch_slots"],
                }
            )
        return info

    def test_resolved_config_rejects_silent_graph_and_prefetch_fallbacks(self):
        for method in overall.METHODS:
            cell = self.cell | {"method": method}
            info = self.server_info(method)
            overall.validate_server_info(info, self.cfg, cell, "decode")
            bad = copy.deepcopy(info)
            bad["disable_cuda_graph"] = True
            with self.assertRaises(ValueError):
                overall.validate_server_info(bad, self.cfg, cell, "decode")
        info = self.server_info("lagom")
        info["hisparse_config"].pop("prefetcher")
        with self.assertRaises(ValueError):
            overall.validate_server_info(info, self.cfg, self.cell, "decode")

    def test_startup_timeout_records_no_tps_and_always_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "attempt-0001"
            with patch.object(overall, "spawn"), patch.object(overall, "assert_free_ports"), \
                 patch.object(overall, "runtime_env", return_value={k: "" for k in (
                     *self.cfg["env"], "SGLANG_RAGGED_VERIFY_MODE",
                     "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT",
                     "SGLANG_DISAGGREGATION_WAITING_TIMEOUT",
                 )}), patch.object(overall, "cleanup") as cleanup, \
                 patch.object(overall, "wait_ready", side_effect=TimeoutError("budget")):
                result = overall.run_cell(self.cfg, self.cell, Path("/data"), folder)
            self.assertEqual(result["status"], "timeout")
            self.assertNotIn("output_tps", result)
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "timeout")
            cleanup.assert_called_once()

    def test_prediction_window_is_configurable_and_shared_with_prefill_layout(self):
        self.assertEqual(self.cfg["p"], 4)
        flag = "--disaggregation-dsv4-state-execution-width"
        for prediction_steps in (1, 2, 4, 7, 8):
            cfg = self.cfg | {"p": prediction_steps}
            cmds, _ = overall.commands(cfg, self.cell, Path("/data"), Path("/out"))
            p, d = cmds["prefill"], cmds["decode"]
            self.assertEqual(int(p[p.index(flag) + 1]), cfg["w"] + prediction_steps + 1)
            self.assertEqual(p[p.index("--speculative-num-draft-tokens") + 1], "8")
            self.assertEqual(d[d.index("--speculative-num-draft-tokens") + 1], "8")
            hs = json.loads(d[d.index("--hisparse-config") + 1])
            self.assertEqual(hs["prefetcher_config"]["prediction_steps"], prediction_steps)
            self.assertNotIn("--hisparse-config", p)
            self.assertNotIn(flag, d)

    def test_prefill_resolved_state_width_must_match_the_selected_prediction_window(self):
        info = self.server_info("lagom") | {
            "disaggregation_mode": "prefill", "enable_hisparse": False,
            "hisparse_config": None, "is_lagom": False,
            "disaggregation_dsv4_state_execution_width": self.cfg["w"] + self.cfg["p"] + 1,
        }
        overall.validate_server_info(info, self.cfg, self.cell, "prefill")
        with self.assertRaisesRegex(ValueError, "state execution width"):
            overall.validate_server_info(
                info | {"disaggregation_dsv4_state_execution_width": None},
                self.cfg, self.cell, "prefill",
            )
        with self.assertRaisesRegex(ValueError, "ordinary DSPARK"):
            overall.validate_server_info(
                info | {"is_lagom": True}, self.cfg, self.cell, "prefill"
            )

    def test_pd_compatibility_preflight_finds_all_layout_consumers(self):
        preflight = overall.source_preflight()
        self.assertTrue(preflight["oneshot_source_check_passed"])
        self.assertFalse(preflight["lagom_pd_explicitly_blocked"])
        self.assertFalse(preflight["lagom_pd_layout_ready"])

    def test_python_paths_preserve_venv_symlinks(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / "python"
            interpreter.symlink_to(sys.executable)
            config = Path(tmp) / "cfg.json"
            for value in (str(interpreter), os.path.relpath(interpreter, overall.ROOT)):
                config.write_text(json.dumps({"python": value}))
                self.assertEqual(
                    Path(os.path.abspath(overall.load_config(config)["python"])),
                    interpreter,
                )

    def test_paired_dry_run_uses_same_workload_and_decode_only_trace(self):
        output = subprocess.check_output(
            [
                sys.executable,
                str(overall.HERE / "run_overall.py"),
                "--h2d-pair",
                "--dry-run",
                "--nsys",
                "--sharegpt",
                "/data/local.json",
                "--output",
                "/out",
            ]
        )
        base, forced = json.loads(output)["cells"]
        for cell, force in ((base, False), (forced, True)):
            self.assertEqual((cell["input_len"], cell["max_concurrency"]), (32768, 256))
            cmds = cell["commands"]
            self.assertEqual(cmds["prefill"][0], sys.executable)
            self.assertEqual(cmds["decode"][0], "nsys")
            self.assertEqual(cmds["router"][0], sys.executable)
            self.assertNotIn("nsys", cmds["prefill"])
            self.assertIn("--capture-range=cudaProfilerApi", cmds["decode"])
            self.assertIn("--capture-range-end=stop", cmds["decode"])
            self.assertIn("--cuda-graph-trace=node", cmds["decode"])
            for flag, value in (
                ("--watchdog-timeout", "100000"),
                ("--dist-timeout", "3600"),
            ):
                self.assertEqual(cmds["decode"][cmds["decode"].index(flag) + 1], value)
                self.assertNotIn(flag, cmds["prefill"])
            self.assertEqual(
                cell["decode_profiler_environment"],
                {
                    "TORCH_NCCL_TIMEOUT": "3600000",
                    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "3600",
                },
            )
            self.assertIn("--sharegpt-input-len", cmds["benchmark"])
            self.assertEqual(cell["decode_trace_config"]["required_batch_size"], 256)
            hs = json.loads(
                cmds["decode"][cmds["decode"].index("--hisparse-config") + 1]
            )
            self.assertEqual(hs["device_buffer_size"], 2048)
            self.assertIs(hs["adaptive"], True)
            self.assertIs(hs["resident_first"], False)
            self.assertEqual(hs["refill_pages_per_step"], 0)
            width = int(
                cmds["decode"][
                    cmds["decode"].index("--speculative-num-draft-tokens") + 1
                ]
            )
            self.assertGreater(width * hs["top_k"], hs["device_buffer_size"])
            trace = cell["decode_trace_config"]
            self.assertEqual(trace["min_output_tokens"], overall.OUTPUT_LEN - 1)
            self.assertEqual(width * (trace["skip"] + trace["steps"]), 84)
            self.assertLess(84, trace["min_output_tokens"])
            self.assertIs(hs["force_h2d_each_step"], force)
        self.assertEqual(base["commands"]["prefill"], forced["commands"]["prefill"])
        self.assertEqual(base["commands"]["router"], forced["commands"]["router"])

    def test_graph_cap_covers_measured_batch_and_is_limited_by_concurrency(self):
        self.assertEqual(
            overall.graph_sizes(self.cell | {"graph_bs_cap": 6}), [1, 2, 4, 6]
        )
        self.assertEqual(
            overall.graph_sizes(self.cell | {"graph_bs_cap": 512}), [1, 2, 4, 8]
        )
        argv = [
            sys.executable,
            str(overall.HERE / "run_overall.py"),
            "--h2d-pair",
            "--dry-run",
            "--output",
            "/out",
            "--sharegpt",
            "/data/sharegpt.json",
            "--graph-bs-cap",
            "128",
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("fit the largest captured graph", proc.stderr)
        cell = json.loads(
            subprocess.check_output(argv + ["--trace-batch-size", "128"])
        )["cells"][0]
        decode = cell["commands"]["decode"]
        self.assertEqual(
            max(
                json.loads(decode[decode.index("--cuda-graph-config") + 1])["decode"][
                    "bs"
                ]
            ),
            128,
        )
        self.assertNotIn("--watchdog-timeout", decode)
        self.assertEqual(cell["decode_profiler_environment"], {})

    def test_profile_pair_publishes_only_kernel_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, data_path = Path(tmp) / "output", Path(tmp) / "sharegpt.json"
            data_path.write_text("[]")

            def profiled_cell(cfg, cell, data, folder):
                folder.mkdir(parents=True)
                row = overall.validate_result(
                    self.valid_result(), self.cell | {"nsys": True}
                )
                self.assertEqual(row, {"completed": 32})
                overall.save(folder / "status.json", cell | row | {"status": "ok"})
                overall.save(
                    folder / "nsys-kernels.json",
                    [
                        {
                            "device": r + 4,
                            "rank": r,
                            "steps": 12,
                            "families": [
                                {
                                    "family": "swap_in_reread",
                                    "kernel_ms": (
                                        3 if cell["force_h2d_each_step"] else 1
                                    ),
                                }
                            ],
                        }
                        for r in range(4)
                    ],
                )
                return {"status": "ok"}

            with patch.object(
                sys,
                "argv",
                [
                    "run_overall",
                    "--nsys",
                    "--h2d-pair",
                    "--sharegpt",
                    str(data_path),
                    "--output",
                    str(output),
                ],
            ), patch.object(overall, "run_cell", side_effect=profiled_cell):
                overall.main()
            comparison = json.loads(
                next(output.rglob("nsys-comparison.json")).read_text()
            )
            self.assertEqual(
                [r["families"][0]["delta_kernel_ms"] for r in comparison["ranks"]],
                [2] * 4,
            )
            self.assertFalse(list(output.rglob("comparison.json")))
            with (output / "summary.csv").open() as handle:
                for row in csv.DictReader(handle):
                    self.assertEqual(
                        (row["nsys"], row["output_tps"], row["mean_tpot_ms"]),
                        ("True", "", ""),
                    )

    def test_profile_shutdown_allows_writer_to_finish(self):
        import psutil

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            env = os.environ | {
                "SGLANG_LAGOM_PERF_TRACE": str(folder / "trace_config.json")
            }
            cell = self.cell | {"nsys": True}
            decode = overall.commands(self.cfg, cell, folder, folder)[0]["decode"]
            # Use the exact target wrapper from the production argv.
            target = decode[decode.index("sh") : decode.index(self.cfg["python"])] + [
                "sleep",
                "30",
            ]
            supervisor = [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys\n"
                    "subprocess.run(sys.argv[2:], check=False)\n"
                    "pathlib.Path(sys.argv[1]).write_text('done')"
                ),
                str(folder / "closed"),
                *target,
            ]
            with subprocess.Popen(supervisor, env=env) as process:
                expires = time.monotonic() + 5
                pidfile = folder / "decode-target.pid"
                while time.monotonic() < expires and not (
                    pidfile.exists() and pidfile.stat().st_size
                ):
                    time.sleep(0.01)
                overall.finish_profile(
                    process, psutil.Process(int(pidfile.read_text())), expires
                )
                self.assertEqual((folder / "closed").read_text(), "done")

    def test_local_sharegpt_fixed_ids_through_dataset_loader(self):
        from argparse import Namespace
        from types import SimpleNamespace
        from sglang.benchmark.datasets import get_dataset

        tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens: list(text.encode())
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sharegpt.json"
            path.write_text(
                json.dumps(
                    [
                        {"conversations": [{"from": "human", "value": str(i) * 40}]}
                        for i in range(5)
                    ]
                )
            )
            args = Namespace(
                dataset_name="sharegpt",
                dataset_path=str(path),
                num_prompts=3,
                sharegpt_input_len=32,
                sharegpt_output_len=24,
                sharegpt_context_len=None,
                tokenize_prompt=True,
                apply_chat_template=False,
                prompt_suffix="",
                seed=42,
            )
            rows = get_dataset(args, tokenizer)
            self.assertEqual(
                [row.prompt for row in rows],
                [row.prompt for row in get_dataset(args, tokenizer)],
            )
            self.assertTrue(
                all(
                    len(row.prompt) == row.prompt_len == 32 and row.output_len == 24
                    for row in rows
                )
            )
            args.num_prompts = 100
            with self.assertRaisesRegex(ValueError, "insufficient tokens"):
                get_dataset(args, tokenizer)


if __name__ == "__main__":
    unittest.main()
