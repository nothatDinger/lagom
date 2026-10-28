"""Native bench protocol checks, including the real CLI against a CPU SSE fixture."""

import asyncio
import io
import json
import math
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from common import HERE, ROOT, load_config, write_json
from run import main, preflight
from sglang_bench import (
    audit_result,
    dataset_metadata,
    measure,
    read_result,
    run_bench,
)


def point():
    return {
        "model": "qwen3-8b",
        "method": "oasiskv",
        "protocol": "fig11-sglang-bench",
        "num_requests": 4,
        "input_length": 16,
        "output_length": 3,
        "concurrency": 2,
    }


def native_result():
    return {
        "backend": "sglang",
        "dataset_name": "random",
        "random_input_len": 16,
        "random_output_len": 3,
        "random_range_ratio": 1,
        "max_concurrency": 2,
        "request_rate": math.inf,
        "completed": 4,
        "total_input_tokens": 64,
        "total_output_tokens": 12,
        "input_lens": [16] * 4,
        "output_lens": [3] * 4,
        "errors": [""] * 4,
        "generated_texts": ["hello"] * 4,
        "ttfts": [0.1] * 4,
        "itls": [[0.2, 0.2]] * 4,
        "accept_length": 1,
        "output_throughput": 6.0,
        "duration": 2.0,
        "mean_tpot_ms": 200,
        "median_tpot_ms": 200,
        "p95_tpot_ms": 200,
        "mean_ttft_ms": 100,
    }


class NativeProtocolTests(unittest.TestCase):
    def test_plan_keeps_tp8_global_concurrency_and_exact_lengths(self):
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "run.py",
                "plan",
                "--client",
                "sglang-bench",
                "--models",
                "qwen3-235b-a22b",
                "--concurrencies",
                "8",
                "16",
                "--input-lengths",
                "16384",
            ],
        ), patch("sys.stdout", stdout):
            self.assertEqual(main(), 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["count"], 12)
        for row in plan["points"]:
            server = shlex.split(row["command"])
            bench = shlex.split(row["client"]["command"])
            warmup = shlex.split(row["client"]["warmup_command"])
            self.assertEqual(server[server.index("--tp") + 1], "8")
            for cmd, flag in (
                (server, "--max-running-requests"),
                (bench, "--max-concurrency"),
            ):
                self.assertEqual(int(cmd[cmd.index(flag) + 1]), row["concurrency"])
            for key, value in {
                "--dataset-name": "random",
                "--random-range-ratio": "1",
                "--random-input-len": "16384",
                "--random-output-len": "2048",
                "--num-prompts": "1024",
                "--request-rate": "inf",
                "--warmup-requests": "0",
            }.items():
                self.assertEqual(bench[bench.index(key) + 1], value)
            self.assertIn("--tokenize-prompt", bench)
            self.assertNotIn("--disable-ignore-eos", bench)
            self.assertNotIn("--disable-stream", bench)
            self.assertEqual(row["protocol"], "fig11-sglang-bench")
            self.assertEqual(warmup[warmup.index("--max-concurrency") + 1], "1")

    def test_audit_does_not_use_retokenized_counts_or_client_concurrency(self):
        raw = native_result()
        raw.update(total_output_tokens_retokenized=999, concurrency=999)
        metrics, reasons = audit_result(raw, point())
        self.assertFalse(reasons)
        self.assertEqual(metrics["output_tps"], 6)
        self.assertEqual(metrics["total_output_tokens"], 12)
        self.assertNotIn("running_batch_mean", metrics)
        self.assertIsNone(metrics["decode_window_tps"])
        for key, value in (
            ("completed", 3),
            ("output_lens", [3, 3, 3, 2]),
            ("input_lens", [15] * 4),
            ("errors", ["timeout"] * 4),
            ("generated_texts", [""] * 4),
            ("ttfts", [0] * 4),
            ("itls", [[]] * 4),
            ("accept_length", 1.5),
            ("mean_tpot_ms", math.nan),
            ("output_throughput", 7),
        ):
            with self.subTest(key=key):
                _, reasons = audit_result({**raw, key: value}, point())
                self.assertTrue(reasons)
        metrics, reasons = audit_result({**raw, "accept_length": None}, point())
        self.assertFalse(reasons)
        self.assertEqual(metrics["acceptance_telemetry"], "unavailable")

    def test_raw_infinity_is_retained_but_normalized_json_is_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            path.write_text(json.dumps(native_result()) + "\n")
            metrics, reasons = audit_result(read_result(path), point())
            self.assertFalse(reasons)
            write_json(Path(tmp) / "result.json", metrics)
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                read_result(path)

    def test_native_preflight_requires_sharegpt_but_not_prepared_npy(self):
        config = load_config(HERE / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.json"
            write_json(
                source,
                [{"conversations": [{"value": "hello"}, {"value": "world"}]}] * 4,
            )
            config.update(dataset=str(source), num_requests=4)
            write_json(root / "config.json", {})
            write_json(root / "tokenizer.json", {})
            (root / "model.safetensors").write_bytes(b"fixture")
            config["models"]["qwen3-8b"].update(path=str(root), draft=str(root))
            gpu = SimpleNamespace(
                stdout=json.dumps([{"name": "H100", "bytes": 80 * 1024**3}])
            )
            with patch("run.subprocess.run", return_value=gpu), patch(
                "run.importlib.util.find_spec", return_value=True
            ):
                checked = preflight(
                    config,
                    ["qwen3-8b"],
                    ["oasiskv"],
                    root / "missing-prepared-data",
                    "sglang-bench",
                )
            self.assertTrue(checked["ok"], checked["errors"])
            self.assertEqual(
                checked["resolved"]["qwen3-8b"]["data_manifest"]["eligible_prompts"], 4
            )
            with self.assertRaises(ValueError):
                dataset_metadata(source, 5)


class NativeCLITests(unittest.IsolatedAsyncioTestCase):
    async def test_real_cli_sends_exact_ids_streams_and_separates_warmup(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        config = load_config(HERE / "config.json")
        seen, active, peaks = [], 0, {"warmup": 0, "measurement": 0}

        class Collector:
            def __init__(self):
                self.phase = "warmup"
                self.rows = []
                self.errors = []

            async def wait_idle(self):
                pass

        collector = Collector()

        async def generate(request):
            nonlocal active
            payload = await request.json()
            seen.append((collector.phase, payload))
            active += 1
            peaks[collector.phase] = max(peaks[collector.phase], active)
            if collector.phase == "measurement":
                collector.rows.append(
                    {
                        "counter_id": len(collector.rows),
                        "scheduled_requests": {"num_decode_requests": active},
                    }
                )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for i in range(1, payload["sampling_params"]["max_new_tokens"] + 1):
                await asyncio.sleep(0.01)
                chunk = {
                    "text": "hello " * i,
                    "meta_info": {
                        "completion_tokens": i,
                        "prompt_tokens": len(payload["input_ids"]),
                    },
                }
                await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            active -= 1
            return response

        async def info(request):
            return web.json_response(
                {"internal_states": [{"avg_spec_accept_length": 1}]}
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        app.router.add_get("/server_info", info)
        app.router.add_get("/v1/models", info)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tok = Tokenizer(
                    WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
                )
                tok.pre_tokenizer = Whitespace()
                PreTrainedTokenizerFast(
                    tokenizer_object=tok, unk_token="[UNK]"
                ).save_pretrained(root)
                config["models"]["qwen3-8b"]["path"] = str(root)
                config.update(
                    dataset=str(root / "sharegpt.json"),
                    warmup_requests=2,
                    warmup_output_length=2,
                    point_timeout_s=60,
                )
                write_json(
                    config["dataset"],
                    [{"conversations": [{"value": "hello world"}, {"value": "hello"}]}]
                    * 4,
                )
                env = {
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "python"),
                    "SGLANG_USE_CPU_ENGINE": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
                try:
                    result = await measure(config, point(), root, port, env, collector)
                except Exception:
                    for log in root.glob("*.log"):
                        print(log.read_text()[-6000:])
                    raise
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["decode_forward_samples"], 4)
                self.assertEqual(result["running_batch_max"], 2)
                self.assertEqual(result["total_output_tokens"], 12)
                self.assertEqual(peaks, {"warmup": 1, "measurement": 2})
                self.assertEqual(len(seen), 6)
                for phase, payload in seen:
                    self.assertEqual(payload["input_ids"], [1, 2] * 8)
                    self.assertTrue(payload["stream"])
                    self.assertTrue(payload["sampling_params"]["ignore_eos"])
                    self.assertEqual(
                        payload["sampling_params"]["max_new_tokens"],
                        2 if phase == "warmup" else 3,
                    )
                for path in (
                    "bench.jsonl",
                    "bench-warmup.jsonl",
                    "bench-launch.json",
                    "bench-warmup-launch.json",
                ):
                    self.assertTrue((root / path).is_file())
        finally:
            await runner.cleanup()

    async def test_timeout_reaps_owned_bench_process(self):
        config = load_config(HERE / "config.json")
        config["point_timeout_s"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pidfile = root / "pid"
            script = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
            with (
                patch(
                    "sglang_bench.bench_command",
                    return_value=[sys.executable, "-c", script, str(pidfile)],
                ),
                self.assertRaises(TimeoutError),
            ):
                await run_bench(config, point(), root, 30000, os.environ)
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
