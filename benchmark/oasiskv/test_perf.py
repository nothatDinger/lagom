"""CPU protocol tests; all model/server data here is synthetic test data."""

import asyncio
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiohttp
import msgspec
import numpy as np
import zmq.asyncio
from aiohttp import web
from client import ForwardCollector, generate, sse_events, workload
from common import (
    HERE,
    load_config,
    matrix,
    select_workload,
    server_command,
    summarize,
    write_json,
)
from prepare import packed_prompts, prepare
from report import collect, save_csv
from run import ensure_port_free, main, resolve_checkpoint, validate_server_info


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode())


class ProtocolTests(unittest.TestCase):
    def test_resolved_backend_and_lookahead_gate_are_checked(self):
        config = load_config(HERE / "config.json")
        point = next(matrix(config, ["qwen3-8b"], ["oasiskv"]))
        state = {
            "attention_backend": "oasiskv",
            "tp_size": 1,
            "disable_radix_cache": True,
            "context_length": 40960,
            "max_running_requests": 8,
            "is_oasiskv_lookahead": True,
            "speculative_num_draft_tokens": 2,
            "cuda_graph_config": {"decode": {"backend": "full", "max_bs": 8}},
        }
        validate_server_info({"internal_states": [state]}, config, point)
        for key, value in (
            ("attention_backend", "fa3"),
            ("is_oasiskv_lookahead", False),
            ("disable_cuda_graph", True),
            ("cuda_graph_config", {"decode": {"backend": "disabled", "max_bs": 8}}),
            ("cuda_graph_config", {"decode": {"backend": "full", "max_bs": 4}}),
        ):
            with self.assertRaises(ValueError):
                validate_server_info(
                    {"internal_states": [{**state, key: value}]}, config, point
                )

    def test_preparation_writes_verifiable_memmap_and_refuses_overwrite(self):
        config = load_config(HERE / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            write_json(model / "tokenizer.json", {"fixture": True})
            source = root / "sharegpt.json"
            write_json(
                source, [{"conversations": [{"from": "human", "value": "sample text"}]}]
            )
            config.update(
                dataset=str(source),
                num_requests=3,
                warmup_requests=1,
                input_lengths=[8, 16],
            )
            config["models"]["qwen3-8b"]["path"] = str(model)
            with patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ) as loader:
                prepare(config, root / "prepared", ["qwen3-8b"])
                self.assertTrue(loader.call_args.kwargs["local_files_only"])
                with self.assertRaises(FileExistsError):
                    prepare(config, root / "prepared", ["qwen3-8b"])
            array = np.load(
                root / "prepared/qwen3-8b/prompts.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            self.assertEqual(array.shape, (4, 16))
            manifest = json.loads(
                (root / "prepared/qwen3-8b/manifest.json").read_text()
            )
            self.assertEqual(manifest["count"], 4)
            self.assertEqual(len(manifest["prompts_sha256"]), 64)

    def test_matrix_and_commands(self):
        config = load_config(HERE / "config.json")
        points = list(matrix(config))
        self.assertEqual(len(points), 144)
        self.assertEqual({p["concurrency"] for p in points}, {8, 16, 32, 64, 128, 256})
        self.assertEqual({p["output_length"] for p in points}, {2048})
        self.assertEqual({p["input_length"] for p in points}, {16384, 32768})
        oasis = next(p for p in points if p["method"] == "oasiskv")
        cmd = server_command(config, oasis, 30000, "ipc:///tmp/example", "/draft")
        self.assertEqual(cmd[cmd.index("--speculative-draft-model-path") + 1], "/draft")
        self.assertIn("--disable-radix-cache", cmd)
        self.assertNotIn("--enable-hisparse", cmd)
        self.assertNotIn("--max-total-tokens", cmd)
        self.assertEqual(
            json.loads(cmd[cmd.index("--cuda-graph-config") + 1]),
            {"decode": {"backend": "full", "max_bs": oasis["concurrency"]}},
        )
        dense = server_command(config, points[0], 30000, "ipc:///tmp/example")
        self.assertNotIn("--speculative-draft-model-path", dense)
        self.assertEqual(dense[dense.index("--attention-backend") + 1], "fa3")
        smoke = list(matrix(config, ["qwen3-8b"], smoke=True))
        self.assertEqual(len(smoke), 2)
        self.assertTrue(
            all(p["concurrency"] == 1 and p["protocol"] == "smoke" for p in smoke)
        )

    def test_config_rejects_context_and_concurrency_errors(self):
        config = load_config(HERE / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for key, value in (
                ("context_length", 32768),
                ("concurrencies", [512]),
                ("num_requests", 1),
            ):
                modified = copy.deepcopy(config)
                modified[key] = value
                write_json(path, modified)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_concurrency_filters_keep_client_server_and_graph_limits_aligned(self):
        config = load_config(HERE / "config.json")
        selected = select_workload(config, [8, 16], [16384])
        points = list(matrix(selected, ["qwen3-235b-a22b"]))
        self.assertEqual(len(points), 12)
        self.assertEqual(config["concurrencies"], [8, 16, 32, 64, 128, 256])
        for point in points:
            cmd = server_command(selected, point, 30000, "ipc:///tmp/example")
            self.assertEqual(cmd[cmd.index("--tp") + 1], "8")
            self.assertEqual(
                int(cmd[cmd.index("--max-running-requests") + 1]), point["concurrency"]
            )
            self.assertEqual(
                json.loads(cmd[cmd.index("--cuda-graph-config") + 1])["decode"][
                    "max_bs"
                ],
                point["concurrency"],
            )
            self.assertEqual(point["num_requests"], 1024)
            self.assertEqual(point["output_length"], 2048)
            self.assertEqual(point["input_length"], 16384)
        for kwargs in (
            {"concurrencies": [8], "smoke": True},
            {"input_lengths": [16384], "smoke": True},
            {"concurrencies": [8, 8]},
            {"concurrencies": [7]},
            {"input_lengths": [16000]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_workload(config, **kwargs)

    def test_plan_exposes_the_client_load_and_device_selection(self):
        output = io.StringIO()
        with patch(
            "sys.argv",
            [
                "run.py",
                "plan",
                "--models",
                "qwen3-8b",
                "--concurrencies",
                "8",
                "16",
                "--input-lengths",
                "16384",
            ],
        ), patch("sys.stdout", output):
            self.assertEqual(main(), 0)
        planned = json.loads(output.getvalue())
        self.assertEqual(planned["count"], 12)
        for point in planned["points"]:
            self.assertEqual(point["cuda_visible_devices"], "0")
            self.assertEqual(
                point["client"],
                {
                    "implementation": "custom",
                    "mode": "closed-loop",
                    "max_concurrency": point["concurrency"],
                    "num_requests": 1024,
                    "input_length": 16384,
                    "output_length": 2048,
                    "request_rate_limit": None,
                },
            )

    def test_packing_is_exact_deterministic_and_uses_all_turns(self):
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "World"},
                ]
            },
            {"conversation": [{"role": "user", "content": "Goodbye"}]},
        ]
        a = list(packed_prompts(records, Tokenizer(), 5, 37, 42))
        b = list(packed_prompts(records, Tokenizer(), 5, 37, 42))
        self.assertEqual(a, b)
        self.assertEqual([len(p) for p in a], [37] * 5)
        self.assertIn(b"World", bytes(sum(a, [])))
        self.assertNotEqual(a[0], a[1])
        with self.assertRaises(ValueError):
            list(packed_prompts([], Tokenizer(), 1, 32, 42))

    def test_metrics_use_real_batch_and_request_tpot(self):
        requests = [
            {
                "success": True,
                "start_s": 0,
                "first_token_s": 1,
                "last_token_s": 4,
                "output_tokens": 4,
                "first_chunk_tokens": 1,
            },
            {
                "success": True,
                "start_s": 0,
                "first_token_s": 2,
                "last_token_s": 8,
                "output_tokens": 4,
                "first_chunk_tokens": 1,
            },
        ]
        frames = [
            {
                "counter_id": seq,
                "scheduled_requests": {
                    "num_decode_requests": b,
                    "num_prefill_requests": p,
                },
            }
            for seq, b, p in ((10, 0, 2), (11, 4, 0), (13, 2, 0))
        ]
        result = summarize(requests, frames, 0, 10)
        self.assertEqual(result["output_tps"], 0.8)
        self.assertAlmostEqual(result["decode_window_tps"], 6 / 7)
        self.assertEqual(result["tpot_mean_ms"], 1500)
        self.assertEqual(result["running_batch_mean"], 3)
        self.assertEqual(result["fpm_sequence_gaps"], 1)
        failed = summarize([{"success": False}], [], 0, 1)
        self.assertEqual(failed["failed_requests"], 1)
        self.assertIsNone(failed["running_batch_mean"])
        self.assertIsNone(failed["tpot_mean_ms"])

    def test_report_excludes_failed_points_and_archived_attempts(self):
        point = next(matrix(load_config(HERE / "config.json")))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "a" / "result.json", {**point, "status": "ok", "output_tps": 100}
            )
            write_json(
                root / "b" / "result.json",
                {**point, "repeat": 1, "status": "failed", "output_tps": 0},
            )
            write_json(
                root / "b.attempt-1" / "result.json",
                {**point, "status": "ok", "output_tps": 999},
            )
            rows, aggregate = collect(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(aggregate[0]["valid_repeats"], 1)
            self.assertEqual(aggregate[0]["output_tps"], 100)
            save_csv(root / "summary.csv", aggregate)
            self.assertTrue((root / "summary.csv").is_file())

    def test_checkpoint_requires_all_weight_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "config.json", {})
            write_json(
                root / "model.safetensors.index.json",
                {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}},
            )
            (root / "one.safetensors").touch()
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(tmp)
            (root / "two.safetensors").touch()
            self.assertEqual(resolve_checkpoint(tmp), str(root.resolve()))


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.active = 0
        self.peak = 0
        self.fail = False
        self.bad_length = False
        self.payloads = []
        self.runner = web.AppRunner(self.make_app())
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    def make_app(self):
        app = web.Application()
        app.router.add_post("/generate", self.handler)
        return app

    async def handler(self, request):
        payload = await request.json()
        self.payloads.append(payload)
        if self.fail:
            return web.Response(status=500, text="simulated failure")
        self.active += 1
        self.peak = max(self.peak, self.active)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        n = payload["sampling_params"]["max_new_tokens"]
        for count in range(1, n + 1 - int(self.bad_length)):
            event = {
                "text": "",
                "meta_info": {
                    "completion_tokens": count,
                    "prompt_tokens": len(payload["input_ids"]),
                    "finish_reason": {"type": "length"} if count == n else None,
                },
            }
            wire = b"data: " + json.dumps(event).encode() + b"\n\n"
            # Fragment each frame, even the 'data:' prefix; text is empty but
            # completion_tokens still makes this a real output-token event.
            await response.write(wire[:3])
            await asyncio.sleep(0.001)
            await response.write(wire[3:])
        await response.write(b"data: [DONE]\n\n")
        self.active -= 1
        return response

    async def test_stream_uses_counts_not_text(self):
        result = await generate(self.session, self.url, [4, 5, 6], 4, "test")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["output_tokens"], 4)
        self.assertEqual(result["first_chunk_tokens"], 1)
        self.assertGreater(result["last_token_s"], result["first_token_s"])
        self.assertTrue(self.payloads[0]["sampling_params"]["ignore_eos"])

    async def test_error_and_truncated_response_are_failures(self):
        self.fail = True
        result = await generate(self.session, self.url, [1, 2], 4, 0)
        self.assertFalse(result["success"])
        self.assertIn("HTTP 500", result["error"])
        self.fail = False
        self.bad_length = True
        result = await generate(self.session, self.url, [1, 2], 4, 0)
        self.assertFalse(result["success"])
        self.assertIn("received 3", result["error"])

    async def test_closed_loop_respects_max_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, start, end = await workload(
                self.session,
                self.url,
                np.ones((7, 16), dtype=np.int32),
                8,
                4,
                3,
                Path(tmp) / "requests.jsonl",
            )
            self.assertEqual(len(result), 7)
            self.assertTrue(all(r["success"] for r in result))
            self.assertEqual(self.peak, 3)
            self.assertGreater(end, start)
            self.assertEqual({r["request_id"] for r in result}, set(range(7)))
            self.assertEqual(
                len((Path(tmp) / "requests.jsonl").read_text().splitlines()), 7
            )

    async def test_occupied_port_is_rejected(self):
        with self.assertRaises(OSError):
            ensure_port_free(self.port)

    async def test_multiline_sse_and_eof(self):
        class Content:
            def __init__(self, data):
                self.data = io.BytesIO(data)

            async def readline(self):
                return self.data.readline()

        events = [
            v
            async for v in sse_events(
                Content(b':comment\r\ndata: {"x":\r\ndata: 1}\r\n\r\ndata: {"x":2}\n')
            )
        ]
        self.assertEqual(events, [{"x": 1}, {"x": 2}])

    async def test_forward_collector_warmup_boundary_and_sequence_gap(self):
        with tempfile.TemporaryDirectory(prefix="okv-test-") as tmp:
            endpoint = "ipc://" + tmp + "/fpm"
            context = zmq.asyncio.Context()
            publisher = context.socket(zmq.XPUB)
            publisher.bind(endpoint + ".0")
            collector = ForwardCollector(endpoint, Path(tmp) / "forward.jsonl")
            try:
                await asyncio.wait_for(publisher.recv(), 3)  # SUB connected.

                async def send(seq, batch):
                    record = {
                        "version": 1,
                        "dp_rank": 0,
                        "counter_id": seq,
                        "wall_time": 0.01 if batch else 0,
                        "scheduled_requests": {
                            "num_decode_requests": batch,
                            "num_prefill_requests": 0,
                        },
                    }
                    await publisher.send_multipart(
                        [b"", seq.to_bytes(8, "big"), msgspec.msgpack.encode(record)]
                    )

                waiter = asyncio.create_task(collector.wait_idle(3))
                await asyncio.sleep(0)
                await send(0, 4)
                await send(1, 0)
                await waiter
                self.assertEqual(collector.rows, [])
                collector.phase = "measurement"
                waiter = asyncio.create_task(collector.wait_idle(3))
                await asyncio.sleep(0)
                await send(2, 8)
                await send(4, 4)
                await send(5, 0)
                await waiter
                result = summarize([], collector.rows, 0, 1)
                self.assertEqual(result["running_batch_mean"], 6)
                self.assertEqual(result["fpm_sequence_gaps"], 1)
                self.assertEqual(len(collector.errors), 1)
                self.assertIn("2 -> 4", collector.errors[0])
            finally:
                await collector.close()
                publisher.close(linger=0)
                context.term()


if __name__ == "__main__":
    unittest.main()
