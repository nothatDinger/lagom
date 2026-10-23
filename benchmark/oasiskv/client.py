"""Streaming native /generate client and per-forward scheduler telemetry."""

import asyncio
import json
import time


async def sse_events(content):
    """Read complete SSE lines, including frames split across TCP packets."""
    lines = []
    while True:
        line = await content.readline()
        if not line:
            if lines:
                data = b"\n".join(lines)
                if data != b"[DONE]":
                    yield json.loads(data)
            return
        line = line.rstrip(b"\r\n")
        if not line:
            if not lines:
                continue
            data = b"\n".join(lines)
            lines.clear()
            if data == b"[DONE]":
                return
            yield json.loads(data)
        elif line.startswith(b"data:"):
            lines.append(line[5:].lstrip(b" "))


async def generate(session, base_url, prompt, output_length, request_id):
    result = {
        "request_id": request_id,
        "start_s": time.perf_counter(),
        "success": False,
        "input_tokens": len(prompt),
        "output_tokens": 0,
        "first_chunk_tokens": 0,
    }
    payload = {
        "input_ids": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_length,
            "ignore_eos": True,
        },
        "stream": True,
    }
    try:
        async with session.post(base_url + "/generate", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:2000]}"
                )
            last_meta = {}
            async for event in sse_events(response.content):
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                meta = event.get("meta_info", {})
                count = meta.get("completion_tokens", 0)
                if count < result["output_tokens"]:
                    raise ValueError("Non-monotonic completion_tokens")
                if count > result["output_tokens"]:
                    now = time.perf_counter()
                    if result["output_tokens"] == 0:
                        result["first_token_s"] = now
                        result["first_chunk_tokens"] = count
                    result["last_token_s"] = now
                    result["output_tokens"] = count
                last_meta.update(meta)
            result["server_prompt_tokens"] = last_meta.get("prompt_tokens")
            result["finish_reason"] = last_meta.get("finish_reason")
            result["spec_accept_length"] = last_meta.get("spec_accept_length")
            if result["output_tokens"] != output_length:
                raise ValueError(
                    f"Expected {output_length} output tokens, received {result['output_tokens']}"
                )
            if result["server_prompt_tokens"] != len(prompt):
                raise ValueError(
                    "Server prompt token count does not match the prepared input"
                )
            reason = result["finish_reason"]
            if not isinstance(reason, dict) or reason.get("type") != "length":
                raise ValueError(f"Expected length finish_reason, received {reason}")
            result["success"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["end_s"] = time.perf_counter()
    return result


async def workload(
    session, base_url, prompts, length, output_length, concurrency, output_file
):
    """Closed loop: maintain up to concurrency outstanding requests until drained."""
    cursor = iter(range(len(prompts)))
    results = []
    with open(output_file, "w") as stream:

        async def worker():
            for index in cursor:
                row = await generate(
                    session,
                    base_url,
                    prompts[index, :length].tolist(),
                    output_length,
                    index,
                )
                results.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        end = time.perf_counter()
    return results, start, end


class ForwardCollector:
    def __init__(self, endpoint, output_file):
        import msgspec
        import zmq
        import zmq.asyncio

        self.zmq = zmq
        self.decode = msgspec.msgpack.decode
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 100000)
        self.socket.connect(endpoint + ".0")  # TP only; one publishing rank, no DP.
        self.stream = open(output_file, "w")
        self.phase = "warmup"
        self.rows = []
        self.errors = []
        self.running = True
        self.last_received = time.perf_counter()
        self.last_counter = None
        self.seen_heartbeat = asyncio.Event()
        self.task = asyncio.create_task(self.collect())

    async def collect(self):
        while self.running:
            if not await self.socket.poll(timeout=100):
                continue
            frames = await self.socket.recv_multipart()
            try:
                if len(frames) != 3:
                    raise ValueError("Unexpected FPM frame count")
                data = self.decode(frames[2])
                if data.get("version") != 1 or data.get("dp_rank") != 0:
                    raise ValueError("Unsupported FPM schema or DP topology")
                if int.from_bytes(frames[1], "big") != data["counter_id"]:
                    raise ValueError("FPM sequence mismatch")
                if (
                    self.last_counter is not None
                    and data["counter_id"] != self.last_counter + 1
                ):
                    self.errors.append(
                        f"FPM sequence discontinuity: {self.last_counter} -> {data['counter_id']}"
                    )
                self.last_counter = data["counter_id"]
                now = time.perf_counter()
                data.update(received_s=now, phase=self.phase)
                self.last_received = now
                self.stream.write(json.dumps(data) + "\n")
                if self.phase == "measurement":
                    self.rows.append(data)
                if data["wall_time"] == 0:
                    self.seen_heartbeat.set()
            except Exception as error:
                self.errors.append(str(error))

    async def wait_idle(self, timeout=30):
        # Publisher FIFO heartbeat drains preceding forward records. Warmup and
        # measurement remain separate even if the publisher thread is delayed.
        self.seen_heartbeat.clear()
        await asyncio.wait_for(self.seen_heartbeat.wait(), timeout)

    async def close(self):
        self.running = False
        try:
            await self.task
        finally:
            self.socket.close(linger=0)
            self.context.term()
            self.stream.close()
