"""Start a bounded profile directly on the decode server during steady load."""

import argparse
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path


def decode_profile_response(status, content_type, body):
    text = body.decode("utf-8", errors="replace")
    result = dict(
        status=status, content_type=content_type, body=text, bytes_received=len(body)
    )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    result["json"] = parsed
    result["accepted"] = 200 <= status < 300 and (
        (isinstance(parsed, dict) and parsed.get("success") is True)
        or text.strip() == "Start profiling."
    )
    return result


def capture_artifacts(output_dir, request, expected_ranks):
    """Check the V1 torch exporter completion order without loading large traces.

    _stop_profile exports/finishes traces, crosses a TP barrier, then writes the
    probes. A readable current-capture probe plus its trace is an export barrier,
    not a substitute for the downstream trace/probe timing validation.
    """
    output_dir = Path(output_dir)
    capture_id = request["profile_id"]
    if (
        request.get("profile_by_stage") is not False
        or request.get("merge_profiles") is not False
        or request.get("activities") != ["CPU", "GPU"]
    ):
        raise ValueError("Waiting supports V1 torch, unmerged, non-stage profiles only")
    prefix = request.get("profile_prefix", "")
    stem = f"{prefix}-" if prefix else ""
    ready, pending = [], {}
    for rank in range(expected_ranks):
        base = f"{stem}{capture_id}-TP-{rank}"
        probe_pattern = re.compile(re.escape(base) + r"-PID-(\d+)-GPU-(\d+)\.h2d\.json")
        trace_pattern = re.compile(
            re.escape(base) + r"(?:-(?:DP|PP|EP)-\d+)*\.trace\.json\.gz"
        )
        probes = [
            p for p in output_dir.glob("*.h2d.json") if probe_pattern.fullmatch(p.name)
        ]
        traces = [
            p
            for p in output_dir.glob("*.trace.json.gz")
            if trace_pattern.fullmatch(p.name)
        ]
        if len(probes) > 1 or len(traces) > 1:
            raise ValueError(f"TP{rank}: ambiguous artifacts for capture {capture_id}")
        if not probes or not traces:
            pending[rank] = f"probe={len(probes)} trace={len(traces)}"
            continue
        try:
            report = json.loads(probes[0].read_text())
        except (OSError, json.JSONDecodeError):
            pending[rank] = "probe is not yet fully readable"
            continue
        match = probe_pattern.fullmatch(probes[0].name)
        pid, device = map(int, match.groups())
        if (
            report.get("capture_id") != capture_id
            or report.get("pid") != pid
            or report.get("device") != device
        ):
            raise ValueError(f"TP{rank}: probe capture/PID/device identity mismatch")
        samples = report.get("samples", [])
        if (
            report.get("schema_version") != 1
            or report.get("overflow") is not False
            or not samples
            or report.get("total_records") != len(samples)
        ):
            raise ValueError(
                f"TP{rank}: incomplete/empty/overflowed probe; retain artifacts"
            )
        try:
            size = traces[0].stat().st_size
        except OSError:
            size = 0
        if size == 0:
            pending[rank] = "trace is empty or not yet visible"
            continue
        ready.append(
            dict(
                rank=rank,
                pid=pid,
                device=device,
                samples=len(samples),
                probe=probes[0].name,
                trace=traces[0].name,
                trace_bytes=size,
            )
        )
    if len({r["device"] for r in ready}) != len(ready):
        raise ValueError("Expected one distinct CUDA device per TP rank")
    return dict(
        capture_id=capture_id,
        expected_ranks=expected_ranks,
        artifacts_complete=len(ready) == expected_ranks,
        timing_analysis_validated=False,
        ranks=ready,
        pending=pending,
    )


def write_status(path, status):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def wait_for_capture(output_dir, request, expected_ranks, timeout_seconds):
    start = time.monotonic()
    last_print = -math.inf
    while True:
        status = capture_artifacts(output_dir, request, expected_ranks)
        elapsed = time.monotonic() - start
        status["wait_elapsed_seconds"] = elapsed
        if status["artifacts_complete"]:
            write_status(Path(output_dir) / "capture-complete.json", status)
            print(
                f"Capture {status['capture_id']} artifacts complete: "
                f"{expected_ranks}/{expected_ranks} TP ranks; "
                "run the existing analyzer to validate timings.",
                flush=True,
            )
            return status
        if elapsed >= timeout_seconds:
            write_status(Path(output_dir) / "capture-incomplete.json", status)
            raise TimeoutError(
                f"Capture {status['capture_id']} timed out: {status['pending']}. "
                "Do not treat the HTTP acknowledgment as completion. "
                "Retain artifacts; --wait-only can resume without another POST."
            )
        if elapsed - last_print >= 10:
            print(
                f"Waiting for {status['capture_id']}: "
                f"{len(status['ranks'])}/{expected_ranks} ranks; pending={status['pending']}",
                flush=True,
            )
            last_print = elapsed
        time.sleep(min(1.0, timeout_seconds - elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--activity", choices=["torch", "nsys"], default="torch")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Return success only after current-capture TP artifacts exist",
    )
    parser.add_argument(
        "--wait-only",
        action="store_true",
        help="Resume waiting from capture-request.json without POSTing again",
    )
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    opts = parser.parse_args()
    if opts.steps <= 0:
        parser.error("steps must be positive")
    waiting = opts.wait or opts.wait_only
    if waiting and (
        opts.activity != "torch"
        or opts.expected_ranks is None
        or opts.expected_ranks <= 0
        or not math.isfinite(opts.timeout_seconds)
        or opts.timeout_seconds <= 0
    ):
        parser.error(
            "Waiting requires --activity torch, positive --expected-ranks and finite positive timeout"
        )
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = opts.output_dir / "capture-request.json"
    if opts.wait_only:
        saved = json.loads(request_path.read_text())
        if not saved["response"].get("accepted"):
            parser.error("The saved request was not accepted; inspect its response")
        wait_for_capture(
            opts.output_dir, saved["request"], opts.expected_ranks, opts.timeout_seconds
        )
        return
    if request_path.exists():
        parser.error(
            "Use a new output directory, or --wait-only to resume the existing capture"
        )
    body = dict(
        activities=["CPU", "GPU"] if opts.activity == "torch" else ["CUDA_PROFILER"],
        num_steps=opts.steps,
        profile_by_stage=False,
        with_stack=False,
        record_shapes=False,
        merge_profiles=False,
        profile_id=f"h2d-{time.time_ns()}",
        profile_prefix="round1",
        output_dir=str(opts.output_dir.resolve()),
    )
    request = urllib.request.Request(
        opts.decode_url.rstrip("/") + "/start_profile",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = decode_profile_response(
            response.status, response.headers.get("Content-Type", ""), response.read()
        )
    request_path.write_text(
        json.dumps(dict(request=body, response=result), indent=2) + "\n"
    )
    if not result["accepted"]:
        raise SystemExit(
            f"Profile response was not recognized as success: {result}. "
            "Check server logs and existing capture files before triggering again."
        )
    print(json.dumps(result, indent=2))
    if waiting:
        wait_for_capture(
            opts.output_dir, body, opts.expected_ranks, opts.timeout_seconds
        )
        return
    print(
        "Profile stops automatically after the requested scheduler steps. "
        "Wait for 'Profiling done' from every decode TP rank before collecting files."
    )


if __name__ == "__main__":
    main()
