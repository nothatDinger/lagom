"""Start a bounded profile directly on the decode server during steady load."""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--activity", choices=["torch", "nsys"], default="torch")
    opts = parser.parse_args()
    if opts.steps <= 0:
        parser.error("steps must be positive")
    opts.output_dir.mkdir(parents=True, exist_ok=True)
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
    (opts.output_dir / "capture-request.json").write_text(
        json.dumps(dict(request=body, response=result), indent=2) + "\n"
    )
    if not result["accepted"]:
        raise SystemExit(
            f"Profile response was not recognized as success: {result}. "
            "Check server logs and existing capture files before triggering again."
        )
    print(json.dumps(result, indent=2))
    print(
        "Profile stops automatically after the requested scheduler steps. "
        "Wait for 'Profiling done' from every decode TP rank before collecting files."
    )


if __name__ == "__main__":
    main()
