"""Opt-in CPU progress markers, also usable with the trace hook disabled."""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("SGLANG_OASISKV_TRACE_DIAGNOSTICS") == "1"
_last_log = {}


def log_progress(stage, reqs, *, throttle_s=0, **details):
    if not ENABLED:
        return
    ids = tuple(str(req.rid) for req in reqs)
    now = time.monotonic()
    key = (stage, ids)
    if throttle_s and now - _last_log.get(key, -float("inf")) < throttle_s:
        return
    _last_log[key] = now
    # Deliberately no CUDA calls/readbacks: "returned" means the Python call
    # returned, not that its asynchronous GPU work has completed.
    logger.info(
        "OasisKV progress %s",
        json.dumps(
            {
                "stage": stage,
                "pid": os.getpid(),
                "monotonic_s": now,
                "request_ids": ids,
                **details,
            },
            sort_keys=True,
        ),
    )


def log_admission(adder, req, result):
    if not ENABLED:
        return
    extend = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
    log_progress(
        "admission",
        [req],
        throttle_s=30,
        result=result.name,
        admitted=req in adder.can_run_list,
        admitted_tokens=req.extend_range.length if req in adder.can_run_list else 0,
        prompt_tokens=len(req.origin_input_ids),
        extend_tokens=extend,
        full_free=adder.rem_total_tokens,
        chunk_limit=adder.rem_chunk_tokens,
        swa_free=adder.rem_swa_tokens if adder.is_hybrid_swa else None,
        swa_capacity=(
            adder.token_to_kv_pool_allocator.size_swa if adder.is_hybrid_swa else None
        ),
        swa_needed=(
            adder._swa_budget_for_req(extend, adder._swa_new_tokens(req))
            if adder.is_hybrid_swa
            else None
        ),
    )
