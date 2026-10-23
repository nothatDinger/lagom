"""Fixed-shape metadata operations for the V4 OasisKV graph pipeline."""

import triton as tr
import triton.language as tl


@tr.jit
def filter_history_candidates(
    predicted,
    prefixes,
    valid_drafts,
    num_real_reqs,
    output,
    INPUT_STRIDE: tl.constexpr,
    K: tl.constexpr,
    COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    active = b < tl.load(num_real_reqs)
    valid = tl.load(valid_drafts + b) & active
    # Neither the root's newly completed C4 nor a draft-only C4 is authoritative
    # host KV yet. The root is promoted to its fixed slot after the graph joins.
    history = tl.load(prefixes + b) // COMPRESS
    candidate = tl.load(predicted + b * INPUT_STRIDE + i, i < K, other=-1)
    candidate = tl.where(
        valid & (candidate >= 0) & (candidate < history), candidate, -1
    )
    tl.store(output + b * K + i, candidate, i < K)


@tr.jit
def publish_hot_metadata(
    req_indices,
    num_real_reqs,
    planned_tokens,
    planned_lru,
    tokens,
    lru,
    HOT: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    # Padding may repeat a live request ID; a masked index_copy_ is insufficient.
    if b < tl.load(num_real_reqs):
        req = tl.load(req_indices + b)
        i = tl.arange(0, BLOCK)
        token = tl.load(planned_tokens + b * HOT + i, i < HOT, other=-1)
        age = tl.load(planned_lru + b * HOT + i, i < HOT, other=0)
        tl.store(tokens + req * TOKEN_STRIDE + i, token, i < HOT)
        tl.store(lru + req * HOT + i, age, i < HOT)
