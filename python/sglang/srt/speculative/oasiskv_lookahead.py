"""OasisKV normal/draft paired-forward primitives.

OasisKV is look-ahead, not speculative decoding: the odd rows are disposable
target-model probes and can never be accepted as output.  This module keeps the
row/position/state contract in one place so model and attention backends do not
have to infer ownership from tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs


def build_oasiskv_pair_rows(
    batch_size: int, device: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct root/probe indices by geometry, never by GPU token values."""
    if batch_size < 0:
        raise ValueError("OasisKV batch size must be nonnegative")
    rows = torch.arange(2 * batch_size, dtype=torch.int64, device=device)
    return rows[0::2], rows[1::2]


class OasisKVCommitBuffers:
    """Runner-owned, read-only metadata for the fixed one-root commit.

    Allocate outside decode/capture. Consumers must not mutate the returned
    views: multiple in-flight batches can share these constants. The one-time
    initialization event handles callers on a different CUDA stream without
    introducing a host wait or a dependency on another decode step.
    """

    def __init__(self, capacity: int, device: Any):
        self.normal_rows, self.draft_rows = build_oasiskv_pair_rows(capacity, device)
        self.accept_lens = torch.ones(capacity, dtype=torch.int32, device=device)
        self._ready = None
        self._ready_streams = set()
        if self.normal_rows.is_cuda:
            stream = torch.cuda.current_stream(self.normal_rows.device)
            self._ready = torch.cuda.Event()
            self._ready.record(stream)
            self._ready_streams.add(stream.cuda_stream)

    def views(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= batch_size <= self.accept_lens.numel():
            raise ValueError("OasisKV commit batch exceeds preallocated capacity")
        if self._ready is not None:
            stream = torch.cuda.current_stream(self.normal_rows.device)
            if stream.cuda_stream not in self._ready_streams:
                stream.wait_event(self._ready)
                # A wait recorded into a graph has not executed yet. Do not
                # let that capture suppress a later eager stream dependency.
                if not torch.cuda.is_current_stream_capturing():
                    self._ready_streams.add(stream.cuda_stream)
        return self.accept_lens[:batch_size], self.normal_rows[:batch_size]


@dataclass(frozen=True)
class OasisKVPairedBatch:
    """Request-major ``[normal0,draft0,normal1,draft1,...]`` batch metadata."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    normal_rows: torch.Tensor
    draft_rows: torch.Tensor
    draft_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.normal_rows.numel()


def select_oasiskv_normal_rows(
    tensor: torch.Tensor, pair_width: int = 2
) -> torch.Tensor:
    """Select committed roots from a request-major target pair."""
    if pair_width != 2 or tensor.shape[0] % pair_width:
        raise ValueError("OasisKV expects a request-major normal/draft tensor")
    return tensor[::pair_width].contiguous()


def compute_oasiskv_logprobs(
    batch: Any, logits_output: Any, predict: torch.Tensor
) -> None:
    """Compute logprobs for the one committed token in each paired request.

    The caller has already compacted both ``next_token_logits`` and
    ``predict`` to B normal rows by the time this is called. EAGLE's usual
    ``accept_index`` still contains indices into the original 2B verify tensor
    (``[0, 2, ...]``), so using it here would either select another request's
    row or read past the compacted tensors.  A width-one chain describes the
    committed output layout directly and deliberately excludes every draft.
    """
    from sglang.srt.layers.logprob_processor import compute_spec_logprobs

    compute_spec_logprobs(batch, logits_output, predict, chain_stride=1)


def build_oasiskv_commit(
    normal_rows: torch.Tensor,
    batch_size: int,
    device: Any,
    *,
    buffers: Optional[OasisKVCommitBuffers] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed-width, root-only commit used by OasisKV.

    Keeping this separate from speculative acceptance is an important safety
    boundary: the lookahead row is useful for its indexer prediction only and
    must not become committed merely because its token happens to match.
    Production callers supply runner-owned constants. Shape/type checks use
    CPU metadata; GPU content checks are debug-only and never return a Python
    bool. Returning canonical roots also avoids trusting arbitrary row values
    for commit indices. LM-head row ownership is established by the paired
    batch constructors, before sampling.
    """
    if normal_rows.ndim != 1 or normal_rows.numel() != batch_size:
        raise ValueError("OasisKV requires exactly one normal row per request")
    if normal_rows.dtype != torch.int64:
        raise ValueError("OasisKV normal rows must be int64 indices")
    requested_device = torch.device(device)
    if normal_rows.device.type != requested_device.type or (
        requested_device.index is not None
        and normal_rows.device.index != requested_device.index
    ):
        raise ValueError("OasisKV normal rows and commit must use the same device")
    if buffers is None:
        # Standalone/reference callers; serving allocates once at pool setup.
        buffers = OasisKVCommitBuffers(batch_size, normal_rows.device)
    if buffers.normal_rows.device != normal_rows.device:
        raise ValueError("OasisKV commit buffers and rows must use the same device")
    accept_lens, expected = buffers.views(batch_size)
    message = "OasisKV normal rows must be request-major pair roots"
    if normal_rows.device.type == "cpu":
        if not torch.equal(normal_rows, expected):
            raise ValueError(message)
    elif envs.SGLANG_DEBUG_OASISKV_COMMIT.get():
        torch._assert_async((normal_rows == expected).all(), message)
    return accept_lens, expected.reshape(batch_size, 1)


def submit_oasiskv_pending_prefetches(forward_batch: Any) -> None:
    """Drain deferred selection/transfers after target attention and C4 commit."""
    pending = getattr(forward_batch, "_oasiskv_pending_prefetch", None)
    if not pending:
        return
    # Dict insertion order is target-layer execution order. Clear ownership
    # first so exception cleanup cannot submit any task twice.
    forward_batch._oasiskv_pending_prefetch = {}
    for coordinator, kwargs in pending.values():
        coordinator.submit_oasiskv_prefetch(**kwargs)


def plan_oasiskv_layer_prefetch(forward_batch: Any, layer_id: int) -> bool:
    """Start read-only selection after the indexer's real-query miss refill.

    Eager C4 transactions use post-commit residency. Captured plans restrict
    candidates to committed history and join before C4 promotion instead.
    """
    pending = getattr(forward_batch, "_oasiskv_pending_prefetch", None)
    if not pending or layer_id not in pending:
        return False
    coordinator, kwargs = pending[layer_id]
    if getattr(forward_batch, "is_oasiskv_graph_capture", False):
        if "task" not in kwargs:
            coordinator.oasiskv_graph_state.plan(
                forward_batch,
                layer_id=kwargs["layer_id"],
                predicted_c4_entries=kwargs["predicted_c4_entries"],
                compressed_seq_lens=kwargs["compressed_seq_lens"],
            )
            kwargs["task"] = "captured"
        return True
    if getattr(coordinator, "_active_dspark_window", None) is not None:
        return False
    if "task" not in kwargs:
        kwargs["task"] = coordinator.plan_oasiskv_prefetch(**kwargs)
    return True


def submit_oasiskv_layer_prefetch(forward_batch: Any, layer_id: int) -> bool:
    """Transfer one layer's prefetch as soon as its attention is done.

    Most decode steps do not complete a new C4 group and therefore have no
    transactional scratch mapping to commit.  On those steps, retaining every
    layer task until the end of the model destroys OasisKV's layer pipeline.
    Eager C4-boundary steps remain deferred until the root-only commit because their
    fixed destination slots must not race prefetch eviction or H2D writes.

    Captured selection/transfer run inside the graph, restrict candidates to
    authoritative history, and join before the root-only C4 commit.
    """
    pending = getattr(forward_batch, "_oasiskv_pending_prefetch", None)
    if not pending or layer_id not in pending:
        return False
    coordinator, kwargs = pending[layer_id]
    if getattr(forward_batch, "is_oasiskv_graph_capture", False):
        if kwargs.get("task") != "captured":
            raise RuntimeError("V4 graph transfer requires this layer's selection")
        coordinator.oasiskv_graph_state.transfer(
            forward_batch, layer_id=kwargs["layer_id"]
        )
        pending.pop(layer_id)
        return True
    if getattr(coordinator, "_active_dspark_window", None) is not None:
        return False
    pending.pop(layer_id)
    coordinator.submit_oasiskv_prefetch(**kwargs)
    return True


def build_oasiskv_paired_batch(
    normal_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    committed_lens: torch.Tensor,
    draft_valid: Optional[torch.Tensor] = None,
) -> OasisKVPairedBatch:
    """Interleave lanes without changing committed sequence lengths."""
    if normal_tokens.ndim != 1 or draft_tokens.shape != normal_tokens.shape:
        raise ValueError("normal and draft tokens must be equal-length 1-D tensors")
    if committed_lens.shape != normal_tokens.shape:
        raise ValueError("committed_lens must have one entry per request")
    bsz = normal_tokens.numel()
    normal_rows, draft_rows = build_oasiskv_pair_rows(bsz, normal_tokens.device)
    input_ids = torch.stack((normal_tokens, draft_tokens), dim=1).reshape(-1)
    # The normal token is the current committed position.  The draft is the
    # following position and is never reflected back into committed_lens.
    positions = torch.stack((committed_lens, committed_lens + 1), dim=1).reshape(-1)
    if draft_valid is None:
        draft_valid = torch.ones(bsz, dtype=torch.bool, device=normal_tokens.device)
    if draft_valid.shape != normal_tokens.shape:
        raise ValueError("draft_valid must have one entry per request")
    return OasisKVPairedBatch(
        input_ids, positions, normal_rows, draft_rows, draft_valid
    )


def paired_batch_from_verify(verify_input: Any, batch_size: int) -> OasisKVPairedBatch:
    """Adopt an EAGLE or DSPARK root+one-draft window without copying it."""
    if verify_input.draft_token_num != 2:
        raise ValueError("OasisKV requires exactly one normal and one draft token")
    if verify_input.draft_token.numel() != 2 * batch_size:
        raise ValueError("Verify tokens do not have request-major 2B layout")
    normal_rows, draft_rows = build_oasiskv_pair_rows(
        batch_size, verify_input.draft_token.device
    )
    return OasisKVPairedBatch(
        input_ids=verify_input.draft_token,
        positions=verify_input.positions,
        normal_rows=normal_rows,
        draft_rows=draft_rows,
        draft_valid=torch.ones(
            batch_size, dtype=torch.bool, device=verify_input.draft_token.device
        ),
    )


def configure_oasiskv_forward_batch(
    forward_batch: Any, paired: OasisKVPairedBatch
) -> None:
    """Install paired geometry on the real :class:`ForwardBatch`.

    Production decode arrives through EAGLE or DSPARK verify preparation and
    carries tree/position metadata in ``spec_info`` rather than the
    ``extend_prefix_lens``/``extend_start_loc`` fields used by a plain EXTEND.
    Either representation supplies the kernel-level ``history -> normal ->
    draft`` causal relation; no detached Python mask is used by production.
    """
    if forward_batch.batch_size != paired.batch_size:
        raise ValueError("OasisKV request count differs from ForwardBatch")
    if forward_batch.input_ids.numel() != 2 * paired.batch_size:
        raise ValueError("OasisKV ForwardBatch must contain two tokens per request")
    verify_width = getattr(
        getattr(forward_batch, "spec_info", None), "draft_token_num", None
    )
    extend_is_paired = (
        list(forward_batch.extend_seq_lens_cpu or ()) == [2] * paired.batch_size
    )
    if not extend_is_paired and verify_width != 2:
        raise ValueError("OasisKV paired forward requires two tokens per request")
    required_2b = {"out_cache_loc": forward_batch.out_cache_loc}
    if extend_is_paired:
        required_2b.update(
            extend_prefix_lens=getattr(forward_batch, "extend_prefix_lens", None),
            extend_start_loc=getattr(forward_batch, "extend_start_loc", None),
        )
    else:
        spec_info = forward_batch.spec_info
        required_2b.update(
            verify_draft_token=getattr(spec_info, "draft_token", None),
            verify_positions=getattr(spec_info, "positions", None),
        )
    missing = [name for name, value in required_2b.items() if value is None]
    if missing:
        raise ValueError(f"OasisKV ForwardBatch lacks metadata: {', '.join(missing)}")
    if forward_batch.out_cache_loc.numel() != 2 * paired.batch_size:
        raise ValueError("OasisKV requires one normal and one scratch cache location")
    if forward_batch.seq_lens is None or forward_batch.seq_lens_cpu is None:
        raise ValueError("OasisKV requires GPU and CPU sequence lengths")
    forward_batch.input_ids = paired.input_ids
    forward_batch.positions = paired.positions
    forward_batch.is_oasiskv_paired = True
    forward_batch.oasiskv_normal_rows = paired.normal_rows
    forward_batch.oasiskv_draft_rows = paired.draft_rows
    forward_batch.oasiskv_draft_valid = paired.draft_valid
