"""Opt-in, eager, single-request V4/DSPARK quality experiment.

This is deliberately not an online prefetch implementation or a benchmark of
decode latency. Host snapshots and synchronous observations isolate shadow
forwards from the accepted trajectory. No hook is installed without the config
environment variable. See benchmark/oasiskv/trace/README.md.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from contextlib import contextmanager, nullcontext
from itertools import zip_longest
from pathlib import Path

import torch
from sglang.srt.speculative.oasiskv_trace_diagnostics import log_progress


def validate_trace_hisparse(runner):
    """Report actual runtime state, without changing the experiment's IO mode."""
    coordinator = getattr(runner, "hisparse_coordinator", None)

    def type_name(value):
        return (
            None
            if value is None
            else f"{type(value).__module__}.{type(value).__name__}"
        )

    prefetcher = getattr(coordinator, "prefetcher", None)
    allocator = getattr(runner, "token_to_kv_pool_allocator", None)
    state = {
        "coordinator_type": type_name(coordinator),
        "is_dsv4_hisparse": getattr(coordinator, "is_dsv4_hisparse", None),
        "prefetcher_type": type_name(prefetcher),
        "prefetcher_name": getattr(coordinator, "prefetcher_name", None),
        "enable_prefetch": getattr(coordinator, "enable_prefetch", None),
        "skip_io": getattr(coordinator, "skip_io", None),
        "runner_enable_hisparse": getattr(runner, "enable_hisparse", None),
        "runner_is_draft_worker": getattr(runner, "is_draft_worker", None),
        "runner_allocator_type": type_name(
            getattr(runner, "token_to_kv_pool_allocator", None)
        ),
        "coordinator_allocator_type": type_name(
            getattr(coordinator, "token_to_kv_pool_allocator", None)
        ),
        "full_pool_tokens": getattr(allocator, "size_full", None),
        "swa_pool_tokens": getattr(allocator, "size_swa", None),
        "c4_pool_units": getattr(allocator, "_size_hisparse", None),
    }
    failed = []
    if coordinator is None:
        failed.append("hisparse_coordinator is None")
    else:
        for name, expected in (
            ("is_dsv4_hisparse", True),
            ("enable_prefetch", False),
            ("skip_io", False),
        ):
            if state[name] != expected:
                failed.append(f"{name}={state[name]!r} (expected {expected!r})")
        if prefetcher is not None:
            failed.append(f"prefetcher is {state['prefetcher_type']} (expected None)")
    if failed:
        raise ValueError(
            "Trace requires V4 HiSparse with real IO and no online prefetcher; "
            f"failed: {'; '.join(failed)}; runtime: {json.dumps(state, sort_keys=True)}"
        )
    return state


class TensorSnapshot:
    """Restore existing allocations in place, including aliased tensor views."""

    def __init__(self, named_tensors, max_bytes):
        self.entries = []
        seen = set()
        selected = []
        for name, tensor in named_tensors:
            key = (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tensor.stride(),
                tensor.dtype,
            )
            if key not in seen and tensor.numel():
                seen.add(key)
                selected.append((name, tensor))
        self.nbytes = sum(t.numel() * t.element_size() for _, t in selected)
        if self.nbytes > max_bytes:
            raise RuntimeError(
                f"Trace snapshot needs {self.nbytes / 2**30:.2f} GiB per TP rank; "
                "increase snapshot_gib or reduce the KV pool allocation"
            )
        for name, tensor in selected:
            self.entries.append((name, tensor, tensor.detach().to("cpu", copy=True)))

    def restore(self):
        for _, destination, saved in self.entries:
            destination.copy_(saved)


def target_state_tensors(runner):
    """Explicit V4 state inventory; model weights are never copied.

    Verification writes SWA, C4/C128, index keys, *both* compression state
    families and HiSparse slot/LRU/scratch tables. C4 rollback alone is not a
    snapshot. Forward metadata is rebuilt by run_compact on each replay.
    """
    pool = runner.token_to_kv_pool
    if getattr(pool, "_unified_kv", True):
        raise ValueError("Trace snapshots currently require the paged V4 KV pool")
    for name in ("swa_kv_pool", "c4_kv_pool", "c128_kv_pool", "c4_indexer_kv_pool"):
        child = getattr(pool, name)
        buffers = (
            child.index_k_with_scale_buffer
            if name == "c4_indexer_kv_pool"
            else child.kv_buffer
        )
        for i, tensor in enumerate(buffers):
            yield f"{name}[{i}]", tensor
        for field, tensor in vars(child).items():
            if isinstance(tensor, torch.Tensor):
                yield f"{name}.{field}", tensor
    for name in ("compress_state_pools", "indexer_compress_state_pools"):
        for i, child in enumerate(getattr(pool, name)):
            if child is not None:
                if child.online:
                    raise ValueError(
                        "Offline trace requires online C128 compression off"
                    )
                yield f"{name}[{i}]", child.kv_score_buffer.kv_score
    # These objects contain mappings/counters, not model parameters. Host KV is
    # read-only during target verify; commit is executed only after all shadows.
    for name, obj in (
        ("pool", pool),
        ("requests", runner.req_to_token_pool),
        ("hisparse", runner.hisparse_coordinator),
    ):
        for field, tensor in vars(obj).items():
            if isinstance(tensor, torch.Tensor):
                yield f"{name}.{field}", tensor


def resident_lookup(raw, tokens, locations):
    """CPU observational helper: preserve query rank, never fetch a missing KV."""
    mapping = {
        token: loc for token, loc in zip(tokens, locations) if token >= 0 and loc >= 0
    }
    return [
        [mapping.get(token, -1) if token >= 0 else -1 for token in row] for row in raw
    ]


def check_live_locations(
    raw, selected, tokens, locations, history_units, *, buffer_size=None
):
    """Check physical references, including valid hot/reserved duplicates.

    A mismatched slot can be caused by eviction OR incomplete metadata. It does
    not by itself prove that the selected history union exceeds capacity.
    """
    live = {
        (token, loc) for token, loc in zip(tokens, locations) if token >= 0 and loc >= 0
    }
    for row, (ids, physical) in enumerate(zip(raw, selected)):
        for token, loc in zip(ids, physical):
            if 0 <= token < history_units and loc >= 0 and (token, loc) not in live:
                diagnostic = {
                    "row": row,
                    "token": token,
                    "returned_location": loc,
                    "live_locations_for_token": sorted(
                        place for item, place in live if item == token
                    ),
                    "tokens_at_returned_location": sorted(
                        item for item, place in live if place == loc
                    ),
                    "history_union_units": len(
                        {
                            item
                            for values in raw
                            for item in values
                            if 0 <= item < history_units
                        }
                    ),
                    "hot_buffer_size": buffer_size,
                    "verify_width": len(raw),
                }
                raise RuntimeError(
                    f"V4 verify row {row} references evicted C4 or inconsistent slot metadata for {token}: "
                    + json.dumps(diagnostic, sort_keys=True)
                    + "; inspect residency metadata before changing capacity"
                )


def compare_topk(left, right):
    """Classify rows without discarding repeated IDs, padding or missing rows."""
    report = {
        "rows_equal": 0,
        "rows_reordered_only": 0,
        "rows_different_ids": 0,
        "rows": [],
    }
    for row, (a, b) in enumerate(zip_longest(left, right)):
        ca, cb = Counter(a or []), Counter(b or [])
        if a == b:
            report["rows_equal"] += 1
            continue
        order_only = a is not None and b is not None and ca == cb
        report["rows_reordered_only" if order_only else "rows_different_ids"] += 1
        report["rows"].append(
            {
                "row": row,
                "reordered_only": order_only,
                "trial_row_present": a is not None,
                "replayed_row_present": b is not None,
                "trial_only_ids": sorted((ca - cb).elements()),
                "replayed_only_ids": sorted((cb - ca).elements()),
            }
        )
    return report


def compare_observations(trial, replay):
    """Compare KV membership, not the atomic scatter order of native Top-K.

    V4 topk_v1.cuh radix_topk() emits IDs with atomicAdd, without sorting.
    Sort within each query/miss event only. Preserve query order, multiplicity,
    padding and per-row fetch counts. In particular, different winners at a tied
    Top-K threshold are NOT equivalent. Never reorder tensors used by attention.
    The original arrays remain in the trace for diagnosis.
    """
    differences = []
    for layer in sorted(trial.keys() | replay.keys()):
        left, right = trial.get(layer), replay.get(layer)
        if left is None or right is None:
            differences.append({"layer": layer, "fields": ["layer_presence"]})
            continue
        fields = []
        if [sorted(row) for row in left["topk"]] != [
            sorted(row) for row in right["topk"]
        ]:
            fields.append("topk")
        if left["resident_after"] != right["resident_after"]:
            fields.append("resident_after")
        if [(x["row"], sorted(x["ids"])) for x in left["miss_events"]] != [
            (x["row"], sorted(x["ids"])) for x in right["miss_events"]
        ]:
            fields.append("miss_events")
        if fields:
            differences.append({"layer": layer, "fields": fields})
    return differences


def replay_diagnostics(trial_layers, layers, trial_logits, logits):
    differences = compare_observations(trial_layers, layers)
    changed_fields = {item["layer"]: item["fields"] for item in differences}
    finite = torch.isfinite(trial_logits) & torch.isfinite(logits)
    delta = (trial_logits.float() - logits.float()).abs()
    return {
        "trial_predictions": trial_logits.argmax(-1).tolist(),
        "replayed_predictions": logits.argmax(-1).tolist(),
        "differing_layers": [item["layer"] for item in differences],
        "layer_differences": differences,
        "rejection_layer_diffs": {
            layer: {
                "topk": compare_topk(
                    trial_layers[layer]["topk"], layers[layer]["topk"]
                ),
                "resident_after_equal": "resident_after"
                not in changed_fields.get(layer, []),
                "miss_events_equal": "miss_events" not in changed_fields.get(layer, []),
            }
            for layer in sorted(trial_layers.keys() & layers.keys())
            if layer in changed_fields
            or trial_layers[layer]["topk"] != layers[layer]["topk"]
        },
        "topk_order_only_layers": [
            layer
            for layer in sorted(trial_layers.keys() & layers.keys())
            if trial_layers[layer]["topk"] != layers[layer]["topk"]
            and [sorted(row) for row in trial_layers[layer]["topk"]]
            == [sorted(row) for row in layers[layer]["topk"]]
        ],
        # Equal argmax tokens do not prove equal logits or complete restoration.
        "logits_exact_equal": torch.equal(trial_logits, logits),
        "logits_max_abs_diff_finite": (
            float(delta[finite].max()) if finite.any() else None
        ),
        "trial_nonfinite_logits": int((~torch.isfinite(trial_logits)).sum()),
        "replayed_nonfinite_logits": int((~torch.isfinite(logits)).sum()),
    }


def greedy_commit_tokens(input_ids, predictions):
    """Independent CPU reference for the fixed-V, greedy acceptance boundary."""
    if len(input_ids) != len(predictions) or not input_ids:
        raise ValueError("Paired baseline needs V+1 inputs and predictions")
    accepted = 0
    for candidate, predicted in zip(input_ids[1:], predictions):
        if candidate != predicted:
            break
        accepted += 1
    return predictions[: accepted + 1]


@contextmanager
def replace_method(obj, name, method):
    # Removing the override restores descriptor binding, including on failure.
    had_instance_value = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, method)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


class ForwardObservation:
    def __init__(
        self, coordinator, real_width, mode, history_units, req_slot, workspace=None
    ):
        self.coordinator = coordinator
        self.real_width = real_width
        self.mode = mode
        self.history_units = history_units
        self.req_slot = req_slot
        self.layers = {}
        self.workspace = workspace

    @contextmanager
    def capture(self):
        coordinator = self.coordinator
        original = coordinator.swap_in_selected_pages_spec

        def swap(**kwargs):
            raw = kwargs["top_k_result"]
            layer = kwargs["layer_id"]
            if layer in self.layers or kwargs["req_pool_indices"].numel() != 1:
                raise RuntimeError("Trace requires one call per C4 layer and B=1")
            lengths_cpu = kwargs["verify_lens_cpu"]
            if lengths_cpu is None or len(lengths_cpu) != 1:
                raise RuntimeError("Trace requires explicit B=1 compact geometry")
            width = lengths_cpu[0]
            if not self.real_width <= width <= raw.shape[0]:
                raise RuntimeError("Invalid trace verify geometry")
            rows = raw[:width].detach().cpu().tolist()
            misses = []
            copied_tokens, copied_locations = [], []
            run_swap = coordinator._run_swap_in_kernel
            src = torch.empty(
                (1, coordinator.top_k), dtype=torch.int64, device=raw.device
            )
            dst = torch.empty_like(src, dtype=torch.int32)
            count = torch.zeros(1, dtype=torch.int32, device=raw.device)
            host = coordinator.req_to_host_pool[self.req_slot].cpu().tolist()
            host_to_logical = {loc: i for i, loc in enumerate(host) if loc >= 0}

            def record_swap(*args, **kw):
                kw.update(record_plan=True, miss_plan=(src, dst, count))
                result = run_swap(*args, **kw)
                size = int(count.item())
                ids = [host_to_logical[x] for x in src[0, :size].cpu().tolist()]
                if self.workspace is not None:
                    row = len(misses)
                    current_tokens, current_locs = self.residency(layer)
                    check_live_locations(
                        [rows[row]],
                        result.cpu().tolist(),
                        current_tokens,
                        current_locs,
                        self.history_units,
                    )
                    copied, locations = self.workspace.copy_row(
                        layer, row, raw[row : row + 1], result, self.history_units
                    )
                    copied_tokens.extend(copied)
                    copied_locations.extend(locations)
                misses.append({"row": len(misses), "ids": ids})
                return result

            with replace_method(coordinator, "_run_swap_in_kernel", record_swap):
                if self.mode == "resident":
                    # Only the real verification prefix may refill the cache.
                    real_kwargs = dict(kwargs)
                    real_kwargs["top_k_result"] = raw[: self.real_width]
                    real_kwargs["verify_lens_cpu"] = [self.real_width]
                    lengths = kwargs["compressed_seq_lens"].reshape(-1)
                    if lengths.numel() > 1:
                        real_kwargs["compressed_seq_lens"] = lengths[: self.real_width]
                    real_kwargs["output_buffer"] = None
                    real = original(**real_kwargs)
                    tokens, locs = self.residency(layer)
                    # Initial short-sequence slot tables can contain logical IDs
                    # for not-yet-written positions. Current-window C4 must be
                    # resolved by the indexer's transaction scratch lookup.
                    old_tokens = [
                        x if 0 <= x < self.history_units else -1 for x in tokens
                    ]
                    probes = resident_lookup(rows[self.real_width :], old_tokens, locs)
                    result = torch.full_like(raw, -1, dtype=torch.int32)
                    result[: self.real_width].copy_(real)
                    result[self.real_width : width].copy_(
                        torch.tensor(probes, dtype=torch.int32, device=raw.device)
                    )
                    misses.extend(
                        {"row": row, "ids": []} for row in range(self.real_width, width)
                    )
                else:
                    result = original(**kwargs)
            tokens, locs = self.residency(layer)
            check_live_locations(
                rows,
                result[:width].cpu().tolist(),
                tokens + copied_tokens,
                locs + copied_locations,
                self.history_units,
                buffer_size=getattr(coordinator, "device_buffer_size", None),
            )
            self.layers[layer] = {
                "topk": rows,
                "resident_after": sorted(
                    {
                        x
                        for x, loc in zip(tokens, locs)
                        if 0 <= x < self.history_units and loc >= 0
                    }
                ),
                "miss_events": misses,
                "workspace_d2d_bytes": (
                    len(copied_tokens) * coordinator.item_size_bytes
                    if copied_tokens
                    else 0
                ),
            }
            return result

        with replace_method(coordinator, "swap_in_selected_pages_spec", swap):
            yield self
        if not self.layers:
            raise RuntimeError("No V4 HiSparse Top-K observations were captured")

    def residency(self, layer):
        c = self.coordinator
        return (
            c.req_device_buffer_tokens[layer, self.req_slot].cpu().tolist(),
            c.req_device_buffer_token_locs[layer, self.req_slot].cpu().tolist(),
        )


class OasisKVTrace:
    @classmethod
    def from_environment(cls, worker):
        path = os.environ.get("SGLANG_OASISKV_TRACE_CONFIG")
        return None if not path else cls(worker, json.loads(Path(path).read_text()))

    def __init__(self, worker, config):
        from sglang.srt.model_executor.cuda_graph_config import (
            cuda_graph_fully_disabled,
        )
        from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        self.worker = worker
        self.config = config
        self.v = config["verify_candidates"]
        self.probes = config["probes"]
        self.max_windows = config["max_windows"]
        self.shadows = config.get("shadows", True)
        workspace_mode = config.get("probe_workspace", "none")
        if workspace_mode not in ("none", "row-copy"):
            raise ValueError("Trace probe_workspace must be none or row-copy")
        if config.get("proposal_gamma", worker.gamma) != worker.gamma:
            raise ValueError("Trace proposal_gamma must match DSPARK runtime gamma")
        topk_order = config.get("topk_order", "native")
        if topk_order not in ("native", "logical", "logical-all"):
            raise ValueError("Trace topk_order must be native, logical or logical-all")
        if type(self.v) is not int or not 1 <= self.v <= worker.gamma:
            raise ValueError("Trace V must be in [1, runtime proposal gamma]")
        if any(
            type(p) is not int or not 1 <= p <= worker.gamma - self.v
            for p in self.probes
        ):
            raise ValueError("Trace requires 1 <= P <= runtime proposal gamma - V")
        if len(set(self.probes)) != len(self.probes) or self.max_windows < 2:
            raise ValueError("Use unique P values and at least two trace windows")
        if (
            worker._oasiskv_lookahead
            or not cuda_graph_fully_disabled()
            or not get_schedule().disable_overlap_schedule
            or get_parallel().enable_dp_attention
            or get_parallel().pp_size != 1
            or worker._simulate_acc_len > 0
            or get_exec().moe.enable_eplb
            or read_ragged_verify_mode() is not RaggedVerifyMode.COMPACT
        ):
            raise ValueError(
                "Trace requires ordinary DSPARK, eager, compact metadata, no overlap scheduler/EPLB, pure TP, PP=1, real acceptance"
            )
        self.runner = worker.model_runner
        self.coordinator = self.runner.hisparse_coordinator
        hisparse_state = validate_trace_hisparse(self.runner)
        # Validate the state inventory at startup, before the first request.
        list(target_state_tensors(self.runner))
        self.coordinator.oasiskv_trace_topk_order = topk_order
        self.writer = None
        if worker.ps.tp_rank == 0:
            directory = Path(config["output"])
            directory.mkdir(parents=True, exist_ok=True)
            self.writer = directory / "trace.jsonl.gz"
            with gzip.open(self.writer, "xt", encoding="utf-8"):
                pass
            (directory / "runtime.json").write_text(
                json.dumps(
                    {
                        **config,
                        "gamma": worker.gamma,
                        "hisparse_runtime": hisparse_state,
                        "tp": get_parallel().tp_size,
                        "c4_bytes": self.coordinator.item_size_bytes,
                        "effective_top_k": self.coordinator.top_k,
                        "device_buffer_size": self.coordinator.device_buffer_size,
                        "c4_layer_to_model_layer": {
                            item.compress_layer_id: layer
                            for layer, item in enumerate(
                                self.runner.token_to_kv_pool.layer_mapping
                            )
                            if item is not None and item.compress_ratio == 4
                        },
                        "timing_valid": False,
                    },
                    indent=2,
                )
            )
        self.windows = {}
        self.pending = None

    def layout(self, width):
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[width], device=self.worker.device, grid=[width]
        )

    def run(self, *, batch, draft_block_ids, draft_tokens, sampling_info):
        self.pending = None
        self.pending_baseline = None
        log_progress(
            "trace-enter", batch.reqs, shadows=self.shadows, probes=self.probes
        )
        if len(batch.reqs) != 1 or batch.has_grammar or batch.return_logprob:
            raise ValueError("Trace requires B=1, no grammar and no logprob request")
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise ValueError("First trace experiment requires temperature=0")
        slot = int(batch.req_pool_indices_cpu[0])
        generation = int(self.runner.req_to_token_pool.req_generation[slot])
        request_id = str(batch.reqs[0].rid)
        executor = self.worker._verify_executor

        def forward(width):
            return executor.run_compact(
                batch=batch,
                layout=self.layout(width),
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=1,
                device=self.worker.device,
                sampling_info=sampling_info,
                inject_gate=False,
            )

        request_ids = self.config.get("request_ids")
        if request_ids is not None and request_id not in request_ids:
            # Server warmup is not experiment data and must not pay for
            # snapshots/shadow replays or appear in coverage/acceptance reports.
            self.pending = None
            return forward(self.v + 1)

        key = (request_id, generation)
        window = self.windows.get(key, 0)
        self.windows[key] = window + 1
        if window >= self.max_windows:
            self.pending = None
            return forward(self.v + 1)

        prefix = int(batch.seq_lens_cpu[0])
        history_units = prefix // 4
        inputs = (
            torch.cat((draft_block_ids[:, :1], draft_tokens), dim=1)[0].cpu().tolist()
        )
        common = {
            "schema_version": 1,
            "run_id": self.config["run_id"],
            "request_id": request_id,
            "generation": generation,
            "window_id": window,
            "snapshot_id": f"{request_id}:{generation}:{window}:{prefix}",
            "verify_candidates": self.v,
            "unit": "c4",
            "unit_tokens": 4,
            "bytes_per_unit": self.coordinator.item_size_bytes,
            "head": 0,
            "target_forward_ms": None,
            "topk_order": self.config.get("topk_order", "native"),
            "checkpoint_gamma": self.config.get("checkpoint_gamma"),
            "proposal_gamma": self.config.get("proposal_gamma"),
            "probe_workspace": self.config.get("probe_workspace", "none"),
        }

        def observe(mode, p):
            from sglang.srt.speculative.oasiskv_trace_workspace import ShadowKVWorkspace

            lease = (
                ShadowKVWorkspace(self.coordinator, self.v + 1 + p)
                if mode == "exact"
                and p
                and self.config.get("probe_workspace") == "row-copy"
                else nullcontext(None)
            )
            with lease as workspace:
                observer = ForwardObservation(
                    self.coordinator, self.v + 1, mode, history_units, slot, workspace
                )
                with observer.capture():
                    log_progress(
                        "trace-forward-enter", batch.reqs, mode=mode, p=p, window=window
                    )
                    result = forward(self.v + 1 + p)
                    log_progress(
                        "trace-forward-returned",
                        batch.reqs,
                        mode=mode,
                        p=p,
                        window=window,
                    )
            return result, observer.layers

        # Wait for outstanding prefill/commit writes before taking a complete
        # pre-forward image. Only one image is retained, in host memory.
        snapshot = None
        records = []
        if self.shadows:
            log_progress("snapshot-sync-enter", batch.reqs)
            torch.cuda.synchronize()
            log_progress("snapshot-copy-enter", batch.reqs)
            snapshot = TensorSnapshot(
                target_state_tensors(self.runner),
                int(self.config["snapshot_gib"] * 2**30),
            )
            log_progress(
                "snapshot-copy-returned", batch.reqs, bytes_per_rank=snapshot.nbytes
            )
            if self.writer is not None and window == 0:
                (self.writer.parent / "snapshot-inventory.json").write_text(
                    json.dumps(
                        {
                            "bytes_per_rank": snapshot.nbytes,
                            "tensors": [
                                {
                                    "name": name,
                                    "shape": list(tensor.shape),
                                    "dtype": str(tensor.dtype),
                                    "bytes": tensor.numel() * tensor.element_size(),
                                }
                                for name, tensor, _ in snapshot.entries
                            ],
                        },
                        indent=2,
                    )
                )
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state()

            def restore():
                torch.cuda.synchronize()
                snapshot.restore()
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)

            try:
                trial, trial_layers = observe("exact", 0)
                trial_logits = (
                    trial[0]
                    .logits_output.next_token_logits[: self.v + 1]
                    .detach()
                    .to("cpu", copy=True)
                )
                trial_hidden = trial[1][: self.v + 1].detach().to("cpu", copy=True)
                del trial
                for p in self.probes:
                    for mode in ("exact", "resident"):
                        restore()
                        shadow, layers = observe(mode, p)
                        del shadow
                        if layers.keys() != trial_layers.keys():
                            raise RuntimeError(
                                "Shadow and baseline C4 layer sets differ"
                            )
                        for layer, data in layers.items():
                            records.append(
                                dict(
                                    common,
                                    kind="probe",
                                    layer=layer,
                                    p=p,
                                    mode=mode,
                                    input_ids=inputs[: self.v + 1 + p],
                                    topk=data["topk"][self.v + 1 :],
                                    probe_demand_h2d_bytes=sum(
                                        len(x["ids"])
                                        for x in data["miss_events"][self.v + 1 :]
                                    )
                                    * self.coordinator.item_size_bytes,
                                    probe_workspace_d2d_bytes=data[
                                        "workspace_d2d_bytes"
                                    ],
                                )
                            )
            finally:
                restore()
            del snapshot

        result, layers = observe("exact", 0)
        if self.shadows:
            logits = (
                result[0]
                .logits_output.next_token_logits[: self.v + 1]
                .detach()
                .to("cpu", copy=True)
            )
            diagnostic = replay_diagnostics(trial_layers, layers, trial_logits, logits)
            hidden = result[1][: self.v + 1].detach().to("cpu", copy=True)
            diagnostic.update(
                paired_audit_version=1,
                input_ids=inputs[: self.v + 1],
                layer_ids=sorted(layers),
                hidden_states_exact_equal=torch.equal(trial_hidden, hidden),
            )
            diagnostic["passed"] = (
                diagnostic["trial_predictions"] == diagnostic["replayed_predictions"]
                and not diagnostic["differing_layers"]
                and diagnostic["logits_exact_equal"]
                and diagnostic["hidden_states_exact_equal"]
            )
            if self.writer is not None:
                with (self.writer.parent / "replay-audit.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(common, **diagnostic)) + "\n")
            if not diagnostic["passed"]:
                self.write_rejection(
                    dict(
                        common,
                        prefix_len=prefix,
                        history_units=history_units,
                        input_ids=inputs[: self.v + 1],
                    ),
                    records,
                    trial_layers,
                    layers,
                    diagnostic,
                )
                raise RuntimeError(
                    "Baseline replay changed after shadow restoration; trace rejected: "
                    + json.dumps(diagnostic, sort_keys=True)
                )
            # Preserve the pre-shadow control, rather than trying to recreate
            # this proposal and prefix in an independent server process.
            expected_tokens = greedy_commit_tokens(
                inputs[: self.v + 1], diagnostic["trial_predictions"]
            )
            self.pending_baseline = [
                dict(
                    common,
                    kind="window",
                    layer=layer,
                    prefix_len=prefix,
                    prompt_len=len(batch.reqs[0].origin_input_ids),
                    history_units=history_units,
                    input_ids=inputs[: self.v + 1],
                    accepted_drafts=len(expected_tokens) - 1,
                    commit_len=len(expected_tokens),
                    committed_output_ids=expected_tokens,
                    **data,
                )
                for layer, data in trial_layers.items()
            ]
        for layer, data in layers.items():
            records.append(
                dict(
                    common,
                    kind="window",
                    layer=layer,
                    prefix_len=prefix,
                    prompt_len=len(batch.reqs[0].origin_input_ids),
                    history_units=history_units,
                    input_ids=inputs[: self.v + 1],
                    **data,
                )
            )
        self.pending = records
        return result

    def write_rejection(self, common, probes, trial, replay, diagnostic):
        """Close diagnostic files before raising; never admit rejected samples."""
        if self.writer is None:
            return
        directory = self.writer.parent
        (directory / "rejection.json").write_text(
            json.dumps(dict(common, **diagnostic), indent=2) + "\n"
        )
        with gzip.open(directory / "rejected-trace.jsonl.gz", "at") as stream:
            for record in probes:
                stream.write(json.dumps(dict(record, kind="rejected_probe")) + "\n")
            for phase, observations in (("trial", trial), ("replay", replay)):
                for layer, data in observations.items():
                    record = dict(
                        common,
                        kind="rejected_window",
                        phase=phase,
                        layer=layer,
                        **data,
                    )
                    stream.write(json.dumps(record) + "\n")

    def finish(self, accept):
        if self.pending is None:
            return
        committed = int(accept.commit_lens[0].item())
        if not 1 <= committed <= self.v + 1:
            raise RuntimeError(
                "Probe escaped the fixed verification acceptance boundary"
            )
        tokens = accept.out_tokens.reshape(1, -1)[0, :committed].cpu().tolist()
        baseline = getattr(self, "pending_baseline", None)
        if baseline is not None and tokens != baseline[0]["committed_output_ids"]:
            error = "Actual acceptance differs from the pre-shadow greedy baseline"
            if self.writer is not None:
                (self.writer.parent / "rejection.json").write_text(
                    json.dumps({"passed": False, "error": error})
                )
            raise RuntimeError(error)
        for record in self.pending:
            if record["kind"] == "window":
                record.update(
                    accepted_drafts=committed - 1,
                    commit_len=committed,
                    committed_output_ids=tokens,
                )
        if self.writer is not None:
            if baseline is not None:
                with gzip.open(
                    self.writer.parent / "paired-baseline.jsonl.gz", "at"
                ) as stream:
                    for record in baseline:
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            # Close every gzip member so a normally terminated server leaves a
            # readable trace even though worker shutdown does not run __del__.
            with gzip.open(self.writer, "at", encoding="utf-8") as stream:
                for record in self.pending:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.pending = None
        self.pending_baseline = None
