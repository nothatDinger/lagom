"""Explicit, restricted serving experiment; split remains the default."""


def validate_verify_backend(
    name,
    *,
    cuda,
    dsv4,
    speculative,
    width,
    top_k,
    hot,
    page_size,
    force_h2d,
    skip_io,
    prefetcher=None,
):
    if name not in ("split", "pr_fused"):
        raise ValueError(f"Unknown SGLANG_HISPARSE_VERIFY_IMPL={name!r}")
    if name == "split":
        return None
    if not (cuda and dsv4 and speculative and width > 0 and force_h2d and not skip_io):
        raise ValueError(
            "pr_fused experiment requires CUDA DSv4 speculative verify, force_h2d_each_step=true and real IO"
        )
    if prefetcher:
        raise ValueError("pr_fused experiment does not support a named prefetcher")
    if width * top_k > min(hot, 8192):
        raise ValueError(
            "pr_fused verify occurrences must fit hot slots and PR capacity"
        )
    # Canonical and unused E placeholders stay fixed; only the remaining tail
    # slots rotate with hot. They belong to the request's existing allocation,
    # not the independent DSPARK writer scratch pool.
    active_tail = 1 + (width + 3) // 4
    if active_tail >= page_size:
        raise ValueError("pr_fused needs positive request-owned tail scratch")
    return dict(
        active_slots=hot + active_tail,
        scratch_offset=hot + active_tail,
        scratch_slots=page_size - active_tail,
    )


def initialize_verify_hash_state(backend, **kwargs):
    if backend == "pr_fused":
        from sglang.benchmark.hisparse_pr40670_reference import (
            initialize_hisparse_spec_state,
        )

        import torch

        kwargs["req_pool_indices"] = kwargs["req_pool_indices"].to(torch.int64)
        initialize_hisparse_spec_state(**kwargs)
    else:
        from sglang.kernels.ops.kvcache.hisparse_window_hash import (
            initialize_window_hash_state,
        )

        initialize_window_hash_state(**kwargs)


def run_pr_fused_verify(
    coordinator,
    *,
    common,
    req_pool_indices,
    compressed_seq_lens,
    top_k_result,
    layer_id,
    verify_width,
    output_buffer,
    skip_io,
    miss_count,
):
    import torch

    num_reqs = req_pool_indices.numel()
    from sglang.benchmark.hisparse_pr40670_retry_candidate import (
        load_cache_to_device_buffer_spec_mla,
    )

    if skip_io or coordinator.skip_io:
        raise ValueError("pr_fused cannot skip KV IO")
    active = coordinator._pr_verify_layout["active_slots"]
    lengths = compressed_seq_lens.reshape(num_reqs, -1)
    if lengths.shape[1] == 1:
        lengths = lengths.expand(num_reqs, verify_width)
    load_cache_to_device_buffer_spec_mla(
        top_k_tokens=top_k_result.reshape(num_reqs, verify_width, coordinator.top_k),
        device_buffer_tokens=coordinator.req_device_buffer_tokens[layer_id, :, :active],
        device_buffer_locs=coordinator.req_device_buffer_token_locs[
            layer_id, :, :active
        ],
        host_cache_locs=common["host_cache_locs"],
        host_cache=common["host_cache"],
        device_buffer=common["device_buffer"],
        top_k_device_locs=output_buffer.view(num_reqs, verify_width, coordinator.top_k),
        req_pool_indices=req_pool_indices.to(torch.int64),
        seq_lens=lengths.to(torch.int32).contiguous().view(-1),
        state=coordinator._window_hash_states[layer_id],
        num_real_reqs=common["num_real_reqs"],
        transfer_kind="dsv4_paged",
        item_size_bytes=coordinator.item_size_bytes,
    )
    if getattr(coordinator, "_h2d_probe", None) is not None:
        # PR stores completed union counts by request slot in row 0.
        # Gather them in batch order only in the separate profile run.
        safe = req_pool_indices.to(torch.int64).clamp(
            0, coordinator.req_device_buffer_tokens.shape[1] - 1
        )
        torch.index_select(
            coordinator._window_hash_states[layer_id].scratch_state[0],
            0,
            safe,
            out=miss_count,
        )
        coordinator._h2d_probe.record(
            miss_count,
            common["num_real_reqs"],
            layer=layer_id,
            width=verify_width,
            item_bytes=coordinator.item_size_bytes,
            copy_blocks=0,
            skip_io=False,
        )
    return output_buffer
