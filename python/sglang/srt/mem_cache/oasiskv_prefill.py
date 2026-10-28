"""Prefill attention over GPU prefix KV and the model's current K/V projections."""

import torch
import torch.nn.functional as F


def use_flash_prefill(query, key, value, indices):
    # Native DSA needs its per-query logical mask. Absorbed MLA and unusual
    # dimensions retain the reference path instead of assuming FA3 support.
    if (
        indices is not None
        or query.device.type != "cuda"
        or query.dtype not in (torch.float16, torch.bfloat16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or not query.shape[-1] == key.shape[-1] == value.shape[-1]
        or query.shape[-1] > 256
        or query.shape[-1] % 8
    ):
        return False
    from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

    return _is_fa3_supported()


def prefill_attention(
    query, key, value, prefix, scale, *, chunk_size, indices=None, logit_cap=0
):
    """K/V are token-major and contain exactly prefix + current query rows."""
    if key.shape[0] != prefix + query.shape[0] or value.shape[:2] != key.shape[:2]:
        raise ValueError("OasisKV prefill KV must contain the prefix and current chunk")
    if use_flash_prefill(query, key, value, indices):
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

        # FA3's bottom-right causal alignment includes the entire old prefix
        # for the first current query. A top-left SDPA is_causal mask would not.
        return flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=torch.tensor(
                [0, query.shape[0]], dtype=torch.int32, device=query.device
            ),
            cu_seqlens_k=torch.tensor(
                [0, key.shape[0]], dtype=torch.int32, device=query.device
            ),
            max_seqlen_q=query.shape[0],
            max_seqlen_k=key.shape[0],
            softmax_scale=scale,
            causal=True,
            softcap=logit_cap,
            ver=3,
        )

    heads, n = key.shape[1], key.shape[0]
    group = query.shape[1] // heads
    kd, vd = key.shape[-1], value.shape[-1]
    key = key.transpose(0, 1)[:, None].expand(-1, group, -1, -1)
    value = value.transpose(0, 1)[:, None].expand(-1, group, -1, -1)
    results = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        q = chunk.transpose(0, 1).reshape(heads, group, -1, kd)
        mask = torch.arange(n, device=query.device)[None] <= (
            prefix + start + torch.arange(chunk.shape[0], device=query.device)[:, None]
        )
        if indices is not None:
            selected = indices[start : start + chunk.shape[0]]
            valid = (selected >= 0) & (selected < n)
            selected_mask = torch.zeros_like(mask, dtype=torch.int32)
            selected_mask.scatter_add_(1, selected.clamp(0, n - 1).long(), valid.int())
            mask &= selected_mask > 0
        if logit_cap:
            logits = torch.matmul(q.float(), key.float().transpose(-1, -2)) * scale
            logits = logit_cap * torch.tanh(logits / logit_cap)
            weights = logits.masked_fill(~mask, -float("inf")).softmax(-1)
            out = torch.matmul(weights, value.float()).to(query.dtype)
        else:
            out = F.scaled_dot_product_attention(
                q, key, value, attn_mask=mask, scale=scale
            )
        results.append(out.reshape(query.shape[1], chunk.shape[0], vd).transpose(0, 1))
    return torch.cat(results, 0)
