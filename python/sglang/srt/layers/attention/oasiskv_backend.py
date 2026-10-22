"""Eager OasisKV backend for MHA, GQA, absorbed MLA and native DSA.

Native DSA delegates only Indexer metadata to SGLang's DSA backend. Its real
logical Top-K becomes the lookahead signal; all models share host ownership,
bounded working sets and the three-stream prefetch pipeline.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class OasisKVAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.req_pool = model_runner.req_to_token_pool
        self.pool = model_runner.token_to_kv_pool
        self.cache = self.pool.oasiskv_cache
        self.use_mha = False  # Native MLA/DSA always passes absorbed projections.
        self.dsa = None
        if self.cache.config.native_sparse:
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            self.dsa = DeepseekSparseAttnBackend(model_runner)
            self.dsa.use_fused_topk = False

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_idle():
            self.rows = []
            return
        if self.dsa is not None:
            self.dsa.init_forward_metadata(forward_batch)
            self.dsa.use_mha = False
        if getattr(forward_batch, "is_oasiskv_dummy", False):
            self.rows = []
            return
        slots = forward_batch.req_pool_indices_cpu
        if slots is None:
            raise ValueError("OasisKV requires scheduler-owned CPU request identities")
        slots = torch.as_tensor(slots, device="cpu").tolist()
        paired = getattr(forward_batch, "is_oasiskv_paired", False)
        if paired:
            prefixes = forward_batch.seq_lens_cpu.tolist()
            lengths = [2] * len(slots)
        elif forward_batch.forward_mode.is_decode():
            prefixes = (forward_batch.seq_lens_cpu - 1).tolist()
            lengths = [1] * len(slots)
        else:
            prefixes = list(forward_batch.extend_prefix_lens_cpu)
            lengths = list(forward_batch.extend_seq_lens_cpu)
        if not len(slots) == len(prefixes) == len(lengths):
            raise ValueError("OasisKV request/sequence metadata lengths disagree")
        self.rows = []
        start = 0
        for slot, prefix, length in zip(slots, prefixes, lengths):
            if prefix == 0 and not paired:
                self.cache.release(slot)
            generation = int(self.req_pool.req_generation[slot])
            self.rows.append((slot, generation, int(prefix), start, int(length)))
            start += int(length)

    def get_indexer_metadata(self, layer_id, forward_batch):
        if self.dsa is None:
            return None
        result = self.dsa.get_indexer_metadata(layer_id, forward_batch)
        return replace(result, force_unfused_topk=True)

    def q8kv8_born_fp8_q_eligible(self, *args, **kwargs):
        return False

    def get_cuda_graph_seq_len_fill_value(self):
        # Eager input buffers and autotuning also ask for this fill value.
        return 1

    def _dummy_attention(self, q, k, v, layer, batch_size):
        """Warm up model compute without appending fake scheduler requests.

        The runner may execute the same dummy batch repeatedly and gives it
        zero-filled slot IDs, without a CPU request identity or prior history.
        Use its current rows for attention and keep live cache ownership intact.
        """
        group = layer.tp_q_head_num // layer.tp_k_head_num
        q = q.reshape(batch_size, -1, layer.tp_q_head_num, q.shape[-1]).transpose(1, 2)
        k = (
            k.repeat_interleave(group, dim=1)
            .reshape(batch_size, -1, layer.tp_q_head_num, k.shape[-1])
            .transpose(1, 2)
        )
        v = (
            v.repeat_interleave(group, dim=1)
            .reshape(batch_size, -1, layer.tp_q_head_num, v.shape[-1])
            .transpose(1, 2)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=layer.scaling
        )
        return out.transpose(1, 2).reshape(-1, layer.tp_q_head_num * v.shape[-1])

    def _forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        topk_indices=None,
        **kwargs,
    ):
        if q.shape[0] == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))
        if not save_kv_cache or k is None or v is None:
            raise ValueError("OasisKV requires explicit per-layer KV writes")
        if layer.is_cross_attention:
            raise ValueError("OasisKV currently supports causal self-attention")
        attention_type = getattr(layer, "attn_type", None)
        if attention_type is not None and attention_type.value != "decoder":
            raise ValueError("OasisKV requires causal decoder attention")
        if self.cache.config.native_sparse and topk_indices is None:
            raise ValueError("Native DSA OasisKV requires logical Indexer Top-K")
        if getattr(layer, "sliding_window_size", -1) > 0:
            raise ValueError(
                "Generic OasisKV supports full attention; use the specialized path for hybrid SWA"
            )
        if any(
            kwargs.get(name) is not None
            for name in ("sinks", "score_mod", "aux_tensors")
        ):
            raise ValueError(
                "Generic OasisKV does not implement attention sinks or custom score modifiers"
            )
        if q_rope is not None:
            q = torch.cat((q.reshape(q.shape[0], layer.tp_q_head_num, -1), q_rope), -1)
        if k_rope is not None:
            k = torch.cat((k, k_rope), -1)
        q = q.reshape(q.shape[0], layer.tp_q_head_num, -1)
        k = k.reshape(k.shape[0], layer.tp_k_head_num, -1)
        v = v.reshape(v.shape[0], layer.tp_v_head_num, -1)
        if getattr(forward_batch, "is_oasiskv_dummy", False):
            return self._dummy_attention(q, k, v, layer, forward_batch.batch_size)
        paired = getattr(forward_batch, "is_oasiskv_paired", False)
        output = q.new_zeros((q.shape[0], layer.tp_q_head_num, v.shape[-1]))
        for slot, generation, prefix, start, length in self.rows:
            if not length:
                continue
            end = start + length
            keys, values = k[start:end], v[start:end]
            state = self.cache.get(slot, generation, layer.layer_id, keys, values)
            if paired:
                if length != 2:
                    raise ValueError(
                        "OasisKV target verify must contain a root/draft pair"
                    )
                state.append(keys[:1], values[:1], prefix)
                root_topk = None if topk_indices is None else topk_indices[start]
                draft_topk = None if topk_indices is None else topk_indices[start + 1]
                # Q/K already include RoPE (and MLA absorption). append() has
                # updated this step's committed summaries, including a block
                # completed by the real token. Prediction needs no attention
                # output and can start before even the cold-start seed.
                candidates = state.predict_prefetch(q[start + 1], draft_topk)
                state.seed(q[start], root_topk)
                plan = state.plan_prefetch(candidates)
                output[start:end] = state.attend(
                    q[start:end],
                    layer.scaling,
                    draft_key=keys[1],
                    draft_value=values[1],
                    logit_cap=layer.logit_cap,
                )
                # Publish the new mapping and overwrite slots only after the
                # current pair has finished reading the old working set.
                state.submit_prefetch(plan)
            else:
                state.append(keys, values, prefix)
                if forward_batch.forward_mode.is_decode():
                    selection = None if topk_indices is None else topk_indices[start]
                    state.seed(q[start], selection)
                    output[start:end] = state.attend(
                        q[start:end], layer.scaling, logit_cap=layer.logit_cap
                    )
                else:
                    selection = (
                        None if topk_indices is None else topk_indices[start:end]
                    )
                    output[start:end] = state.dense_prefill(
                        q[start:end],
                        prefix,
                        layer.scaling,
                        indices=selection,
                        logit_cap=layer.logit_cap,
                    )
        return output.reshape(q.shape[0], -1)

    forward_extend = _forward
    forward_decode = _forward
