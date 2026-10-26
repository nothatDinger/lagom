"""Runtime pool adapters for the generic OasisKV attention backend."""

import torch
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, KVCache
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache


class OasisKVPoolMixin:
    def init_oasiskv(self, config, req_to_token_pool):
        self.oasiskv_cache = OasisKVCache(
            config, self.device, max_requests=req_to_token_pool.size
        )
        req_to_token_pool.oasiskv_cache = self.oasiskv_cache

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("OasisKV KV writes must go through its attention backend")

    set_mla_kv_buffer = set_kv_buffer

    def move_kv_cache(self, tgt_loc, src_loc):
        # Target history is request/position-addressed. EAGLE slot compaction
        # does not move its payload, but the native DSA indexer is slot-addressed.
        if hasattr(self, "index_key_cache"):
            self.index_key_cache.move(tgt_loc, src_loc)


class OasisKVTokenToKVPool(OasisKVPoolMixin, KVCache):
    def __init__(
        self,
        size,
        page_size,
        dtype,
        layer_num,
        device,
        enable_memory_saver,
        start_layer=None,
        end_layer=None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

    def get_key_buffer(self, layer_id):
        raise RuntimeError(
            "OasisKV exposes bounded head-wise working sets, not dense GPU KV"
        )

    get_value_buffer = get_key_buffer
    get_kv_buffer = get_key_buffer

    def get_kv_size_bytes(self):
        return sum(
            pool.nbytes for pool in self.oasiskv_cache.page_pools.values()
        ) + sum(
            (0 if layer.key_min is None else layer.key_min.numel() * 8)
            for layer in self.oasiskv_cache.layers.values()
        )


class OasisKVDSATokenToKVPool(OasisKVPoolMixin, DSATokenToKVPool):
    def get_kv_size_bytes(self):
        # The native parent counts index keys and its placeholder latent buffer.
        # Add shared pages once, including free pages retained by the allocator.
        return super().get_kv_size_bytes() + (
            OasisKVTokenToKVPool.get_kv_size_bytes(self)
            if hasattr(self, "oasiskv_cache")
            else 0
        )

    def _create_buffers(self):
        # Keep the native index-key pool on GPU, without allocating the latent
        # full-KV pool. The inherited pointer table is never used for attention.
        self.kv_buffer = [
            torch.zeros(
                (1, 1, self.kv_cache_dim), dtype=self.store_dtype, device=self.device
            )
            for _ in range(self.layer_num)
        ]
