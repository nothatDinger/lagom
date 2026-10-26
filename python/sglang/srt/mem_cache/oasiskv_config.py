"""Configuration shared by the generic OasisKV backend and its CPU tests."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class OasisKVConfig:
    # Historical tokens per KV head; the active block is retained separately.
    token_budget: int = 2048
    block_size: int = 16
    # Background admission only: None fetches every predicted miss, zero skips
    # predicted misses. Real-query fallback always refills all required blocks.
    max_new_blocks: int | None = None
    prefill_chunk_size: int = 256
    native_sparse: bool = False
    # Generic full-graph UVA copy implementation. Keep legacy for server A/B.
    copy_kernel: str = "tiled"
    refill_ctas_per_sm: int = 8
    prefetch_ctas_per_sm: int = 1

    def __post_init__(self):
        if self.copy_kernel not in ("tiled", "legacy"):
            raise ValueError("OasisKV copy_kernel must be 'tiled' or 'legacy'")
        for name in (
            "token_budget",
            "block_size",
            "prefill_chunk_size",
            "refill_ctas_per_sm",
            "prefetch_ctas_per_sm",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"OasisKV {name} must be a positive integer")
        if self.token_budget % self.block_size:
            raise ValueError("OasisKV token_budget must be divisible by block_size")
        if self.native_sparse and self.block_size != 1:
            raise ValueError("OasisKV native DSA requires block_size=1")
        if self.max_new_blocks is not None and (
            type(self.max_new_blocks) is not int or self.max_new_blocks < 0
        ):
            raise ValueError("OasisKV max_new_blocks must be a non-negative integer")

    @property
    def history_blocks(self):
        return self.token_budget // self.block_size

    @classmethod
    def from_json(cls, value, *, native_sparse=False, native_top_k=None):
        config = json.loads(value or "{}")
        if not isinstance(config, dict):
            raise ValueError("OasisKV configuration must be a JSON object")
        unknown = set(config) - {
            "top_k",
            "block_size",
            "max_new_blocks",
            "prefill_chunk_size",
            "prefetcher",
            "copy_kernel",
            "refill_ctas_per_sm",
            "prefetch_ctas_per_sm",
        }
        if unknown:
            raise ValueError(
                "Unknown generic OasisKV configuration fields: "
                + ", ".join(sorted(unknown))
            )
        if config.get("prefetcher", "oasiskv") != "oasiskv":
            raise ValueError("The OasisKV backend requires its lookahead prefetcher")
        # Native DSA selects latent entries, not Quest blocks. Preserve its K.
        block_size = config.get("block_size", 1 if native_sparse else 16)
        if native_sparse and block_size != 1:
            raise ValueError("OasisKV native DSA requires block_size=1")
        return cls(
            token_budget=config.get("top_k", native_top_k or 2048),
            block_size=block_size,
            max_new_blocks=config.get("max_new_blocks"),
            prefill_chunk_size=config.get("prefill_chunk_size", 256),
            native_sparse=native_sparse,
            copy_kernel=config.get("copy_kernel", "tiled"),
            refill_ctas_per_sm=config.get("refill_ctas_per_sm", 8),
            prefetch_ctas_per_sm=config.get("prefetch_ctas_per_sm", 1),
        )


def uses_oasiskv_backend(server_args):
    return getattr(server_args, "attention_backend", None) == "oasiskv"
