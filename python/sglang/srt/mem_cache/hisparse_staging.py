"""Transient C4 KV storage, independent of the persistent hot-cache budget."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HiSparseStagingLayout:
    request_slots: int
    verify_width: int
    top_k: int
    sparse_layers: int
    prediction_steps: int = 0
    prefetch_slots: int | None = None
    page_size: int = 64

    def __post_init__(self):
        for name in ("request_slots", "verify_width", "top_k", "sparse_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.page_size != 64:
            raise ValueError("DeepSeek-V4 packed staging requires 64-row C4 pages")
        if type(self.prediction_steps) is not int or not 0 <= self.prediction_steps < self.verify_width:
            raise ValueError("prediction_steps must leave a nonempty verification window")
        if self.verify_capacity > 8192:
            raise ValueError("verification staging supports at most 8192 selections")
        if self.prefetch_slots is not None and (
            type(self.prefetch_slots) is not int or self.prefetch_slots <= 0
        ):
            raise ValueError("prefetch_slots must be a positive integer")
        if self.prediction_steps and self.slot_count < self.sparse_layers:
            raise ValueError(
                "Full-step Lagom prefetch requires buffer_slots >= the number "
                "of CSA layers; a smaller ring needs rolling prefetch scheduling"
            )

    @property
    def verify_capacity(self):
        return self.verify_width * self.top_k

    @property
    def forecast_capacity(self):
        return self.prediction_steps * self.top_k

    @property
    def slot_count(self):
        if not self.prediction_steps:
            return 0
        return self.sparse_layers if self.prefetch_slots is None else self.prefetch_slots

    def slot_for_layer(self, sparse_ordinal):
        if not self.slot_count or not 0 <= sparse_ordinal < self.sparse_layers:
            raise ValueError("expected a sparse-layer ordinal with prefetch enabled")
        return sparse_ordinal % self.slot_count

    def payload_bytes(self, bytes_per_page):
        def pages(rows):
            return (rows + self.page_size - 1) // self.page_size + 1
        return bytes_per_page * (
            pages(self.request_slots * self.verify_capacity)
            + self.slot_count * pages(self.request_slots * self.forecast_capacity)
        )

    def workspace_bytes(self):
        r, c = self.request_slots, self.verify_capacity
        h = 1 << (2 * c - 1).bit_length()
        shared = r * (2 * h * 4 + c * 20 + 24) + 4
        if not self.prediction_steps:
            return shared
        f = self.forecast_capacity
        fh = 1 << (2 * f - 1).bit_length()
        return shared + self.slot_count * r * (2 * fh * 4 + f * 8 + 4) + (
            self.sparse_layers * (r * (f * 4 + 8) + 4)
        )
