from dataclasses import dataclass


@dataclass(frozen=True)
class HiSparseSpecLayout:
    """Persistent hash/CLOCK workspace; KV scratch reuses request-owned tail slots."""

    hot_slots: int
    verify_width: int
    top_k: int

    def __post_init__(self):
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.hot_slots, self.verify_width, self.top_k)
        ):
            raise ValueError("Fused HiSparse dimensions must be integers.")
        if not (
            1 < self.hot_slots <= 32768
            and self.verify_width > 0
            and self.top_k > 0
            and self.occurrences <= min(self.hot_slots, 8192)
        ):
            raise ValueError(
                "Fused HiSparse requires 0 < verify_width * top_k <= "
                f"min(hot_slots, 8192), with 1 < hot_slots <= 32768, got {self}."
            )

    @property
    def occurrences(self) -> int:
        return self.verify_width * self.top_k

    @property
    def hash_slots(self) -> int:
        return 1 << (2 * self.hot_slots - 1).bit_length()

    def policy_width(self, num_req_slots: int) -> int:
        return max(num_req_slots, self.hot_slots)

    def state_width(self, num_req_slots: int) -> int:
        width = max(4 * num_req_slots, 5 * self.occurrences)
        return width + width % 2

    def fixed_state_bytes(self, *, num_layers: int, num_req_slots: int) -> int:
        return num_layers * (
            num_req_slots * 2 * self.hash_slots * 8
            + (num_req_slots + 1) * self.policy_width(num_req_slots) * 4
            + (num_req_slots + 1) * self.state_width(num_req_slots) * 4
        )


def validate_pr_fused_config(
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
