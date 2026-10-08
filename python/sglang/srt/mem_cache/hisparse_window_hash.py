from dataclasses import dataclass


@dataclass(frozen=True)
class HiSparseWindowHashLayout:
    """Packed persistent hash/CLOCK state from PR #40670, without KV scratch."""

    hot_slots: int
    verify_width: int
    top_k: int

    def __post_init__(self):
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.hot_slots, self.verify_width, self.top_k)
        ):
            raise ValueError("Window hash dimensions must be integers.")
        if not (
            1 < self.hot_slots <= 32768
            and self.verify_width > 0
            and self.top_k > 0
            and self.occurrences <= self.hot_slots
        ):
            raise ValueError(
                "Window hash requires 0 < verify_width * top_k <= "
                f"hot_slots <= 32768 and hot_slots > 1, got {self}."
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
