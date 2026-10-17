"""Pure CPU input and final-state diagnostics for the same-geometry W6 check."""

WIDTH, TOP_K, HOT, TAIL, SCRATCH = 6, 16, 128, 16, 13


def input_tokens(unique_misses):
    if unique_misses not in (8, 96):
        raise ValueError("Use the declared 8/96 unique-cold-miss pair")
    return [HOT + i % unique_misses for i in range(WIDTH * TOP_K)]


def decode_request(
    metadata,
    counters,
    policy,
    tokens,
    locations,
    request_id,
    request_slots,
    requested,
    missing_tokens,
):
    """Decode POST-commit state; never claim to observe an intermediate queue."""
    count = WIDTH * TOP_K
    epoch = counters[3 * request_slots + request_id] & 0x7FFFFFFF
    row = metadata[request_id + 1]

    def pair(i):
        return (row[2 * i] & 0xFFFFFFFF) | ((row[2 * i + 1] & 0xFFFFFFFF) << 32)

    keys = [pair(i) for i in range(count)]
    indices = [pair(count + i) for i in range(count)]
    compact = row[4 * count : 5 * count]
    current = {i: k & 0xFFFFFFFF for i, k in enumerate(keys) if k >> 32 == epoch}
    requested = set(requested)
    details = []
    physical = set(locations)
    for token in sorted(set(missing_tokens)):
        matches = []
        for slot, t in current.items():
            if t != token:
                continue
            raw = indices[slot]
            value = raw & 0xFFFFFFFF
            iepoch = raw >> 32
            if iepoch != epoch:
                status = "index_epoch_mismatch"
            elif value == 0xFFFFFFFF:
                status = "commit_marked_no_location"
            elif value in physical:
                status = "final_index_is_physical_location_but_output_missing"
            elif value < SCRATCH:
                status = "index_is_possible_scratch_ordinal"
            else:
                status = "unresolved_or_unclassified_index"
            matches.append(
                dict(
                    table_slot=slot,
                    index_epoch=iepoch,
                    index_low32=value,
                    compact_ordinals=[
                        i
                        for i, v in enumerate(compact[: counters[request_id]])
                        if (v & 8191) == slot
                    ],
                    status=status,
                )
            )
        details.append(
            dict(
                token=token,
                table_matches=matches,
                hot_slots=[i for i, t in enumerate(tokens[:HOT]) if t == token],
                classification="missing_from_final_union_table"
                if not matches
                else "see_table_matches",
            )
        )
    cursor_bits = (HOT - 1).bit_length()
    cache_epoch = (policy[0][request_id] & 0xFFFFFFFF) >> cursor_bits
    return dict(
        request_id=request_id,
        requested_unique=len(requested),
        committed_miss_count=counters[request_id],
        scratch_epoch=epoch,
        final_table_unique=len(set(current.values())),
        requested_missing_from_table=sorted(requested - set(current.values())),
        table_tokens_not_requested=sorted(set(current.values()) - requested),
        protected_hot_slots=sum(x == cache_epoch for x in policy[request_id + 1][:HOT]),
        missing=details,
    )
