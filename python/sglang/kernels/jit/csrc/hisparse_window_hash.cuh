#pragma once

// Whole-window hash/CLOCK planner adapted from sgl-project/sglang PR #40670,
// commit 70bc9e9a643faa998ce529a38260feee83d3c0c6.
// KV I/O uses main's copy_cache_planned kernel. All misses go directly into
// unprotected hot slots; the DSpark writer scratch and canonical tail retain
// their existing physical mappings.

#include <sgl_kernel/tensor.h>  // TensorMatcher and symbolic tensor validation
#include <sgl_kernel/utils.h>   // RuntimeCheck and host utilities

#include <sgl_kernel/utils.cuh>  // LaunchKernel and PDL helpers

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>

namespace sglang {

struct MetadataOnlyKVTransferPolicy {
  __device__ static __forceinline__ void copy_warp(int32_t, const void*, void*, int64_t, int64_t) {}
};

#ifdef USE_ROCM
constexpr int WARP_SIZE = 64;
using BallotMask = uint64_t;
constexpr BallotMask FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFull;
#else
constexpr int WARP_SIZE = 32;
using BallotMask = unsigned int;
constexpr BallotMask FULL_WARP_MASK = 0xFFFFFFFFu;
#endif
constexpr int64_t HASH_DELETED = -2;
constexpr int32_t CLOCK_VICTIM_SAMPLES = 6;
constexpr uint32_t HASH_DEGRADED_FLAG = uint32_t{1} << 31;
constexpr uint32_t SCRATCH_EPOCH_MASK = HASH_DEGRADED_FLAG - 1;

struct SpecCacheState {
  int64_t* __restrict__ hash_primary;
  int64_t* __restrict__ hash_secondary;
  int32_t* __restrict__ ring_state;
  int32_t* __restrict__ ref_epochs;
  int64_t hash_stride;
  int64_t hash_size;
  int64_t ref_epoch_stride;
};

struct SpecMissWorkspace {
  int32_t* __restrict__ locs;
  int32_t* __restrict__ metadata;
  int32_t* __restrict__ counters;
  int64_t loc_stride;
  // A sliced tail view can have a row stride larger than its usable width.
  int64_t loc_capacity;
  int64_t metadata_stride;
  int64_t counter_capacity;
  // Fixed by the backing state width, even when ragged/fallback windows shrink.
  int64_t table_capacity;
};

constexpr int32_t ceil_log2_constexpr(int32_t value) {
  int32_t bits = 0;
  int32_t capacity = 1;
  while (capacity < value) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}

template <int32_t HOT_BUFFER_SIZE>
struct PackedRingState {
  static constexpr int32_t CURSOR_BITS = ceil_log2_constexpr(HOT_BUFFER_SIZE);
  static constexpr int32_t EPOCH_BITS = 32 - CURSOR_BITS < 23 ? 32 - CURSOR_BITS : 23;
  static constexpr uint32_t CURSOR_MASK = (uint32_t{1} << CURSOR_BITS) - 1;
  static constexpr uint32_t EPOCH_MASK = (uint32_t{1} << EPOCH_BITS) - 1;

  static_assert(HOT_BUFFER_SIZE > 1, "speculative hot buffer must contain at least two slots.");
  static_assert(CURSOR_BITS <= 15, "speculative CLOCK cursor requires hot_buffer_size <= 32768.");

  __device__ static int32_t next_epoch(int32_t packed_state) {
    int32_t epoch = static_cast<int32_t>((static_cast<uint32_t>(packed_state) >> CURSOR_BITS) + 1) & EPOCH_MASK;
    return epoch == 0 ? 1 : epoch;
  }

  __device__ static int32_t cursor(int32_t packed_state) {
    return static_cast<int32_t>(static_cast<uint32_t>(packed_state) & CURSOR_MASK) % HOT_BUFFER_SIZE;
  }

  __device__ static int32_t pack(int32_t epoch, int32_t cursor) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(epoch) << CURSOR_BITS) | (static_cast<uint32_t>(cursor) & CURSOR_MASK));
  }
};

template <int32_t HOT_BUFFER_SIZE>
struct PackedRingEntry {
  static constexpr int64_t TOKEN_CAPACITY = int64_t{1} << 31;

  // Keep the token in the low word so the common lookup path can compare it
  // without a 64-bit shift; the slot is decoded only after a token match.
  __device__ static int64_t pack(int32_t token, int32_t slot) {
    return static_cast<int64_t>(
        (static_cast<uint64_t>(static_cast<uint32_t>(slot)) << 32) | static_cast<uint32_t>(token));
  }

  __device__ static int32_t token(int64_t packed) {
    return static_cast<int32_t>(static_cast<uint32_t>(packed));
  }

  __device__ static int32_t slot(int64_t packed) {
    return static_cast<int32_t>(static_cast<uint64_t>(packed) >> 32);
  }
};

__device__ __forceinline__ int64_t atomic_cas_i64(int64_t* address, int64_t expected, int64_t desired) {
  return static_cast<int64_t>(atomicCAS(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(expected),
      static_cast<unsigned long long>(desired)));
}

__device__ __forceinline__ int64_t atomic_exch_i64(int64_t* address, int64_t value) {
  return static_cast<int64_t>(
      atomicExch(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
}

// Knuth multiplicative hash for open-addressing table of size hash_size.
__device__ __forceinline__ int hash_slot(int32_t key, int hash_size) {
  const uint32_t size = static_cast<uint32_t>(hash_size);
  const uint32_t hash = static_cast<uint32_t>(key) * 2654435761u;
  return static_cast<int>((size & (size - 1)) == 0 ? hash & (size - 1) : hash % size);
}

__device__ __forceinline__ int next_hash_slot(int slot, int hash_size) {
  const uint32_t size = static_cast<uint32_t>(hash_size);
  const uint32_t next = static_cast<uint32_t>(slot + 1);
  return static_cast<int>((size & (size - 1)) == 0 ? next & (size - 1) : next % size);
}

__device__ __forceinline__ int ring_hash_slot(int32_t key, int64_t hash_size) {
  return static_cast<int>((static_cast<uint32_t>(key) * 2654435761u) & static_cast<uint32_t>(hash_size - 1));
}

__device__ __forceinline__ int ring_hash_slot_secondary(int32_t key, int64_t hash_size) {
  uint32_t hash = static_cast<uint32_t>(key);
  hash ^= hash >> 16;
  hash *= 0x7FEB352Du;
  hash ^= hash >> 15;
  hash *= 0x846CA68Bu;
  hash ^= hash >> 16;
  return static_cast<int>(hash & static_cast<uint32_t>(hash_size - 1));
}

__device__ __forceinline__ bool mark_cache_epoch(int32_t* __restrict__ cache_ref, int32_t cache_epoch) {
  return atomicExch(cache_ref, cache_epoch) != cache_epoch;
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ int32_t ring_hash_lookup(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ vals,
    int64_t hash_size,
    int32_t token,
    const int32_t* __restrict__ req_device_buffer_tokens,
    bool hash_degraded) {
  if (token < 0) {
    return -1;
  }
  using Entry = PackedRingEntry<HOT_BUFFER_SIZE>;
  const int64_t primary = keys[ring_hash_slot(token, hash_size)];
  if (primary >= 0 && Entry::token(primary) == token) {
    const int32_t slot = Entry::slot(primary);
    if (slot < HOT_BUFFER_SIZE && req_device_buffer_tokens[slot] == token) {
      return slot;
    }
  }
  const int64_t secondary = vals[ring_hash_slot_secondary(token, hash_size)];
  if (secondary >= 0 && Entry::token(secondary) == token) {
    const int32_t slot = Entry::slot(secondary);
    if (slot < HOT_BUFFER_SIZE && req_device_buffer_tokens[slot] == token) {
      return slot;
    }
  }
  if (hash_degraded) {
    for (int32_t slot = 0; slot < HOT_BUFFER_SIZE; ++slot) {
      if (req_device_buffer_tokens[slot] == token) {
        return slot;
      }
    }
  }
  return -1;
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ int32_t hot_cache_lookup(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ vals,
    int64_t hash_size,
    int32_t token,
    const int32_t* __restrict__ req_device_buffer_tokens,
    bool hash_degraded) {
  if (token >= 0 && token < HOT_BUFFER_SIZE && req_device_buffer_tokens[token] == token) {
    return token;
  }
  return ring_hash_lookup<HOT_BUFFER_SIZE>(keys, vals, hash_size, token, req_device_buffer_tokens, hash_degraded);
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ int32_t ring_hash_insert_atomic(
    int64_t* __restrict__ keys, int64_t* __restrict__ vals, int64_t hash_size, int32_t token, int32_t buf_slot) {
  if (token < 0) {
    return -1;
  }
  using Entry = PackedRingEntry<HOT_BUFFER_SIZE>;
  const int64_t packed = Entry::pack(token, buf_slot);
  const int32_t primary_slot = ring_hash_slot(token, hash_size);
  const int32_t secondary_slot = ring_hash_slot_secondary(token, hash_size);
  int64_t old = keys[primary_slot];
  if (old >= 0 && Entry::token(old) == token) {
    atomic_exch_i64(keys + primary_slot, packed);
    return primary_slot;
  }
  if (old < 0 && atomic_cas_i64(keys + primary_slot, old, packed) == old) {
    return primary_slot;
  }
  old = vals[secondary_slot];
  if (old >= 0 && Entry::token(old) == token) {
    atomic_exch_i64(vals + secondary_slot, packed);
    return secondary_slot;
  }
  if (old < 0 && atomic_cas_i64(vals + secondary_slot, old, packed) == old) {
    return secondary_slot;
  }

  int64_t current = packed;
  bool use_secondary = false;
  for (int32_t kick = 0; kick < 64; ++kick) {
    const int32_t current_token = Entry::token(current);
    int64_t* table = use_secondary ? vals : keys;
    const int32_t slot =
        use_secondary ? ring_hash_slot_secondary(current_token, hash_size) : ring_hash_slot(current_token, hash_size);
    const int64_t displaced = atomic_exch_i64(table + slot, current);
    if (displaced < 0 || Entry::token(displaced) == current_token) {
      return slot;
    }
    current = displaced;
    use_secondary = !use_secondary;
  }
  return -1;
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ void ring_hash_erase_atomic(
    int64_t* __restrict__ keys, int64_t* __restrict__ vals, int64_t hash_size, int32_t token, int32_t buf_slot) {
  if (token < 0) {
    return;
  }
  using Entry = PackedRingEntry<HOT_BUFFER_SIZE>;
  const int64_t expected = Entry::pack(token, buf_slot);
  const int32_t primary_slot = ring_hash_slot(token, hash_size);
  if (atomic_cas_i64(keys + primary_slot, expected, HASH_DELETED) == expected) {
    return;
  }
  const int32_t secondary_slot = ring_hash_slot_secondary(token, hash_size);
  atomic_cas_i64(vals + secondary_slot, expected, HASH_DELETED);
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ bool try_get_active_tail_device_loc(
    int32_t token_idx,
    int64_t seq_len,
    const int32_t* __restrict__ req_device_buffer_tokens,
    const int32_t* __restrict__ req_device_buffer_locs,
    int64_t active_tail_slots,
    int32_t* __restrict__ out_loc) {
  int64_t slot = -1;
  if (static_cast<int64_t>(token_idx) >= seq_len - active_tail_slots) {
    if (active_tail_slots == 4) {
      const int4 page_tokens = *reinterpret_cast<const int4*>(req_device_buffer_tokens + HOT_BUFFER_SIZE);
      if (page_tokens.x == token_idx) {
        slot = HOT_BUFFER_SIZE;
      } else if (page_tokens.y == token_idx) {
        slot = HOT_BUFFER_SIZE + 1;
      } else if (page_tokens.z == token_idx) {
        slot = HOT_BUFFER_SIZE + 2;
      } else if (page_tokens.w == token_idx) {
        slot = HOT_BUFFER_SIZE + 3;
      }
    } else {
      for (int64_t candidate_slot = HOT_BUFFER_SIZE; candidate_slot < HOT_BUFFER_SIZE + active_tail_slots;
           candidate_slot++) {
        if (req_device_buffer_tokens[candidate_slot] == token_idx) {
          slot = candidate_slot;
          break;
        }
      }
    }
  }

  if (slot < HOT_BUFFER_SIZE || slot >= HOT_BUFFER_SIZE + active_tail_slots) {
    return false;
  }
  const int32_t loc = req_device_buffer_locs[slot];
  if (loc < 0) {
    return false;
  }
  *out_loc = loc;
  return true;
}

// Flatten all speculative steps. Each lane resolves one occurrence and
// atomically claims a unique-miss ordinal; KV bytes are copied later.
template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    int NUM_STEPS,
    bool RecordMissPlan,
    typename KVTransferPolicy>
__global__ void load_cache_to_device_buffer_spec_gather_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    void* __restrict__ device_buffer_k,
    int32_t* __restrict__ top_k_device_locs,
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ seq_lens,
    SpecCacheState cache_state,
    SpecMissWorkspace miss_workspace,
    const int32_t* __restrict__ num_real_reqs,
    int64_t* __restrict__ miss_src_out,
    int32_t* __restrict__ miss_dst_out,
    int32_t* __restrict__ miss_count_out,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t plan_stride,
    int64_t active_tail_slots) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  int32_t* req_top_k_device_locs = top_k_device_locs + bid * top_k_device_locs_stride;
  if (bid >= num_real_reqs[0]) {
    if constexpr (RecordMissPlan) {
      if (blockIdx.y == 0 && tid == 0) {
        miss_count_out[bid] = 0;
      }
    }
    const int64_t i = static_cast<int64_t>(blockIdx.y) * BLOCK_SIZE + tid;
    if (i < total_occurrences) req_top_k_device_locs[i] = -1;
    return;
  }

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int64_t total_warps = static_cast<int64_t>(gridDim.y) * NUM_WARPS;
  const int64_t global_warp = static_cast<int64_t>(blockIdx.y) * NUM_WARPS + warp_id;
  const int64_t occ = static_cast<int64_t>(lane_id) * total_warps + global_warp;
  const int64_t rid = req_pool_indices[bid];
  const int64_t buffer_offset = rid * buffer_stride_0;

  int32_t* req_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  int32_t* req_device_buffer_locs = device_buffer_locs + buffer_offset;
  const int64_t* req_host_cache_locs = host_cache_locs + rid * host_stride;
  int64_t* req_ring_hash_keys = cache_state.hash_primary + rid * cache_state.hash_stride;
  int64_t* req_ring_hash_vals = cache_state.hash_secondary + rid * cache_state.hash_stride;
  int32_t* req_cache_ref_bits = cache_state.ref_epochs + rid * cache_state.ref_epoch_stride;
  int32_t* req_scratch_tokens = miss_workspace.metadata + rid * miss_workspace.metadata_stride;
  auto* req_scratch_table = reinterpret_cast<unsigned long long*>(req_scratch_tokens);
  auto* req_scratch_indices = req_scratch_table + miss_workspace.table_capacity;
  int32_t* req_compact_hash_positions = req_scratch_tokens + 4 * miss_workspace.table_capacity;
  int32_t* req_work_count = miss_workspace.counters + miss_workspace.counter_capacity + rid;
  int32_t* req_union_hit_count = miss_workspace.counters + 2 * miss_workspace.counter_capacity + rid;
  int32_t* req_scratch_generation = miss_workspace.counters + 3 * miss_workspace.counter_capacity + rid;
  using RingState = PackedRingState<HOT_BUFFER_SIZE>;
  const int32_t cache_epoch = RingState::next_epoch(cache_state.ring_state[rid]);
  const uint32_t packed_scratch_generation = static_cast<uint32_t>(*req_scratch_generation);
  const bool hash_degraded = (packed_scratch_generation & HASH_DEGRADED_FLAG) != 0;
  uint32_t next_scratch_epoch = (packed_scratch_generation & SCRATCH_EPOCH_MASK) + 1;
  next_scratch_epoch = next_scratch_epoch > SCRATCH_EPOCH_MASK ? 1 : next_scratch_epoch;
  const int32_t scratch_epoch = static_cast<int32_t>(next_scratch_epoch);
  const int32_t* req_top_k_tokens = top_k_tokens + bid * top_k_tokens_stride;

  int32_t token = -1;
  int32_t loc = -2;  // Invalid for this row, even if another row selects the token.
  int32_t cache_slot = -1;
  int64_t src_loc = -1;
  bool needs_cache_lookup = false;
  if (occ < total_occurrences) {
    const int64_t step = occ / NUM_TOP_K;
    token = req_top_k_tokens[occ];
    const int64_t seq_len = static_cast<int64_t>(seq_lens[bid * NUM_STEPS + step]);
    if (token >= 0 && token < seq_len) {
      loc = -1;
      int32_t direct_loc = -1;
      if (try_get_active_tail_device_loc<HOT_BUFFER_SIZE>(
              token, seq_len, req_device_buffer_tokens, req_device_buffer_locs, active_tail_slots, &direct_loc)) {
        loc = direct_loc;
      } else {
        needs_cache_lookup = true;
      }
    }
  }

  // The flattened occurrence mapping places the same top-k position from
  // different speculative steps in one warp. Stable hits only need one Ring lookup.
  int32_t lookup_owner_lane = lane_id;
  bool reuse_owner_loc = false;
  if (total_warps <= NUM_TOP_K && NUM_TOP_K % total_warps == 0) {
    const int32_t lanes_per_step = static_cast<int32_t>(NUM_TOP_K / total_warps);
    if (lanes_per_step > 0 && NUM_STEPS * lanes_per_step <= WARP_SIZE) {
      lookup_owner_lane = lane_id % lanes_per_step;
      const int32_t owner_token = __shfl_sync(FULL_WARP_MASK, token, lookup_owner_lane);
      const int32_t owner_needs_lookup = __shfl_sync(FULL_WARP_MASK, needs_cache_lookup ? 1 : 0, lookup_owner_lane);
      reuse_owner_loc =
          needs_cache_lookup && lane_id != lookup_owner_lane && owner_needs_lookup != 0 && owner_token == token;
    }
  }

  if (needs_cache_lookup && !reuse_owner_loc) {
    cache_slot = hot_cache_lookup<HOT_BUFFER_SIZE>(
        req_ring_hash_keys, req_ring_hash_vals, cache_state.hash_size, token, req_device_buffer_tokens, hash_degraded);
    if (cache_slot >= 0) {
      loc = req_device_buffer_locs[cache_slot];
    } else {
      src_loc = req_host_cache_locs[token];
      if (src_loc >= 0) {
        int32_t hash_pos = hash_slot(token, static_cast<int>(miss_workspace.table_capacity));
        const auto epoch_bits = static_cast<unsigned long long>(static_cast<uint32_t>(scratch_epoch)) << 32;
        const auto token_bits = static_cast<uint32_t>(token);
        const auto packed = epoch_bits | token_bits;
        for (int64_t attempt = 0; attempt < miss_workspace.table_capacity; ++attempt) {
          auto old = atomicCAS(req_scratch_table + hash_pos, 0ull, 0ull);
          if (static_cast<uint32_t>(old >> 32) != static_cast<uint32_t>(scratch_epoch)) {
            const auto claimed = atomicCAS(req_scratch_table + hash_pos, old, packed);
            if (claimed == old) {
              const int32_t unique_idx = atomicAdd(req_work_count, 1);
              req_compact_hash_positions[unique_idx] = hash_pos;
              __threadfence();
              atomicExch(req_scratch_indices + hash_pos, epoch_bits | static_cast<uint32_t>(unique_idx));
              break;
            }
            // A lost CAS retries this same slot; it did not probe a new slot.
            --attempt;
            continue;
          }
          if (static_cast<uint32_t>(old) == token_bits) {
            // The commit kernel runs after every gather CTA has finished, so
            // duplicate lanes need not spin on another lane's publication.
            break;
          }
          hash_pos = next_hash_slot(hash_pos, static_cast<int>(miss_workspace.table_capacity));
        }
      }
    }
  }

  const int32_t owner_loc = __shfl_sync(FULL_WARP_MASK, loc, lookup_owner_lane);
  const int32_t owner_cache_slot = __shfl_sync(FULL_WARP_MASK, cache_slot, lookup_owner_lane);
  if (reuse_owner_loc) {
    loc = owner_loc;
    cache_slot = owner_cache_slot;
  }
  if (cache_slot >= 0) {
    if (mark_cache_epoch(req_cache_ref_bits + cache_slot, cache_epoch)) {
      atomicAdd(req_union_hit_count, 1);
    }
  }
  if (occ < total_occurrences) {
    req_top_k_device_locs[occ] = loc;
  }
}

// Commit the flattened miss union. Unlike the source PR's optional scratch
// rotation/approximate admission, every selected token fits in main's hot set.
// Current-window hits and already admitted misses remain protected throughout.
template <
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    int NUM_STEPS,
    bool RecordMissPlan,
    typename KVTransferPolicy,
    bool USE_PDL>
__global__ void load_cache_to_device_buffer_spec_commit_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ top_k_device_locs,
    int32_t* __restrict__ device_buffer_tokens,
    int32_t* __restrict__ device_buffer_locs,
    const int64_t* __restrict__ host_cache_locs,
    const void* __restrict__ host_cache_k,
    void* __restrict__ device_buffer_k,
    const int64_t* __restrict__ req_pool_indices,
    SpecCacheState cache_state,
    SpecMissWorkspace miss_workspace,
    const int32_t* __restrict__ num_real_reqs,
    int64_t* __restrict__ miss_src_out,
    int32_t* __restrict__ miss_dst_out,
    int32_t* __restrict__ miss_count_out,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t plan_stride) {
  const int bid = blockIdx.x;
  if (bid >= num_real_reqs[0]) return;
  const int64_t rid = req_pool_indices[bid];
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  using RingState = PackedRingState<HOT_BUFFER_SIZE>;
  const int32_t packed_state = cache_state.ring_state[rid];
  const int32_t cache_epoch = RingState::next_epoch(packed_state);
  const int32_t clock_cursor = RingState::cursor(packed_state);
  int32_t* tokens = device_buffer_tokens + rid * buffer_stride_0;
  const int32_t* locs = device_buffer_locs + rid * buffer_stride_0;
  int64_t* primary = cache_state.hash_primary + rid * cache_state.hash_stride;
  int64_t* secondary = cache_state.hash_secondary + rid * cache_state.hash_stride;
  int32_t* refs = cache_state.ref_epochs + rid * cache_state.ref_epoch_stride;
  int32_t* metadata = miss_workspace.metadata + rid * miss_workspace.metadata_stride;
  const auto* table = reinterpret_cast<const unsigned long long*>(metadata);
  auto* indices = reinterpret_cast<unsigned long long*>(metadata) + miss_workspace.table_capacity;
  const int32_t* compact = metadata + 4 * miss_workspace.table_capacity;
  int32_t* work_count = miss_workspace.counters + miss_workspace.counter_capacity + rid;
  int32_t* hit_count = miss_workspace.counters + 2 * miss_workspace.counter_capacity + rid;
  int32_t* generation = miss_workspace.counters + 3 * miss_workspace.counter_capacity + rid;
  const uint32_t old_generation = static_cast<uint32_t>(*generation);
  uint32_t scratch_epoch = (old_generation & SCRATCH_EPOCH_MASK) + 1;
  scratch_epoch = scratch_epoch > SCRATCH_EPOCH_MASK ? 1 : scratch_epoch;
  const auto epoch_bits = static_cast<unsigned long long>(scratch_epoch) << 32;
  const int32_t count = *work_count;
  __shared__ int32_t hash_degraded;
  if (threadIdx.x == 0) hash_degraded = (old_generation & HASH_DEGRADED_FLAG) != 0;
  __syncthreads();

  for (int32_t base = 0; base < count; base += blockDim.x) {
    const int32_t ordinal = base + threadIdx.x;
    int32_t victim = -1;
    int32_t hash_pos = -1;
    int32_t token = -1;
    if (ordinal < count) {
      hash_pos = compact[ordinal];
      token = static_cast<int32_t>(static_cast<uint32_t>(table[hash_pos]));
      // PR #40670's disjoint CLOCK partitions: sample old unprotected entries,
      // then use a global atomic fallback when a partition is fully protected.
      const int32_t partition_size = (HOT_BUFFER_SIZE - 1 - ordinal) / count + 1;
      const int32_t samples = min(partition_size, CLOCK_VICTIM_SAMPLES);
      uint32_t oldest_age = 0;
      for (int32_t sample = 0; sample < samples; ++sample) {
        const int32_t linear = ordinal + (sample * partition_size / samples) * count;
        const int32_t candidate = (clock_cursor + linear) % HOT_BUFFER_SIZE;
        const int32_t ref = refs[candidate];
        if (ref == cache_epoch) continue;
        const uint32_t age = (static_cast<uint32_t>(cache_epoch) - static_cast<uint32_t>(ref)) & RingState::EPOCH_MASK;
        if (victim < 0 || age > oldest_age) {
          victim = candidate;
          oldest_age = age;
        }
      }
      if (victim >= 0) refs[victim] = cache_epoch;
    }
    // All partition claims must be visible before fallback claims start.
    __syncthreads();
    if (ordinal < count && victim < 0) {
      const uint32_t start =
          (static_cast<uint32_t>(clock_cursor) + static_cast<uint32_t>(ordinal) * 2654435761u) % HOT_BUFFER_SIZE;
      for (int32_t attempt = 0; attempt < HOT_BUFFER_SIZE; ++attempt) {
        const int32_t candidate = (start + attempt) % HOT_BUFFER_SIZE;
        const int32_t observed = refs[candidate];
        if (observed != cache_epoch && atomicCAS(refs + candidate, observed, cache_epoch) == observed) {
          victim = candidate;
          break;
        }
      }
    }
    if (ordinal < count) {
      const int32_t dst = victim >= 0 ? locs[victim] : -1;
      const int64_t src = host_cache_locs[rid * host_stride + token];
      if (dst >= 0 && src >= 0) {
        const int32_t old_token = tokens[victim];
        tokens[victim] = token;
        ring_hash_erase_atomic<HOT_BUFFER_SIZE>(primary, secondary, cache_state.hash_size, old_token, victim);
        if (ring_hash_insert_atomic<HOT_BUFFER_SIZE>(primary, secondary, cache_state.hash_size, token, victim) < 0) {
          atomicExch(&hash_degraded, 1);
        }
      }
      atomicExch(indices + hash_pos, epoch_bits | static_cast<uint32_t>(dst));
      miss_src_out[bid * plan_stride + ordinal] = src;
      miss_dst_out[bid * plan_stride + ordinal] = dst;
    }
    // A fallback may claim a later iteration's partition. Join every warp
    // before that iteration samples references, avoiding duplicate victims.
    __syncthreads();
  }

  int32_t* output = top_k_device_locs + bid * top_k_device_locs_stride;
  const int32_t* input = top_k_tokens + bid * top_k_tokens_stride;
  for (int64_t occ = threadIdx.x; occ < total_occurrences; occ += blockDim.x) {
    if (output[occ] == -2) {
      output[occ] = -1;
    } else if (output[occ] == -1 && input[occ] >= 0) {
      int32_t pos = hash_slot(input[occ], static_cast<int32_t>(miss_workspace.table_capacity));
      for (int64_t attempt = 0; attempt < miss_workspace.table_capacity; ++attempt) {
        const auto packed = table[pos];
        if (static_cast<uint32_t>(packed >> 32) != scratch_epoch) break;
        if (static_cast<uint32_t>(packed) == static_cast<uint32_t>(input[occ])) {
          output[occ] = static_cast<int32_t>(static_cast<uint32_t>(indices[pos]));
          break;
        }
        pos = next_hash_slot(pos, static_cast<int32_t>(miss_workspace.table_capacity));
      }
    }
  }
  // Wrapped epochs must not mistake ancient reference bits for current hits.
  if (cache_epoch == RingState::EPOCH_MASK) {
    for (int32_t slot = threadIdx.x; slot < HOT_BUFFER_SIZE; slot += blockDim.x)
      refs[slot] = 0;
  }
  __syncthreads();
  if (scratch_epoch == SCRATCH_EPOCH_MASK) {
    for (int64_t i = threadIdx.x; i < miss_workspace.metadata_stride; i += blockDim.x)
      metadata[i] = 0;
  }
  if (threadIdx.x == 0) {
    miss_count_out[bid] = count;
    *work_count = 0;
    *hit_count = 0;
    *generation = static_cast<int32_t>(
        (scratch_epoch == SCRATCH_EPOCH_MASK ? 0 : scratch_epoch) | (hash_degraded ? HASH_DEGRADED_FLAG : 0));
    cache_state.ring_state[rid] = RingState::pack(cache_epoch, (clock_cursor + count) % HOT_BUFFER_SIZE);
  }
}

template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    int NUM_STEPS,
    bool RecordMissPlan,
    typename KVTransferPolicy,
    bool USE_PDL>
void load_cache_to_device_buffer_spec(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView cache_index,
    tvm::ffi::TensorView cache_policy,
    tvm::ffi::TensorView scratch_locs,
    tvm::ffi::TensorView scratch_state,
    tvm::ffi::TensorView num_real_reqs,
    int64_t active_tail_slots,
    tvm::ffi::TensorView miss_src_out,
    tvm::ffi::TensorView miss_dst_out,
    tvm::ffi::TensorView miss_count_out) {
  using namespace host;

  static_assert(RecordMissPlan && !USE_PDL, "window hash only records metadata with ordered launches.");
  static_assert(NUM_STEPS > 0, "HiSparse speculative swap requires at least one step.");
  static_assert(NUM_TOP_K > 0, "HiSparse speculative swap requires a positive top_k.");
  static_assert(NUM_STEPS * NUM_TOP_K <= 32768, "window hash supports at most 32768 occurrences.");
  static_assert(NUM_STEPS * NUM_TOP_K <= HOT_BUFFER_SIZE, "the complete verify union must fit in the hot buffer.");

  const int64_t bs = top_k_tokens.shape()[0];
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  RuntimeCheck(top_k_tokens.ndim() == 3, "speculative top_k_tokens must have shape [batch, steps, top_k].");
  RuntimeCheck(top_k_device_locs.ndim() == 3, "speculative output must have shape [batch, steps, top_k].");
  RuntimeCheck(top_k_tokens.shape()[1] == NUM_STEPS, "top_k_tokens step dimension mismatch.");
  RuntimeCheck(top_k_tokens.shape()[2] == NUM_TOP_K, "top_k_tokens top-k dimension mismatch.");
  RuntimeCheck(
      cache_index.ndim() == 3 && cache_index.shape()[1] == 2,
      "speculative cache_index must have shape [num_requests, 2, hash_size].");
  const int64_t ring_hash_size = cache_index.shape()[2];
  RuntimeCheck(
      ring_hash_size > 0 && (ring_hash_size & (ring_hash_size - 1)) == 0, "ring hash capacity must be a power of two.");
  RuntimeCheck(scratch_locs.ndim() == 2, "speculative scratch_locs must have shape [num_requests, capacity].");
  const int64_t num_request_slots = scratch_locs.shape()[0];
  const int64_t scratch_capacity = scratch_locs.shape()[1];
  RuntimeCheck(scratch_capacity == 0, "window hash uses direct hot-slot admission, without cache scratch rotation.");
  RuntimeCheck(
      cache_index.shape()[0] >= num_request_slots,
      "speculative cache_index request capacity is smaller than scratch_locs.");
  RuntimeCheck(
      cache_policy.ndim() == 2 && cache_policy.shape()[0] >= num_request_slots + 1 &&
          cache_policy.shape()[1] >= HOT_BUFFER_SIZE,
      "speculative cache_policy must contain one CLOCK control row and one reference-epoch row per request.");
  RuntimeCheck(
      scratch_state.ndim() == 2 && scratch_state.shape()[0] >= num_request_slots + 1 &&
          scratch_state.shape()[1] >= 4 * num_request_slots && scratch_state.shape()[1] >= 5 * total_occurrences,
      "speculative scratch_state must contain one counter row and one miss-metadata row per request.");
  RuntimeCheck(scratch_state.strides()[0] % 2 == 0, "speculative scratch metadata stride must be 64-bit aligned.");

  const int64_t host_stride = host_cache_locs.shape()[1];
  RuntimeCheck(
      host_stride <= PackedRingEntry<HOT_BUFFER_SIZE>::TOKEN_CAPACITY,
      "speculative packed ring metadata supports sequence lengths up to ",
      PackedRingEntry<HOT_BUFFER_SIZE>::TOKEN_CAPACITY,
      ", got ",
      host_stride);
  const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
  const int64_t top_k_tokens_stride = top_k_tokens.strides()[0];
  const int64_t top_k_device_locs_stride = top_k_device_locs.strides()[0];
  const int64_t cache_index_stride_0 = cache_index.strides()[0];
  const int64_t cache_index_stride_1 = cache_index.strides()[1];
  const int64_t cache_policy_stride_0 = cache_policy.strides()[0];
  const int64_t scratch_stride_0 = scratch_locs.strides()[0];
  const int64_t scratch_state_stride_0 = scratch_state.strides()[0];
  int64_t* const miss_src_ptr = RecordMissPlan ? static_cast<int64_t*>(miss_src_out.data_ptr()) : nullptr;
  int32_t* const miss_dst_ptr = RecordMissPlan ? static_cast<int32_t*>(miss_dst_out.data_ptr()) : nullptr;
  int32_t* const miss_count_ptr = RecordMissPlan ? static_cast<int32_t*>(miss_count_out.data_ptr()) : nullptr;
  const int64_t plan_stride = RecordMissPlan ? miss_src_out.strides()[0] : 0;
  if constexpr (RecordMissPlan) {
    RuntimeCheck(
        miss_src_out.ndim() == 2 && miss_src_out.shape()[0] >= bs && miss_src_out.shape()[1] >= total_occurrences,
        "speculative miss_src must have shape [batch, >= steps * top_k].");
    RuntimeCheck(
        miss_dst_out.ndim() == 2 && miss_dst_out.shape()[0] >= bs && miss_dst_out.shape()[1] >= total_occurrences,
        "speculative miss_dst must have shape [batch, >= steps * top_k].");
    RuntimeCheck(
        miss_count_out.ndim() == 1 && miss_count_out.shape()[0] >= bs,
        "speculative miss_count must have shape [batch].");
    RuntimeCheck(miss_dst_out.strides()[0] == plan_stride, "speculative miss_src/miss_dst row strides differ.");
  }

  // Preserve the device kernels' independent restrict-qualified views while
  // exposing only four packed tensors through FFI. Row 0 is the control
  // plane; request-owned rows begin at row 1.
  auto* cache_index_ptr = static_cast<int64_t*>(cache_index.data_ptr());
  auto* cache_policy_ptr = static_cast<int32_t*>(cache_policy.data_ptr());
  auto* scratch_state_ptr = static_cast<int32_t*>(scratch_state.data_ptr());
  const SpecCacheState cache_state{
      cache_index_ptr,
      cache_index_ptr + cache_index_stride_1,
      cache_policy_ptr,
      cache_policy_ptr + cache_policy_stride_0,
      cache_index_stride_0,
      ring_hash_size,
      cache_policy_stride_0};
  const SpecMissWorkspace miss_workspace{
      static_cast<int32_t*>(scratch_locs.data_ptr()),
      scratch_state_ptr + scratch_state_stride_0,
      scratch_state_ptr,
      scratch_stride_0,
      scratch_capacity,
      scratch_state_stride_0,
      num_request_slots,
      scratch_state.shape()[1] / 5};
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({bs}).with_dtype<int64_t>().with_device(cuda_device).verify(req_pool_indices);
  TensorMatcher({bs * NUM_STEPS}).with_dtype<int32_t>().with_device(cuda_device).verify(seq_lens);
  const auto device = cuda_device.unwrap();

  const uint32_t tiles = static_cast<uint32_t>((total_occurrences + BLOCK_SIZE - 1) / BLOCK_SIZE);
  LaunchKernel(dim3(static_cast<uint32_t>(bs), tiles), BLOCK_SIZE, device)(
      load_cache_to_device_buffer_spec_gather_kernel<
          BLOCK_SIZE,
          NUM_TOP_K,
          HOT_BUFFER_SIZE,
          NUM_STEPS,
          RecordMissPlan,
          KVTransferPolicy>,
      static_cast<const int32_t*>(top_k_tokens.data_ptr()),
      static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<const int64_t*>(host_cache_locs.data_ptr()),
      static_cast<int32_t*>(device_buffer_locs.data_ptr()),
      host_cache_k.data_ptr(),
      device_buffer_k.data_ptr(),
      static_cast<int32_t*>(top_k_device_locs.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      cache_state,
      miss_workspace,
      static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      miss_src_ptr,
      miss_dst_ptr,
      miss_count_ptr,
      buffer_stride_0,
      host_stride,
      top_k_tokens_stride,
      top_k_device_locs_stride,
      plan_stride,
      active_tail_slots);

  LaunchKernel(dim3(static_cast<uint32_t>(bs)), 512, device)
      .enable_pdl(USE_PDL)(
          load_cache_to_device_buffer_spec_commit_kernel<
              NUM_TOP_K,
              HOT_BUFFER_SIZE,
              NUM_STEPS,
              RecordMissPlan,
              KVTransferPolicy,
              USE_PDL>,
          static_cast<const int32_t*>(top_k_tokens.data_ptr()),
          static_cast<int32_t*>(top_k_device_locs.data_ptr()),
          static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
          static_cast<int32_t*>(device_buffer_locs.data_ptr()),
          static_cast<const int64_t*>(host_cache_locs.data_ptr()),
          host_cache_k.data_ptr(),
          device_buffer_k.data_ptr(),
          static_cast<const int64_t*>(req_pool_indices.data_ptr()),
          cache_state,
          miss_workspace,
          static_cast<const int32_t*>(num_real_reqs.data_ptr()),
          miss_src_ptr,
          miss_dst_ptr,
          miss_count_ptr,
          top_k_tokens_stride,
          top_k_device_locs_stride,
          buffer_stride_0,
          host_stride,
          plan_stride);
}

template <int HOT_BUFFER_SIZE>
__global__ void initialize_hisparse_spec_state_kernel(
    const int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ req_pool_indices,
    int64_t* __restrict__ cache_index,
    int32_t* __restrict__ cache_policy,
    int32_t* __restrict__ scratch_state,
    int64_t buffer_stride,
    int64_t hash_row_stride,
    int64_t hash_bank_stride,
    int64_t hash_size,
    int64_t policy_row_stride,
    int64_t state_row_stride,
    int64_t num_request_slots) {
  const int64_t rid = req_pool_indices[blockIdx.x];
  int64_t* primary = cache_index + rid * hash_row_stride;
  int64_t* secondary = primary + hash_bank_stride;
  int32_t* ring_state = cache_policy;
  int32_t* ref_epochs = cache_policy + (rid + 1) * policy_row_stride;
  int32_t* counters = scratch_state;
  int32_t* metadata = scratch_state + (rid + 1) * state_row_stride;

  for (int64_t i = threadIdx.x; i < hash_size; i += blockDim.x) {
    primary[i] = -1;
    secondary[i] = -1;
  }
  for (int64_t i = threadIdx.x; i < HOT_BUFFER_SIZE; i += blockDim.x) {
    ref_epochs[i] = 0;
  }
  for (int64_t i = threadIdx.x; i < state_row_stride; i += blockDim.x) {
    metadata[i] = 0;
  }
  if (threadIdx.x == 0) {
    ring_state[rid] = 0;
    counters[rid] = 0;
    counters[num_request_slots + rid] = 0;
    counters[2 * num_request_slots + rid] = 0;
    counters[3 * num_request_slots + rid] = 0;
  }
  __syncthreads();

  const int32_t* tokens = device_buffer_tokens + rid * buffer_stride;
  for (int32_t slot = threadIdx.x; slot < HOT_BUFFER_SIZE; slot += blockDim.x) {
    const int32_t token = tokens[slot];
    if (token >= 0 && ring_hash_insert_atomic<HOT_BUFFER_SIZE>(primary, secondary, hash_size, token, slot) < 0) {
      atomicOr(counters + 3 * num_request_slots + rid, static_cast<int32_t>(HASH_DEGRADED_FLAG));
    }
  }
}

template <int HOT_BUFFER_SIZE>
void initialize_hisparse_spec_state(
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView cache_index,
    tvm::ffi::TensorView cache_policy,
    tvm::ffi::TensorView scratch_state) {
  using namespace host;

  const int64_t batch_size = req_pool_indices.shape()[0];
  const int64_t num_request_slots = cache_index.shape()[0];
  const int64_t hash_size = cache_index.shape()[2];
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({batch_size}).with_dtype<int64_t>().with_device(cuda_device).verify(req_pool_indices);
  const auto device = cuda_device.unwrap();

  LaunchKernel(static_cast<uint32_t>(batch_size), 256, device)(
      initialize_hisparse_spec_state_kernel<HOT_BUFFER_SIZE>,
      static_cast<const int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<int64_t*>(cache_index.data_ptr()),
      static_cast<int32_t*>(cache_policy.data_ptr()),
      static_cast<int32_t*>(scratch_state.data_ptr()),
      device_buffer_tokens.strides()[0],
      cache_index.strides()[0],
      cache_index.strides()[1],
      hash_size,
      cache_policy.strides()[0],
      scratch_state.strides()[0],
      num_request_slots);
}

}  // namespace sglang
