#pragma once
#include "hisparse_spec.cuh"

namespace sglang {
namespace staging {

// Arena/forecast locations never enter the stable hot-cache location table.
enum Source : int32_t { Invalid = 0, Host = 1, Pool = 2, Forecast = 3 };
struct Plan {
  int32_t* table;
  int64_t* sources;
  int32_t* kinds;
  int32_t* demand;
  int32_t* tokens;
  int32_t* counts;
  int64_t capacity;
  int64_t hash_size;
};
__device__ __forceinline__ int32_t find(
    const int32_t* keys, const int32_t* values, int32_t size, int32_t token) {
  int32_t pos = static_cast<uint32_t>(token * 2654435761u) & (size - 1);
  for (int i = 0; i < size; ++i) {
    if (keys[pos] == token) return values[pos];
    if (keys[pos] == -1) return -1;
    pos = (pos + 1) & (size - 1);
  }
  return -1;
}

template <int HotSlots, int Steps, int TopK, int ForecastCapacity>
__global__ void plan_kernel(
    const int32_t* topk, const int32_t* lengths, const int32_t* window,
    const int32_t* history, const int64_t* reqs, const int32_t* live,
    const int32_t* hot_tokens, const int32_t* hot_locs, int64_t hot_stride,
    int64_t active_slots, const int64_t* host_locs, int64_t host_stride,
    SpecCacheState cache, const int32_t* flags, int64_t requests,
    const int32_t* forecast_index, int64_t forecast_hash, Plan plan,
    int32_t* output, int32_t decode_width) {
  const int bid = blockIdx.x;
  if (bid >= live[0]) return;
  constexpr int Capacity = Steps * TopK;
  const int64_t rid = reqs[bid];
  int32_t* keys = plan.table + bid * 2 * plan.hash_size;
  int32_t* values = keys + plan.hash_size;
  int32_t* kinds = plan.kinds + bid * Capacity;
  int32_t* demand = plan.demand + bid * Capacity;
  int32_t* tokens = plan.tokens + bid * Capacity;
  int64_t* sources = plan.sources + bid * Capacity;
  const int32_t* resident = hot_tokens + rid * hot_stride;
  const int32_t* resident_locs = hot_locs + rid * hot_stride;
  int64_t* primary = cache.hash_primary + rid * cache.hash_stride;
  int64_t* secondary = cache.hash_secondary + rid * cache.hash_stride;
  const bool degraded = (static_cast<uint32_t>(flags[3 * requests + rid]) & HASH_DEGRADED_FLAG) != 0;
  const int32_t epoch = PackedRingState<HotSlots>::next_epoch(cache.ring_state[rid]);
  int32_t* references = cache.ref_epochs + rid * cache.ref_epoch_stride;
  const int32_t* forecast_keys = forecast_index + rid * 2 * forecast_hash;
  const int32_t* forecast_values = forecast_keys + forecast_hash;
  for (int i = threadIdx.x; i < 2 * plan.hash_size; i += blockDim.x) keys[i] = -1;
  for (int i = threadIdx.x; i < Capacity; i += blockDim.x) {
    kinds[i] = Invalid;
    demand[i] = 0;
  }
  if (threadIdx.x == 0) plan.counts[bid] = 0;
  __syncthreads();

  // Owners publish indices before a block barrier; duplicates never spin on
  // another warp's producer. All selected rows fit the arena independently of B.
  for (int occ = threadIdx.x; occ < Capacity; occ += blockDim.x) {
    const int32_t token = topk[bid * Capacity + occ];
    if (token < 0 || token >= lengths[bid * Steps + occ / TopK]) continue;
    int32_t pos = static_cast<uint32_t>(token * 2654435761u) & (plan.hash_size - 1);
    for (int probe = 0; probe < plan.hash_size; ++probe) {
      const int32_t old = atomicCAS(keys + pos, -1, token);
      if (old == -1) {
        const int32_t uid = atomicAdd(plan.counts + bid, 1);
        values[pos] = uid;
        tokens[uid] = token;
        int32_t kind = Invalid;
        int64_t source = -1;
        const int32_t writer = window[bid * Capacity + occ];
        if (token >= history[bid] && writer >= 0) {
          kind = Pool;
          source = writer;
        } else {
          const int32_t slot = hot_cache_lookup<HotSlots>(
              primary, secondary, cache.hash_size, token, resident, degraded);
          if (slot >= 0 && resident_locs[slot] >= 0) {
            kind = Pool;
            source = resident_locs[slot];
          } else {
            for (int tail = HotSlots; tail < active_slots; ++tail) {
              if (resident[tail] == token && resident_locs[tail] >= 0) {
                kind = Pool;
                source = resident_locs[tail];
                break;
              }
            }
            if (kind == Invalid && ForecastCapacity > 0) {
              const int32_t found = find(forecast_keys, forecast_values, forecast_hash, token);
              if (found >= 0) {
                kind = Forecast;
                source = rid * ForecastCapacity + found;
              }
            }
            if (kind == Invalid && token < history[bid]) {
              source = host_locs[rid * host_stride + token];
              if (source >= 0) kind = Host;
            }
          }
        }
        sources[uid] = source;
        kinds[uid] = source >= 0 ? kind : Invalid;
        break;
      }
      if (old == token) break;
      pos = (pos + 1) & (plan.hash_size - 1);
    }
  }
  __syncthreads();
  for (int occ = threadIdx.x; occ < decode_width * TopK; occ += blockDim.x) {
    const int32_t token = topk[bid * Capacity + occ];
    if (token < 0 || token >= lengths[bid * Steps + occ / TopK]) continue;
    const int32_t uid = find(keys, values, plan.hash_size, token);
    if (uid >= 0) atomicExch(demand + uid, 1);
    const int32_t slot = hot_cache_lookup<HotSlots>(
        primary, secondary, cache.hash_size, token, resident, degraded);
    if (slot >= 0 && resident_locs[slot] >= 0) atomicExch(references + slot, epoch);
  }
  __syncthreads();
  for (int occ = threadIdx.x; occ < Capacity; occ += blockDim.x) {
    const int32_t token = topk[bid * Capacity + occ];
    int32_t loc = -1;
    if (token >= 0 && token < lengths[bid * Steps + occ / TopK]) {
      const int32_t uid = find(keys, values, plan.hash_size, token);
      if (uid >= 0 && kinds[uid] != Invalid && (kinds[uid] != Host || demand[uid])) {
        loc = static_cast<int32_t>(rid * Capacity + uid);
      }
    }
    output[bid * Capacity + occ] = loc;
  }
}

template <bool HostOnly>
__global__ void copy_kernel(
    Plan plan, const int64_t* reqs, const int32_t* live, const void* host,
    const void* pool, const void* forecast, void* arena) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int warps = gridDim.x * blockDim.x / WARP_SIZE;
  int start = 0;
  for (int bid = 0; bid < live[0]; ++bid) {
    const int count = plan.counts[bid];
    if (!count) continue;
    int first = (warp - start) % warps;
    if (first < 0) first += warps;
    for (int uid = first; uid < count; uid += warps) {
      const int64_t index = bid * plan.capacity + uid;
      const int kind = plan.kinds[index];
      if (kind == Invalid || (kind == Host && !plan.demand[index])) continue;
      if ((kind == Host) != HostOnly) continue;
      const void* src = kind == Host ? host : (kind == Pool ? pool : forecast);
      Dsv4PagedKVTransferPolicy::copy_warp(
          lane, src, arena, plan.sources[index], reqs[bid] * plan.capacity + uid);
    }
    start = (start + count) % warps;
  }
}

template <int HotSlots>
__global__ void promote_kernel(
    Plan plan, const int64_t* reqs, const int32_t* live, const int32_t* history,
    int32_t* hot_tokens, const int32_t* hot_locs, int64_t hot_stride,
    SpecCacheState cache, int32_t* flags, int64_t requests,
    const void* arena, void* pool) {
  const int bid = blockIdx.x;
  if (bid >= live[0]) return;
  const int64_t rid = reqs[bid];
  using Ring = PackedRingState<HotSlots>;
  const int packed = cache.ring_state[rid], epoch = Ring::next_epoch(packed), cursor = Ring::cursor(packed);
  int32_t* references = cache.ref_epochs + rid * cache.ref_epoch_stride;
  int64_t* primary = cache.hash_primary + rid * cache.hash_stride;
  int64_t* secondary = cache.hash_secondary + rid * cache.hash_stride;
  int32_t* tokens = hot_tokens + rid * hot_stride;
  const int32_t* locations = hot_locs + rid * hot_stride;
  const bool degraded = (static_cast<uint32_t>(flags[3 * requests + rid]) & HASH_DEGRADED_FLAG) != 0;
  const int lane = threadIdx.x % WARP_SIZE;
  for (int uid = threadIdx.x / WARP_SIZE; uid < plan.counts[bid]; uid += blockDim.x / WARP_SIZE) {
    const int64_t index = bid * plan.capacity + uid;
    int32_t dst = -1;
    if (lane == 0 && plan.demand[index] && plan.kinds[index] != Invalid) {
      const int token = plan.tokens[index];
      if (token < history[bid] &&
          hot_cache_lookup<HotSlots>(primary, secondary, cache.hash_size, token, tokens, degraded) < 0) {
        // Stable addresses only: overflowing demand stays in the arena.
        for (int probe = 0; probe < HotSlots; ++probe) {
          const int victim = (cursor + uid + probe) % HotSlots;
          if (locations[victim] < 0) continue;
          const int observed = references[victim];
          if (observed == epoch || atomicCAS(references + victim, observed, epoch) != observed) continue;
          ring_hash_erase_atomic<HotSlots>(primary, secondary, cache.hash_size, tokens[victim], victim);
          tokens[victim] = token;
          dst = locations[victim];
          if (ring_hash_insert_atomic<HotSlots>(primary, secondary, cache.hash_size, token, victim) < 0) {
            atomicOr(flags + 3 * requests + rid, static_cast<int32_t>(HASH_DEGRADED_FLAG));
          }
          break;
        }
      }
    }
    dst = __shfl_sync(FULL_WARP_MASK, dst, 0);
    if (dst >= 0) Dsv4PagedKVTransferPolicy::copy_warp(lane, arena, pool, rid * plan.capacity + uid, dst);
  }
  __syncthreads();
  if (threadIdx.x == 0) cache.ring_state[rid] = Ring::pack(epoch, (cursor + plan.counts[bid]) % HotSlots);
}

template <int HotSlots, int Capacity>
__global__ void forecast_plan_kernel(
    const int32_t* forecast, const int64_t* reqs, const int32_t* live,
    const int64_t* host_locs, int64_t host_stride, int32_t* index,
    int64_t hash_size, int64_t* sources, int32_t* counts,
    const int32_t* hot_tokens, int64_t hot_stride, int64_t active_slots,
    SpecCacheState cache, const int32_t* flags, int64_t requests) {
  const int bid = blockIdx.x;
  if (bid >= live[0]) return;
  const int64_t rid = reqs[bid];
  int32_t* keys = index + rid * 2 * hash_size;
  int32_t* values = keys + hash_size;
  for (int i = threadIdx.x; i < 2 * hash_size; i += blockDim.x) keys[i] = -1;
  if (threadIdx.x == 0) counts[rid] = 0;
  __syncthreads();
  for (int occ = threadIdx.x; occ < Capacity; occ += blockDim.x) {
    const int token = forecast[bid * Capacity + occ];
    if (token < 0) continue;
    const int64_t source = host_locs[rid * host_stride + token];
    if (source < 0) continue;
    int pos = static_cast<uint32_t>(token * 2654435761u) & (hash_size - 1);
    for (int probe = 0; probe < hash_size; ++probe) {
      const int old = atomicCAS(keys + pos, -1, token);
      if (old == -1) {
        // Deduplicate before cache lookup. A ready hot hit needs no H2D.
        const int32_t* resident = hot_tokens + rid * hot_stride;
        const bool degraded = (static_cast<uint32_t>(flags[3 * requests + rid]) & HASH_DEGRADED_FLAG) != 0;
        bool hit = hot_cache_lookup<HotSlots>(
            cache.hash_primary + rid * cache.hash_stride,
            cache.hash_secondary + rid * cache.hash_stride,
            cache.hash_size, token, resident, degraded) >= 0;
        // The mutable canonical tail can be overwritten by acceptance/commit.
        // Only stable hot hits may omit the next-round forecast copy.
        if (!hit) {
          const int uid = atomicAdd(counts + rid, 1);
          values[pos] = uid;
          sources[rid * Capacity + uid] = source;
        }
        break;
      }
      if (old == token) break;
      pos = (pos + 1) & (hash_size - 1);
    }
  }
}
template <int Capacity>
__global__ void forecast_copy_kernel(
    const int64_t* reqs, const int32_t* live, const int64_t* sources,
    const int32_t* counts, const void* host, void* arena) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int warps = gridDim.x * blockDim.x / WARP_SIZE;
  int start = 0;
  for (int bid = 0; bid < live[0]; ++bid) {
    const int64_t rid = reqs[bid];
    const int count = counts[rid];
    if (!count) continue;
    int first = (warp - start) % warps;
    if (first < 0) first += warps;
    for (int uid = first; uid < count; uid += warps) {
      Dsv4PagedKVTransferPolicy::copy_warp(lane, host, arena, sources[rid * Capacity + uid], rid * Capacity + uid);
    }
    start = (start + count) % warps;
  }
}

inline uint32_t transfer_blocks(int64_t batch) {
  return static_cast<uint32_t>(batch < 4 ? 4 : (batch > 32 ? 32 : batch));
}

template <typename T> inline T* ptr(tvm::ffi::TensorView t) { return static_cast<T*>(t.data_ptr()); }
#define ST_V(name) tvm::ffi::TensorView name
inline Plan make_plan(ST_V(table), ST_V(sources), ST_V(kinds), ST_V(demand), ST_V(tokens), ST_V(counts)) {
  return {ptr<int32_t>(table), ptr<int64_t>(sources), ptr<int32_t>(kinds), ptr<int32_t>(demand),
          ptr<int32_t>(tokens), ptr<int32_t>(counts), sources.shape()[1], table.shape()[1] / 2};
}
inline SpecCacheState make_cache(ST_V(index), ST_V(policy)) {
  return {ptr<int64_t>(index), ptr<int64_t>(index) + index.strides()[1], ptr<int32_t>(policy),
          ptr<int32_t>(policy) + policy.strides()[0], index.strides()[0], index.shape()[2], policy.strides()[0]};
}
template <int HotSlots, int Steps, int TopK, int ForecastCapacity>
void plan_verify(
    ST_V(topk), ST_V(lengths), ST_V(window), ST_V(history), ST_V(reqs), ST_V(live),
    ST_V(hot_tokens), ST_V(hot_locs), ST_V(host_locs), ST_V(cache_index),
    ST_V(cache_policy), ST_V(cache_flags), ST_V(forecast_index),
    ST_V(table), ST_V(sources), ST_V(kinds), ST_V(demand), ST_V(tokens), ST_V(counts),
    ST_V(output), int64_t decode_width) {
  using namespace host;
  RuntimeCheck(topk.ndim() == 3 && topk.shape()[1] == Steps && topk.shape()[2] == TopK,
               "verification staging geometry mismatch");
  RuntimeCheck(sources.shape()[1] == Steps * TopK && table.shape()[1] / 2 >= 2 * Steps * TopK,
               "verification staging plan capacity mismatch");
  const auto device = LaunchKernel::resolve_device(topk.device());
  LaunchKernel(static_cast<uint32_t>(topk.shape()[0]), 256, device)(
      plan_kernel<HotSlots, Steps, TopK, ForecastCapacity>,
      ptr<int32_t>(topk), ptr<int32_t>(lengths), ptr<int32_t>(window), ptr<int32_t>(history),
      ptr<int64_t>(reqs), ptr<int32_t>(live), ptr<int32_t>(hot_tokens), ptr<int32_t>(hot_locs),
      hot_tokens.strides()[0], hot_tokens.shape()[1], ptr<int64_t>(host_locs), host_locs.strides()[0],
      make_cache(cache_index, cache_policy), ptr<int32_t>(cache_flags), cache_index.shape()[0],
      ptr<int32_t>(forecast_index), forecast_index.shape()[1] / 2,
      make_plan(table, sources, kinds, demand, tokens, counts), ptr<int32_t>(output), static_cast<int32_t>(decode_width));
}
template <bool HostOnly>
void copy_verify(
    ST_V(table), ST_V(sources), ST_V(kinds), ST_V(demand), ST_V(tokens), ST_V(counts),
    ST_V(reqs), ST_V(live), ST_V(host), ST_V(pool), ST_V(forecast), ST_V(arena)) {
  const auto device = LaunchKernel::resolve_device(sources.device());
  LaunchKernel(transfer_blocks(reqs.shape()[0]), 256, device)(
      copy_kernel<HostOnly>, make_plan(table, sources, kinds, demand, tokens, counts),
      ptr<int64_t>(reqs), ptr<int32_t>(live), host.data_ptr(), pool.data_ptr(), forecast.data_ptr(), arena.data_ptr());
}
template <int HotSlots>
void promote(
    ST_V(table), ST_V(sources), ST_V(kinds), ST_V(demand), ST_V(tokens), ST_V(counts),
    ST_V(reqs), ST_V(live), ST_V(history), ST_V(hot_tokens), ST_V(hot_locs),
    ST_V(cache_index), ST_V(cache_policy), ST_V(cache_flags), ST_V(arena), ST_V(pool)) {
  const auto device = LaunchKernel::resolve_device(sources.device());
  LaunchKernel(static_cast<uint32_t>(reqs.shape()[0]), 256, device)(
      promote_kernel<HotSlots>, make_plan(table, sources, kinds, demand, tokens, counts),
      ptr<int64_t>(reqs), ptr<int32_t>(live), ptr<int32_t>(history),
      ptr<int32_t>(hot_tokens), ptr<int32_t>(hot_locs), hot_tokens.strides()[0],
      make_cache(cache_index, cache_policy), ptr<int32_t>(cache_flags), cache_index.shape()[0],
      arena.data_ptr(), pool.data_ptr());
}
template <int HotSlots, int Capacity>
void prefetch(
    ST_V(forecast), ST_V(reqs), ST_V(live), ST_V(host_locs), ST_V(index),
    ST_V(sources), ST_V(counts), ST_V(host), ST_V(arena),
    ST_V(hot_tokens), ST_V(cache_index), ST_V(cache_policy), ST_V(cache_flags)) {
  const auto device = LaunchKernel::resolve_device(forecast.device());
  const uint32_t batch = static_cast<uint32_t>(reqs.shape()[0]);
  LaunchKernel(batch, 256, device)(
      forecast_plan_kernel<HotSlots, Capacity>,
      ptr<int32_t>(forecast), ptr<int64_t>(reqs), ptr<int32_t>(live),
      ptr<int64_t>(host_locs), host_locs.strides()[0], ptr<int32_t>(index), index.shape()[1] / 2,
      ptr<int64_t>(sources), ptr<int32_t>(counts), ptr<int32_t>(hot_tokens), hot_tokens.strides()[0],
      hot_tokens.shape()[1], make_cache(cache_index, cache_policy), ptr<int32_t>(cache_flags), cache_index.shape()[0]);
  LaunchKernel(transfer_blocks(batch), 256, device)(
      forecast_copy_kernel<Capacity>, ptr<int64_t>(reqs), ptr<int32_t>(live),
      ptr<int64_t>(sources), ptr<int32_t>(counts), host.data_ptr(), arena.data_ptr());
}
#undef ST_V
}  // namespace staging
}  // namespace sglang
