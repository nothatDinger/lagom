#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <tvm/ffi/container/tensor.h>

#include "hisparse_spec_transfer.cuh"
#include <algorithm>
#include <assert.h>
#include <type_traits>

namespace sglang {
using namespace host;

struct DHiSparsePlan {
  int32_t *keys, *locations, *hot_indices, *tokens;
  int64_t *sources, *hits;
  SwapInCounts* counts;
  int32_t *miss_order, *activity;
  int64_t hash_size, capacity;
};

__device__ inline int dhisparse_hash(int token, int size) {
  return (static_cast<uint32_t>(token) * 2654435761u) & (size - 1);
}

__device__ inline int dhisparse_find(int32_t* keys, int token, int size) {
  int slot = dhisparse_hash(token, size);
  while (keys[slot] != token) {
    if (keys[slot] == -1) return -1;
    slot = (slot + 1) & (size - 1);
  }
  return slot;
}

__device__ inline int dhisparse_insert(int32_t* keys, int token, int size, bool& owner) {
  int slot = dhisparse_hash(token, size);
  for (int probe = 0; probe < size; ++probe) {
    int old = atomicCAS(keys + slot, -1, token);
    if (old == -1 || old == token) {
      owner = old == -1;
      return slot;
    }
    slot = (slot + 1) & (size - 1);
  }
  assert(false && "DHiSparse union hash exhausted");
  return -1;
}

__global__ void dhisparse_plan_kernel(
    const int32_t* selected,
    const int64_t* requests,
    const int32_t* lengths,
    const int32_t* optional,
    const int64_t* host_locs,
    const int32_t* buffer_tokens,
    const int32_t* buffer_locs,
    int64_t* ages,
    int64_t* epochs,
    int32_t* output,
    DHiSparsePlan plan,
    const int32_t* real,
    int width,
    int top_k,
    int hot,
    int buffer_stride,
    int logical_stride,
    int64_t host_stride,
    int64_t staging_offset,
    int staging_stride) {
  const int batch = blockIdx.x, tid = threadIdx.x;
  const int occurrences = width * top_k;
  if (tid == 0) plan.activity[batch] = -1;
  if (batch >= *real) {
    for (int i = tid; i < occurrences; i += blockDim.x)
      output[batch * occurrences + i] = -1;
    return;
  }
  const int request = requests[batch];
  int32_t* keys = plan.keys + batch * plan.hash_size;
  int32_t* locations = plan.locations + batch * plan.hash_size;
  int32_t* hot_indices = plan.hot_indices + batch * plan.hash_size;
  int64_t* sources = plan.sources + batch * plan.capacity;
  int32_t* tokens = plan.tokens + batch * plan.capacity;
  int64_t* hits = plan.hits ? plan.hits + batch * plan.capacity : nullptr;
  __shared__ int unresolved;
  if (tid == 0) {
    unresolved = 0;
    ++epochs[request];
  }
  for (int i = tid; i < plan.capacity; i += blockDim.x) {
    sources[i] = -1;
    tokens[i] = -1;
    if (hits) hits[i] = -1;
  }
  __syncthreads();
  // Direct mappings are checked before touching the bounded replacement set.
  for (int i = tid; i < occurrences; i += blockDim.x) {
    const int token = selected[batch * occurrences + i];
    int loc = -1;
    if (token >= 0 && token < lengths[batch * width + i / top_k]) {
      assert(token < logical_stride);
      for (int slot = hot; slot < buffer_stride; ++slot) {
        if (buffer_tokens[request * buffer_stride + slot] == token) {
          loc = buffer_locs[request * buffer_stride + slot];
          break;
        }
      }
      if (loc < 0) loc = optional[request * logical_stride + token];
      if (loc < 0) atomicAdd(&unresolved, 1);
    }
    output[batch * occurrences + i] = loc;
  }
  __syncthreads();
  if (unresolved == 0) return;

  for (int i = tid; i < plan.hash_size; i += blockDim.x) {
    keys[i] = -1;
    locations[i] = occurrences;
    hot_indices[i] = -1;
  }
  __syncthreads();
  for (int slot = tid; slot < hot; slot += blockDim.x) {
    const int token = buffer_tokens[request * buffer_stride + slot];
    if (token >= 0) {
      bool owner;
      const int pos = dhisparse_insert(keys, token, plan.hash_size, owner);
      if (owner) {
        locations[pos] = buffer_locs[request * buffer_stride + slot];
        hot_indices[pos] = slot;
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < occurrences; i += blockDim.x) {
    const int token = selected[batch * occurrences + i];
    if (token < 0 || token >= lengths[batch * width + i / top_k] || output[batch * occurrences + i] >= 0) continue;
    bool owner;
    const int pos = dhisparse_insert(keys, token, plan.hash_size, owner);
    // Choose the first query occurrence deterministically. Warp scheduling
    // must not change which overflow misses are admitted to hot.
    if (hot_indices[pos] < 0) atomicMin(locations + pos, i);
  }
  __syncthreads();
  for (int i = tid; i < occurrences; i += blockDim.x) {
    const int token = selected[batch * occurrences + i];
    if (token < 0 || token >= lengths[batch * width + i / top_k] || output[batch * occurrences + i] >= 0) continue;
    const int pos = dhisparse_find(keys, token, plan.hash_size);
    assert(pos >= 0);
    const int slot = hot_indices[pos];
    if (slot >= 0) {
      output[batch * occurrences + i] = locations[pos];
      auto* age = reinterpret_cast<unsigned long long*>(ages + request * hot + slot);
      const auto epoch = static_cast<unsigned long long>(epochs[request]);
      // The first use in this step elects exactly one reread for a warm hit.
      if (atomicExch(age, epoch) != epoch && hits && lengths[batch * width] > hot) {
        const auto src = host_locs[request * host_stride + token];
        if (src >= 0) {
          hits[i] = src;
          if (plan.counts) atomicAdd(&plan.counts->eligible_hits, 1);
        }
      }
    } else {
      const int first = locations[pos];
      output[batch * occurrences + i] = staging_offset + batch * staging_stride + first;
      if (i == first) {
        const int64_t src = host_locs[request * host_stride + token];
        assert(src >= 0 && "Adaptive HiSparse selected a row without host backing");
        sources[i] = src;
        tokens[i] = token;
        if (plan.counts) atomicAdd(&plan.counts->gather_copies, 1);
      }
    }
  }

  // Compact unique misses in query order; hash insertion itself is unordered.
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage temp;
  __syncthreads();
  int base = 0;
  for (int start = 0; start < occurrences; start += blockDim.x) {
    const int i = start + tid;
    const int miss = i < occurrences && sources[i] >= 0;
    int rank, total;
    Scan(temp).ExclusiveSum(miss, rank, total);
    if (miss) plan.miss_order[batch * plan.capacity + base + rank] = i;
    base += total;
    __syncthreads();
  }
  if (tid == 0) plan.activity[batch] = base;
}

// Stable partition of the bounded LRU: protect this step's hits and choose
// only unselected victims. Direct-only requests do not touch the hot set.
__global__ void dhisparse_admit_kernel(
    const int64_t* requests,
    const int64_t* ages,
    const int64_t* epochs,
    int32_t* lru,
    int32_t* victims,
    int32_t* activity,
    const int32_t* real,
    int hot) {
  const int batch = blockIdx.x;
  if (batch >= *real || activity[batch] < 0) return;
  const int request = requests[batch], tid = threadIdx.x;
  lru += request * hot;
  ages += request * hot;
  victims += batch * hot;
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage temp;
  int available = 0;
  for (int start = 0; start < hot; start += blockDim.x) {
    const int i = start + tid;
    const int slot = i < hot ? lru[i] : 0;
    const int cold = i < hot && ages[slot] != epochs[request];
    int before, total;
    Scan(temp).ExclusiveSum(cold, before, total);
    before += available;
    if (i < hot) victims[cold ? before : hot - 1 - (i - before)] = slot;
    available += total;
    __syncthreads();
  }
  const int admitted = min(available, activity[batch]);
  // All lanes must read the original number of misses before thread 0 updates it.
  __syncthreads();
  if (tid == 0) activity[batch] = admitted;
  for (int i = tid; i < hot; i += blockDim.x) {
    if (i < available - admitted)
      lru[i] = victims[i + admitted];
    else if (i < hot - admitted)
      lru[i] = victims[hot - 1 - (i - (available - admitted))];
    else
      lru[i] = victims[i - (hot - admitted)];
  }
}

enum class DHiSparseCopy { Refill, Reread, Promote };

template <int BYTES, bool PACKED, DHiSparseCopy KIND>
__global__ void dhisparse_copy_kernel(
    const int64_t* requests,
    const int64_t* sources,
    const int32_t* tokens,
    const void* src,
    void* dst,
    int32_t* buffer_tokens,
    const int32_t* buffer_locs,
    int64_t* ages,
    const int64_t* epochs,
    const int32_t* victims,
    const int32_t* order,
    const int32_t* activity,
    SwapInCounts* counts,
    const int32_t* real,
    int capacity,
    int buffer_stride,
    int hot,
    int64_t staging_offset,
    int staging_stride) {
  const int batch = blockIdx.x;
  if (batch >= *real) return;
  const int request = requests[batch];
  const int lane = threadIdx.x % 32;
  const int warp = (blockIdx.y * blockDim.x + threadIdx.x) / 32;
  const int warps = gridDim.y * blockDim.x / 32;
  using Transfer = std::conditional_t<PACKED, Dsv4PagedKVTransferPolicy, LinearKVTransferPolicy<BYTES>>;
  int limit = capacity;
  int copied = 0;
  if constexpr (KIND == DHiSparseCopy::Promote) limit = activity[batch];
  for (int rank = warp; rank < limit; rank += warps) {
    int i = rank;
    if constexpr (KIND == DHiSparseCopy::Promote) i = order[batch * capacity + rank];
    int64_t source = sources[batch * capacity + i];
    if (source < 0) continue;
    int64_t destination = staging_offset + batch * staging_stride + i;
    int victim = -1;
    if constexpr (KIND == DHiSparseCopy::Promote) {
      victim = victims[batch * hot + rank];
      source = destination;
      destination = buffer_locs[request * buffer_stride + victim];
    }
    Transfer::copy_warp(lane, src, dst, source, destination);
    __syncwarp();
    if constexpr (KIND == DHiSparseCopy::Reread) ++copied;
    if constexpr (KIND == DHiSparseCopy::Promote) {
      if (lane == 0) {
        buffer_tokens[request * buffer_stride + victim] = tokens[batch * capacity + i];
        ages[request * hot + victim] = epochs[request];
        if (counts) atomicAdd(&counts->admissions, 1);
      }
    }
  }
  if constexpr (KIND == DHiSparseCopy::Reread) {
    if (counts) {
      // One global counter update per block, outside the row-copy loop.
      using Reduce = cub::BlockReduce<int, 256>;
      __shared__ typename Reduce::TempStorage temp;
      const int rows = Reduce(temp).Sum(lane == 0 ? copied : 0);
      if (threadIdx.x == 0 && rows) atomicAdd(&counts->reread_copies, rows);
    }
  }
}

template <int BYTES, bool PACKED>
void dhisparse_resolve(
    tvm::ffi::TensorView selected,
    tvm::ffi::TensorView requests,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView optional,
    tvm::ffi::TensorView host_locs,
    tvm::ffi::TensorView buffer_tokens,
    tvm::ffi::TensorView buffer_locs,
    tvm::ffi::TensorView ages,
    tvm::ffi::TensorView epochs,
    tvm::ffi::TensorView host,
    tvm::ffi::TensorView device,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView hot_indices,
    tvm::ffi::TensorView sources,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView hits,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView activity,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView real,
    int64_t staging_offset,
    int64_t staging_stride,
    int64_t hot) {
  const int batch = selected.shape()[0], width = selected.shape()[1], top_k = selected.shape()[2];
  RuntimeCheck(width * top_k <= staging_stride && width * top_k <= sources.shape()[1], "DHiSparse staging overflow");
  DHiSparsePlan plan{
      static_cast<int32_t*>(keys.data_ptr()),
      static_cast<int32_t*>(locations.data_ptr()),
      static_cast<int32_t*>(hot_indices.data_ptr()),
      static_cast<int32_t*>(tokens.data_ptr()),
      static_cast<int64_t*>(sources.data_ptr()),
      hits.numel() ? static_cast<int64_t*>(hits.data_ptr()) : nullptr,
      counts.numel() ? static_cast<SwapInCounts*>(counts.data_ptr()) : nullptr,
      static_cast<int32_t*>(order.data_ptr()),
      static_cast<int32_t*>(activity.data_ptr()),
      keys.shape()[1],
      sources.shape()[1]};
  LaunchKernel(dim3(batch), 256, selected.device())(
      dhisparse_plan_kernel,
      static_cast<const int32_t*>(selected.data_ptr()),
      static_cast<const int64_t*>(requests.data_ptr()),
      static_cast<const int32_t*>(lengths.data_ptr()),
      static_cast<const int32_t*>(optional.data_ptr()),
      static_cast<const int64_t*>(host_locs.data_ptr()),
      static_cast<const int32_t*>(buffer_tokens.data_ptr()),
      static_cast<const int32_t*>(buffer_locs.data_ptr()),
      static_cast<int64_t*>(ages.data_ptr()),
      static_cast<int64_t*>(epochs.data_ptr()),
      static_cast<int32_t*>(output.data_ptr()),
      plan,
      static_cast<const int32_t*>(real.data_ptr()),
      width,
      top_k,
      static_cast<int>(hot),
      static_cast<int>(buffer_tokens.strides()[0]),
      static_cast<int>(optional.strides()[0]),
      host_locs.strides()[0],
      staging_offset,
      static_cast<int>(staging_stride));
  const int blocks = std::min(32, std::max(4, batch));
  LaunchKernel(dim3(batch, blocks), 256, selected.device())(
      dhisparse_copy_kernel<BYTES, PACKED, DHiSparseCopy::Refill>,
      static_cast<const int64_t*>(requests.data_ptr()),
      plan.sources,
      plan.tokens,
      host.data_ptr(),
      device.data_ptr(),
      static_cast<int32_t*>(buffer_tokens.data_ptr()),
      static_cast<const int32_t*>(buffer_locs.data_ptr()),
      static_cast<int64_t*>(ages.data_ptr()),
      static_cast<const int64_t*>(epochs.data_ptr()),
      static_cast<const int32_t*>(nullptr),
      static_cast<const int32_t*>(nullptr),
      nullptr,
      nullptr,
      static_cast<const int32_t*>(real.data_ptr()),
      static_cast<int>(sources.shape()[1]),
      static_cast<int>(buffer_tokens.strides()[0]),
      static_cast<int>(hot),
      staging_offset,
      static_cast<int>(staging_stride));
}

template <int BYTES, bool PACKED>
void dhisparse_reread(
    tvm::ffi::TensorView requests,
    tvm::ffi::TensorView hits,
    tvm::ffi::TensorView host,
    tvm::ffi::TensorView device,
    tvm::ffi::TensorView real,
    tvm::ffi::TensorView counts,
    int64_t offset,
    int64_t stride) {
  if (counts.numel()) TensorMatcher({5}).with_dtype<int32_t>().with_device(requests.device()).verify(counts);
  LaunchKernel(dim3(requests.numel(), std::clamp<int>(requests.numel(), 4, 32)), 256, requests.device())(
      dhisparse_copy_kernel<BYTES, PACKED, DHiSparseCopy::Reread>,
      static_cast<const int64_t*>(requests.data_ptr()),
      static_cast<const int64_t*>(hits.data_ptr()),
      nullptr,
      host.data_ptr(),
      device.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      counts.numel() ? static_cast<SwapInCounts*>(counts.data_ptr()) : nullptr,
      static_cast<const int32_t*>(real.data_ptr()),
      static_cast<int>(hits.shape()[1]),
      0,
      0,
      offset,
      static_cast<int>(stride));
}

template <int BYTES, bool PACKED>
void dhisparse_promote(
    tvm::ffi::TensorView requests,
    tvm::ffi::TensorView buffer_tokens,
    tvm::ffi::TensorView buffer_locs,
    tvm::ffi::TensorView ages,
    tvm::ffi::TensorView epochs,
    tvm::ffi::TensorView device,
    tvm::ffi::TensorView sources,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView victims,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView activity,
    tvm::ffi::TensorView lru,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView real,
    int64_t staging_offset,
    int64_t staging_stride,
    int64_t hot) {
  LaunchKernel(dim3(requests.numel()), 256, requests.device())(
      dhisparse_admit_kernel,
      static_cast<const int64_t*>(requests.data_ptr()),
      static_cast<const int64_t*>(ages.data_ptr()),
      static_cast<const int64_t*>(epochs.data_ptr()),
      static_cast<int32_t*>(lru.data_ptr()),
      static_cast<int32_t*>(victims.data_ptr()),
      static_cast<int32_t*>(activity.data_ptr()),
      static_cast<const int32_t*>(real.data_ptr()),
      static_cast<int>(hot));
  LaunchKernel(dim3(requests.numel(), 4), 256, requests.device())(
      dhisparse_copy_kernel<BYTES, PACKED, DHiSparseCopy::Promote>,
      static_cast<const int64_t*>(requests.data_ptr()),
      static_cast<const int64_t*>(sources.data_ptr()),
      static_cast<const int32_t*>(tokens.data_ptr()),
      device.data_ptr(),
      device.data_ptr(),
      static_cast<int32_t*>(buffer_tokens.data_ptr()),
      static_cast<const int32_t*>(buffer_locs.data_ptr()),
      static_cast<int64_t*>(ages.data_ptr()),
      static_cast<const int64_t*>(epochs.data_ptr()),
      static_cast<const int32_t*>(victims.data_ptr()),
      static_cast<const int32_t*>(order.data_ptr()),
      static_cast<const int32_t*>(activity.data_ptr()),
      counts.numel() ? static_cast<SwapInCounts*>(counts.data_ptr()) : nullptr,
      static_cast<const int32_t*>(real.data_ptr()),
      static_cast<int>(sources.shape()[1]),
      static_cast<int>(buffer_tokens.strides()[0]),
      static_cast<int>(hot),
      staging_offset,
      static_cast<int>(staging_stride));
}
}  // namespace sglang
