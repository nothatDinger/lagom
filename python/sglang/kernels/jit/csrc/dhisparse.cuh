#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "hisparse_spec_transfer.cuh"
#include <algorithm>
#include <assert.h>
#include <type_traits>

namespace sglang {
using namespace host;

struct DHiSparsePlan {
  int32_t *keys, *locations, *hot_indices, *tokens;
  int64_t* sources;
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
  if (batch >= *real) {
    for (int i = tid; i < occurrences; i += blockDim.x)
      output[batch * occurrences + i] = -1;
    return;
  }
  const int request = requests[batch];
  int32_t* keys = plan.keys + request * plan.hash_size;
  int32_t* locations = plan.locations + request * plan.hash_size;
  int32_t* hot_indices = plan.hot_indices + request * plan.hash_size;
  int64_t* sources = plan.sources + request * plan.capacity;
  int32_t* tokens = plan.tokens + request * plan.capacity;
  __shared__ int unresolved;
  if (tid == 0) {
    unresolved = 0;
    ++epochs[request];
  }
  for (int i = tid; i < plan.capacity; i += blockDim.x) {
    sources[i] = -1;
    tokens[i] = -1;
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
    if (owner) {
      const int64_t src = host_locs[request * host_stride + token];
      assert(src >= 0 && "DHiSparse row has neither GPU residency nor host backing");
      locations[pos] = staging_offset + request * staging_stride + i;
      sources[i] = src;
      tokens[i] = token;
    }
  }
  __syncthreads();
  for (int i = tid; i < occurrences; i += blockDim.x) {
    const int token = selected[batch * occurrences + i];
    if (token < 0 || token >= lengths[batch * width + i / top_k] || output[batch * occurrences + i] >= 0) continue;
    const int pos = dhisparse_find(keys, token, plan.hash_size);
    assert(pos >= 0);
    output[batch * occurrences + i] = locations[pos];
    const int slot = hot_indices[pos];
    if (slot >= 0) ages[request * hot + slot] = epochs[request];
  }
}

template <int BYTES, bool PACKED, bool PROMOTE>
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
    const int32_t* ranks,
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
  for (int i = warp; i < capacity; i += warps) {
    int64_t source = sources[request * capacity + i];
    if (source < 0) continue;
    int64_t destination = staging_offset + request * staging_stride + i;
    int victim = -1;
    if constexpr (PROMOTE) {
      const int rank = ranks[request * capacity + i];
      if (rank >= hot) continue;
      victim = victims[request * hot + rank];
      source = destination;
      destination = buffer_locs[request * buffer_stride + victim];
    }
    Transfer::copy_warp(lane, src, dst, source, destination);
    __syncwarp();
    if constexpr (PROMOTE) {
      if (lane == 0) {
        buffer_tokens[request * buffer_stride + victim] = tokens[request * capacity + i];
        ages[request * hot + victim] = epochs[request];
      }
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
      dhisparse_copy_kernel<BYTES, PACKED, false>,
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
      static_cast<const int32_t*>(real.data_ptr()),
      static_cast<int>(sources.shape()[1]),
      static_cast<int>(buffer_tokens.strides()[0]),
      static_cast<int>(hot),
      staging_offset,
      static_cast<int>(staging_stride));
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
    tvm::ffi::TensorView ranks,
    tvm::ffi::TensorView real,
    int64_t staging_offset,
    int64_t staging_stride,
    int64_t hot) {
  LaunchKernel(dim3(requests.numel(), 4), 256, requests.device())(
      dhisparse_copy_kernel<BYTES, PACKED, true>,
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
      static_cast<const int32_t*>(ranks.data_ptr()),
      static_cast<const int32_t*>(real.data_ptr()),
      static_cast<int>(sources.shape()[1]),
      static_cast<int>(buffer_tokens.strides()[0]),
      static_cast<int>(hot),
      staging_offset,
      static_cast<int>(staging_stride));
}
}  // namespace sglang
