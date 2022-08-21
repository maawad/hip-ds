/*
 *   Copyright 2021 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once

#include "hip/hip_runtime.h"

namespace bght {
namespace detail {
namespace cooperative_groups {
template <int width = 64>
struct tiled_partition {
  __device__ static auto thread_rank() { return threadIdx.x & 0x1f; }
  __device__ static uint64_t ballot(int predicate) { return __ballot(predicate); }
  template <typename T>
  __device__ static auto shfl(T var, int src_lane) {
    static_assert(sizeof(T) <= 8);
    if constexpr (sizeof(T) > 4) {
      T result;
      result.first  = __shfl(var.first, src_lane, width);
      result.second = __shfl(var.second, src_lane, width);
      return result;
    } else {
      return __shfl(var, src_lane, width);
    }
  }
  __device__ static auto all(int predicate) { return __all(predicate); }
};
};  // namespace cooperative_groups

}  // namespace detail

}  // namespace bght
