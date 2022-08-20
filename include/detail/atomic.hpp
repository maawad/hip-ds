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

#include <atomic>
#include "hip/hip_runtime.h"

namespace bght {
namespace detail {
template <typename T, auto Scope>
struct atomic {
  static_assert(sizeof(T) <= 8);
  atomic(T* ptr) : data_(ptr) {}
  atomic(const atomic&) = delete;

  __device__ T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    return *data_;
  }
  __device__ void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
    *data_ = desired;
  }
  __device__ T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) {
    return atomicExch(data_, desired);
  }
  __device__ bool compare_exchange_weak(T& expected,
                                        T desired,
                                        std::memory_order success,
                                        std::memory_order failure) {
    auto old = atomicCAS(data_, expected, desired);
    return old == expected;
  }
  __device__ bool compare_exchange_strong(T& expected,
                                          T desired,
                                          std::memory_order success,
                                          std::memory_order failure) {
    auto old = atomicCAS(data_, expected, desired);
    return old == expected;
  }

 private:
  T* data_;
};
enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};
}  // namespace detail

}  // namespace bght
