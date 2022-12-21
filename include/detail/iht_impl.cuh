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
#include <detail/atomic.hpp>
#include <detail/benchmark_metrics.cuh>
#include <detail/bucket.hpp>
#include <detail/cooperative_groups.hpp>
#include <detail/kernels.hpp>
#include <detail/rng.hpp>
#include <iht.hpp>
#include <iterator>
#include <random>

namespace bght {
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iht(std::size_t capacity,
                                                                 Key empty_key_sentinel,
                                                                 T empty_value_sentinel,
                                                                 Allocator const& allocator)
    : capacity_{std::max(capacity, std::size_t{1})}
    , sentinel_key_{empty_key_sentinel}
    , sentinel_value_{empty_value_sentinel}
    , allocator_{allocator}
    , atomic_pairs_allocator_{allocator}
    , bool_allocator_{allocator}
    , size_type_allocator_{allocator} {
  capacity_    = detail::get_valid_capacity<bucket_size>(capacity_);
  num_buckets_ = capacity_ / bucket_size;

  std::cout << "Num buckets: " << num_buckets_ << std::endl;
  d_table_ = std::allocator_traits<atomic_pair_allocator_type>::allocate(atomic_pairs_allocator_,
                                                                         capacity_);
  table_   = std::shared_ptr<atomic_pair_type>(d_table_, bght::hip_deleter<atomic_pair_type>());

  d_build_success_ = std::allocator_traits<bool_allocator_type>::allocate(bool_allocator_, 1);
  build_success_   = std::shared_ptr<bool>(d_build_success_, bght::hip_deleter<bool>());

  hf0_ = hasher(0);
  hf1_ = hasher(1);
  hfp_ = hasher(2);

  clear();
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          class Allocator,
          int B,
          int Threshold>
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iht(const iht& other)
    : capacity_(other.capacity_)
    , sentinel_key_(other.sentinel_key_)
    , sentinel_value_(other.sentinel_value_)
    , allocator_(other.allocator_)
    , atomic_pairs_allocator_(other.atomic_pairs_allocator_)
    , bool_allocator_(other.bool_allocator_)
    , d_table_(other.d_table_)
    , table_(other.table_)
    , d_build_success_(other.d_build_success_)
    , build_success_(other.build_success_)
    , hfp_(other.hfp_)
    , hf0_(other.hf0_)
    , hf1_(other.hf1_)
    , num_buckets_(other.num_buckets_) {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::~iht() {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
void iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::clear() {
  value_type empty_pair{sentinel_key_, sentinel_value_};
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (capacity_ + block_size - 1) / block_size;
  hipLaunchKernelGGL(detail::kernels::fill,
                     dim3(num_blocks),
                     dim3(block_size),
                     0,
                     0,
                     d_table_,
                     d_table_ + capacity_,
                     empty_pair);
  bool success = true;
  hip_try(hipMemcpy(d_build_success_, &success, sizeof(bool), hipMemcpyHostToDevice));
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
template <typename InputIt>
bool iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::insert(InputIt first,
                                                                         InputIt last,
                                                                         hipStream_t stream) {
  const auto num_keys       = std::distance(first, last);
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
  hipLaunchKernelGGL(detail::kernels::tiled_insert_kernel,
                     dim3(num_blocks),
                     dim3(block_size),
                     0,
                     stream,
                     first,
                     last,
                     *this);
  bool success;
  hip_try(hipMemcpyAsync(&success, d_build_success_, sizeof(bool), hipMemcpyDeviceToHost, stream));
  return success;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
template <typename InputIt, typename OutputIt>
void iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::find(InputIt first,
                                                                       InputIt last,
                                                                       OutputIt output_begin,
                                                                       hipStream_t stream) {
  const auto num_keys       = last - first;
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;

  hipLaunchKernelGGL(detail::kernels::tiled_find_kernel,
                     dim3(num_blocks),
                     dim3(block_size),
                     0,
                     stream,
                     first,
                     last,
                     output_begin,
                     *this);
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          class Allocator,
          int B,
          int Threshold>
template <typename tile_type>
__device__ bool bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::insert(
    value_type const& pair,
    tile_type const& tile) {
  auto primary_bucket    = hfp_(pair.first) % num_buckets_;
  auto lane_id           = tile.thread_rank();
  const int elected_lane = 0;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};

  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;
  bool print        = false;
  // if (pair.first == 62) print = true;
  int load = 0;
  bucket_type bucket(&d_table_[primary_bucket * bucket_size], tile);
  if (threshold_ > 0) {
    bucket.load(std::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    load = bucket.compute_load(sentinel_pair);
    if (print) bucket.print(primary_bucket);
  }
  do {
    if (load >= threshold_) {
      // Secondary hashing scheme
      // Using Double hashing
      auto bucket_id = hf0_(pair.first) % num_buckets_;
      auto step_size = (hf1_(pair.first) % (capacity_ / bucket_size - 1) + 1);
      while (true) {
        bucket = bucket_type(&d_table_[bucket_id * bucket_size], tile);
        bucket.load(std::memory_order_relaxed);
        INCREMENT_PROBES_IN_TILE
        load = bucket.compute_load(sentinel_pair);
        if (print) bucket.print(bucket_id);

        while (load < bucket_size) {
          bool cas_success = false;
          if (lane_id == elected_lane) {
            cas_success = bucket.strong_cas_at_location(
                pair, load, sentinel_pair, std::memory_order_relaxed, std::memory_order_relaxed);
          }
          cas_success = tile.shfl(cas_success, elected_lane);
          if (print && tile.thread_rank() == 0)
            printf("cas_success @ %i = %i\n", load, cas_success);
          if (cas_success) { return true; }
          load++;
        }
        bucket_id = (bucket_id + step_size) % num_buckets_;
      }
    } else {
      bool cas_success = false;
      if (lane_id == elected_lane) {
        cas_success = bucket.strong_cas_at_location(
            pair, load, sentinel_pair, std::memory_order_relaxed, std::memory_order_relaxed);
      }
      cas_success = tile.shfl(cas_success, elected_lane);
      if (print && tile.thread_rank() == 0) printf("cas_success @ %i = %i\n", load, cas_success);
      if (print) bucket.print(primary_bucket);

      if (cas_success) { return true; }
      load++;
    }
  } while (true);
  return false;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          class Allocator,
          int B,
          int Threshold>
template <typename tile_type>
__device__ typename bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::mapped_type
bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::find(key_type const& key,
                                                                        tile_type const& tile) {
  auto bucket_id    = hfp_(key) % num_buckets_;
  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;

  // primary hash function
  bucket_type cur_bucket(&d_table_[bucket_id * bucket_size], tile);
  bool print = false;
  // if (key == 8) print = true;
  if (threshold_ > 0) {
    cur_bucket.load(std::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    int key_location = cur_bucket.find_key_location(key, key_equal{});
    if (print) cur_bucket.print(bucket_id);
    if (print) printf("key_location %i\n", key_location);
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location);
      if (print) printf("key_location %i -> %i\n", key_location, found_value);
      return found_value;
    }
  }

  // double-hashing
  bucket_id           = hf0_(key) % num_buckets_;
  auto initial_bucket = bucket_id;
  auto step_size      = (hf1_(key) % (capacity_ / bucket_size - 1) + 1);
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  do {
    cur_bucket = bucket_type(&d_table_[bucket_id * bucket_size], tile);
    cur_bucket.load(std::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    int key_location = cur_bucket.find_key_location(key, key_equal{});
    if (print) printf("Trying doublehashing\n");
    if (print) cur_bucket.print(bucket_id);
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location);
      return found_value;
    } else if (cur_bucket.compute_load(sentinel_pair) < bucket_size) {
      if (print)
        printf("sentinel_value_ %i -> %i, %i\n",
               bucket_id,
               cur_bucket.compute_load(sentinel_pair),
               bucket_size);
      return sentinel_value_;
    }
    bucket_id = (bucket_id + step_size) % num_buckets_;
    if (print) printf("bucket_id %i -> %i\n", bucket_id, initial_bucket);
    if (bucket_id == initial_bucket) break;
  } while (true);

  return sentinel_value_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
template <typename RNG>
void iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::randomize_hash_functions(
    RNG& rng) {
  hfp_ = initialize_hf<hasher>(rng);
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::size_type
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::size(hipStream_t stream) {
  return 0;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::const_iterator
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::begin() const {
  return d_table_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::const_iterator
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::end() const {
  return d_table_ + capacity_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          detail::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::size_type
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::max_size() const {
  return capacity_;
}
}  // namespace bght
