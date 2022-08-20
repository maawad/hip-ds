
#pragma once
#include <cassert>
#include <vector>

#include <hip_helpers.hpp>

#include "hip/hip_runtime.h"

template <typename T>
struct hip_array {
  hip_array(const std::size_t size) : size_(size), ptr_(nullptr) { allocate(); }
  hip_array(const std::size_t size, const T value) : size_(size), ptr_(nullptr) {
    allocate();
    set_value(value);
  }
  hip_array() : size_(0), ptr_(nullptr) {}
  std::size_t size() const { return size_; }
  // todo: don't clear the array
  void resize(std::size_t new_size) {
    free();
    size_ = new_size;
    allocate();
    set_value(0);
  }
  // copy constructor
  hip_array<T>(const hip_array<T>&) = delete;
  hip_array<T>(const std::vector<T>& input) {
    size_ = input.size();
    allocate();
    from_std_vector(input);
  }
  // move constructor
  hip_array<T>(hip_array<T>&&) = delete;
  // move assignment operator
  hip_array<T>& operator=(hip_array<T>&&) = delete;
  // copy assignment operator
  hip_array<T>& operator=(hip_array<T>&) = delete;
  hip_array<T>& operator                 =(const std::vector<T>& input) {
    free();
    size_ = input.size();
    allocate();
    from_std_vector(input);
    return *this;
  };

  void copy_to_host(T* input) {
    hip_try(hipMemcpy(input, ptr_, sizeof(T) * size_, hipMemcpyDeviceToHost));
  }

  void free() {
    if (ptr_) hip_try(hipFree(ptr_));
  }

  ~hip_array<T>() {}

  const T* data() const { return ptr_; }
  T* data() { return ptr_; }

  std::vector<T> to_std_vector() {
    assert(ptr_ != nullptr);
    std::vector<T> h_cpy(size_, static_cast<T>(0));
    auto raw_ptr = h_cpy.data();
    copy_to_host(raw_ptr);
    return h_cpy;
  }

 private:
  void set_value(const T value) { hip_try(hipMemset(ptr_, value, sizeof(T) * size_)); }
  void allocate() { hip_try(hipMalloc((void**)&ptr_, sizeof(T) * size_)); }
  void copy_to_device(const T* input) {
    hip_try(hipMemcpy(ptr_, input, sizeof(T) * size_, hipMemcpyHostToDevice));
  }
  void from_std_vector(const std::vector<T>& input) {
    // make sure everything is correct
    assert(input.size() == size_);
    assert(ptr_ != nullptr);
    copy_to_device(input.data());
  }

  std::size_t size_;
  T* ptr_;
};