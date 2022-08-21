#pragma once
#include <cassert>
#include "hip/hip_runtime.h"

#define DEVICE_QUALIFIER __device__

template <typename T>
DEVICE_QUALIFIER unsigned int hip_ffs(unsigned long long int x) {
  return __ffsll(x);
}
DEVICE_QUALIFIER unsigned int hip_ffs(unsigned int x) { return __ffs(x); }

#define hip_try(call)                                                                 \
  do {                                                                                \
    hipError_t err = call;                                                            \
    if (err != hipSuccess) {                                                          \
      printf("HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

struct hip_device {
  hip_device(const int device_id = 0) {
    device_count_ = get_capable_device_count();
    std::cout << "Num capable devices = " << device_count_ << std::endl;
    select_device(device_id);
  }
  int get_capable_device_count() {
    int count = 0;
    hip_try(hipGetDeviceCount(&count));
    return count;
  }
  void select_device(const int device_id) {
    assert(device_id < device_count_);
    hip_try(hipSetDevice(device_id));
    current_device_id_ = device_id;

    hip_try(hipGetDeviceProperties(&current_device_prop_, device_id));
  }

  void print_device_properties() const {
    std::cout << "Name: ";
    std::cout << current_device_prop_.name << std::endl;
    // Memory
    std::cout << "Global memory (GiBs): ";
    std::cout << to_gibs(current_device_prop_.totalGlobalMem) << std::endl;
    std::cout << "Shared memory (MiBs per block): ";
    std::cout << to_mibs(current_device_prop_.sharedMemPerBlock) << std::endl;
    std::cout << "Shared memory size (MiBs): ";
    std::cout << to_mibs(current_device_prop_.totalConstMem) << std::endl;
    std::cout << "L2 cache size : ";  // not sure about unit
    std::cout << to_mibs(current_device_prop_.l2CacheSize) << std::endl;
    std::cout << "Registers per block: ";
    std::cout << current_device_prop_.regsPerBlock << std::endl;
    std::cout << "Memory bus width (bits): ";
    std::cout << current_device_prop_.memoryBusWidth << std::endl;
    std::cout << "Memory clock rate (khz): ";
    std::cout << current_device_prop_.memoryClockRate << std::endl;

    // Compute
    std::cout << "SMs count: ";
    std::cout << current_device_prop_.multiProcessorCount << std::endl;
    std::cout << "Maximum grid size: ";
    std::cout << "(" << current_device_prop_.maxThreadsDim[0];
    std::cout << ", " << current_device_prop_.maxThreadsDim[1];
    std::cout << ", " << current_device_prop_.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Maximum threads per block: ";
    std::cout << current_device_prop_.maxThreadsPerBlock << std::endl;
    std::cout << "Warp size: ";
    std::cout << current_device_prop_.warpSize << std::endl;
    std::cout << "Clock rate (khz): ";
    std::cout << current_device_prop_.clockRate << std::endl;
  }

 private:
  double to_mibs(std::size_t bytes_count) const {
    return static_cast<double>(bytes_count) / static_cast<double>(1ull << 20);
  }
  double to_gibs(std::size_t bytes_count) const {
    return static_cast<double>(bytes_count) / static_cast<double>(1ull << 30);
  }
  int device_count_;
  int current_device_id_;
  hipDeviceProp_t current_device_prop_;
};