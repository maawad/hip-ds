#include <cstdint>
#include <iostream>
#include <vector>

#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <hip_array.hpp>
#include <iht.hpp>

template <typename T>
__global__ void memcpy_kernel(const T* input, T* output, std::size_t count) {
  auto thread_id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (thread_id < count) { output[thread_id] = input[thread_id]; }
}

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  auto count     = get_arg_value<uint32_t>(arguments, "count").value_or(1ul);

  hip_device device;
  device.select_device(0);
  device.print_device_properties();

  using K              = uint32_t;
  using V              = uint32_t;
  std::size_t capacity = 1024;

  auto invalid_key   = std::numeric_limits<K>::max();
  auto invalid_value = std::numeric_limits<V>::max();

  bght::iht<K, V> test(capacity, invalid_key, invalid_value);

  return 0;
}