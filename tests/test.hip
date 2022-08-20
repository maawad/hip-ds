#include <cstdint>
#include <iostream>
#include <vector>

#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <hip_array.hpp>
#include <template_lib.hpp>

template <typename T>
__global__ void memcpy_kernel(const T* input, T* output, std::size_t count) {
  auto thread_id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (thread_id < count) { output[thread_id] = input[thread_id]; }
}

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  auto count     = get_arg_value<uint32_t>(arguments, "count").value_or(1ul);

  foo f;
  f.do_something(count);

  hip_device device;
  device.select_device(0);
  device.print_device_properties();

  using T = uint32_t;
  hip_array<T> input(count, 0xff);
  hip_array<T> output(count, 0);

  const int num_experiments = 1;
  gpu_timer t;
  t.start_timer();
  const uint32_t block_size = 128;
  const uint32_t num_blocks = static_cast<uint32_t>((count + block_size - 1) / block_size);
  for (std::size_t i = 0; i < num_experiments; i++) {
    memcpy_kernel<<<num_blocks, block_size>>>(input.data(), output.data(), count);
  }
  t.stop_timer();
  double elapsed_seconds = t.get_elapsed_s() / static_cast<double>(num_experiments);
  double bytes_count     = static_cast<double>(count * sizeof(T) * 2.0);
  double gbytes_count    = bytes_count / (1ull << 30);

  std::cout << "Transferred (GiBs): " << gbytes_count << std::endl;
  std::cout << "Time (s): " << elapsed_seconds << std::endl;
  std::cout << "Throughput (GiBs/s) " << gbytes_count / elapsed_seconds << std::endl;

  auto result = output.to_std_vector();
  for (std::size_t i = 0; i < result.size(); i++) {
    if (result[i] != 0xffffffff) {
      std::cout << "Copy kernel failed" << std::endl;
      break;
    }
  }
  return 0;
}