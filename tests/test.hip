#include <cstdint>
#include <iostream>
#include <vector>

#include <benchmark_helpers.hpp>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <hip_array.hpp>
#include <iht.hpp>

int main(int argc, char** argv) {
  auto arguments   = std::vector<std::string>(argv, argv + argc);
  auto num_keys    = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1ul);
  auto load_factor = get_arg_value<float>(arguments, "load-factor").value_or(0.7);

  hip_device device;
  device.select_device(0);
  device.print_device_properties();

  using key_type   = uint32_t;
  using value_type = uint32_t;
  using pair_type  = bght::pair<key_type, value_type>;

  std::size_t capacity = num_keys / load_factor;

  auto invalid_key   = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<value_type>::max();

  std::cout << "Constructing a table with";
  std::cout << "\n num_keys: " << num_keys;
  std::cout << "\n load_factor: " << load_factor;
  std::cout << "\n capacity: " << capacity;
  std::cout << "\n";

  bght::iht<key_type, value_type> mmap(capacity, invalid_key, invalid_value);

  std::cout << "Preparing keys\n";
  std::vector<key_type> h_keys;
  std::vector<key_type> h_values;
  benchmark::generate_uniform_unique_keys(h_keys, num_keys);

  std::vector<pair_type> h_pairs(num_keys);
  auto to_pair = [](key_type x) { return pair_type{x, x * 10}; };
  std::transform(h_keys.begin(), h_keys.end(), h_pairs.begin(), to_pair);

  hip_array d_pairs   = h_pairs;
  hip_array d_queries = h_keys;
  hip_array d_results(num_keys, invalid_value);

  std::cout << "Building the map\n";

  gpu_timer insertion_timer;
  insertion_timer.start_timer();
  bool result = mmap.insert(d_pairs.begin(), d_pairs.end());
  insertion_timer.stop_timer();
  if (!result) { std::cout << "Failed to build the hash map\n"; }
  std::cout << "query the map\n";

  gpu_timer find_timer;
  find_timer.start_timer();
  mmap.find(d_queries.begin(), d_queries.end(), d_results.begin());
  find_timer.stop_timer();

  auto h_queries = d_queries.to_std_vector();
  auto h_results = d_results.to_std_vector();

  for (std::size_t i = 0; i < h_results.size(); i++) {
    auto expected = to_pair(h_queries[i]).second;
    auto found    = h_results[i];
    if (expected != found) {
      std::cout << i << ": key[" << h_queries[i] << "]: ";
      std::cout << "Expected: ";
      std::cout << expected << ", ";
      std::cout << "Found: ";
      std::cout << found << "\n";
      return 0;
    }
  }
  std::cout << "OK\n";

  std::cout << "Insertion rate (Mkey/s): "
            << static_cast<float>(num_keys) / 1.e6 / insertion_timer.get_elapsed_s() << "\n";

  std::cout << "find rate (Mkey/s): "
            << static_cast<float>(num_keys) / 1.e6 / find_timer.get_elapsed_s() << "\n";

  return 0;
}