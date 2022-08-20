#pragma once

#include <cstdint>
#include <hip_helpers.hpp>
#include <iostream>

struct foo {
  foo()     = default;
  foo(foo&) = delete;
  int do_something(uint32_t k) { return 1; }
};
