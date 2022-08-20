#pragma once
#include <hip_helpers.hpp>
#include "hip/hip_runtime.h"

struct gpu_timer {
  gpu_timer(hipStream_t stream = 0) : start_{}, stop_{}, stream_(stream) {
    hip_try(hipEventCreate(&start_));
    hip_try(hipEventCreate(&stop_));
  }
  void start_timer() { hip_try(hipEventRecord(start_, stream_)); }
  void stop_timer() { hip_try(hipEventRecord(stop_, stream_)); }
  float get_elapsed_ms() {
    compute_ms();
    return elapsed_time_;
  }

  float get_elapsed_s() {
    compute_ms();
    return elapsed_time_ * 0.001f;
  }
  ~gpu_timer() {
    hip_try(hipEventDestroy(start_));
    hip_try(hipEventDestroy(stop_));
  };

 private:
  void compute_ms() {
    hip_try(hipEventSynchronize(stop_));
    hip_try(hipEventElapsedTime(&elapsed_time_, start_, stop_));
  }
  hipEvent_t start_, stop_;
  hipStream_t stream_;
  float elapsed_time_ = 0.0f;
};