#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace tensorrt_inferencer
{

// Host function wrapper for CUDA kernel
void launch_normalize_kernel(
  const float *input_data,
  float *output_data,
  int width, int height,
  const float *mean_values,
  const float *std_values,
  cudaStream_t stream);

} // namespace tensorrt_inferencer
