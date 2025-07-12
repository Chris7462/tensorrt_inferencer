#include <cuda_runtime.h>

#include "tensorrt_inferencer/exception.hpp"
#include "tensorrt_inferencer/config.hpp"


namespace tensorrt_inferencer
{

// Simple CUDA kernel for normalization (assumes image is already resized)
__global__ void normalize_kernel(
  const float *input_data,
  float *output_data,
  int width, int height,
  const float *mean_values,
  const float *std_values)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixel_idx = y * width + x;

  // Process each channel (BGR -> RGB conversion handled by OpenCV)
  for (int c = 0; c < 3; ++c) {
    // Input is HWC format from OpenCV, output is CHW format
    float pixel_value = input_data[pixel_idx * 3 + c];

    // Normalize: (pixel - mean) / std
    float normalized = (pixel_value - mean_values[c]) / std_values[c];

    // Store in CHW format (channel-first)
    output_data[c * height * width + pixel_idx] = normalized;
  }
}

// Host function wrapper
void launch_normalize_kernel(
  const float *input_data,
  float *output_data,
  int width, int height,
  const float *mean_values,
  const float *std_values,
  cudaStream_t stream)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y);

  normalize_kernel<<<gridSize, blockSize, 0, stream>>>(
    input_data, output_data,
    width, height,
    mean_values, std_values);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_inferencer
