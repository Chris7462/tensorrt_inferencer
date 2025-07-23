#include "fcn_trt_backend/config.hpp"


namespace fcn_trt_backend
{

// Declare constant memory (visible to all kernels in this compilation unit)
__constant__ float d_mean[3];
__constant__ float d_std[3];

// Initialize constant memory (call once during initialization)
void initialize_mean_std_constants()
{
  // Copy mean values
  float h_mean[3] = {config::MEAN[0], config::MEAN[1], config::MEAN[2]};
  cudaMemcpyToSymbol(d_mean, h_mean, 3 * sizeof(float));

  // Copy std values
  float h_std[3] = {config::STDDEV[0], config::STDDEV[1], config::STDDEV[2]};
  cudaMemcpyToSymbol(d_std, h_std, 3 * sizeof(float));
}

// Simple CUDA kernel for normalization (assumes image is already resized)
__global__ void normalize_kernel(
  const float *input_data,
  float *output_data,
  int width, int height)
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
    float normalized = (pixel_value - d_mean[c]) / d_std[c];

    // Store in CHW format (channel-first)
    output_data[c * height * width + pixel_idx] = normalized;
  }
}

// Host function wrapper
void launch_normalize_kernel(
  const float *input_data,
  float *output_data,
  int width, int height,
  cudaStream_t stream)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y);

  normalize_kernel<<<gridSize, blockSize, 0, stream>>>(
    input_data, output_data,
    width, height);
}

} // namespace fcn_trt_backend
