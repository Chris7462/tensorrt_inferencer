#include <cuda_runtime.h>

#include "tensorrt_inferencer/exception.hpp"


namespace tensorrt_inferencer
{

// decode_segmentation_gpu.cu
__global__ void decode_and_colorize_kernel(
  const float* input, uchar3* output, const uchar3* color_map,
  int width, int height, int num_classes)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  // Argmax
  int best_class = 0;
  float max_score = input[idx];
  for (int c = 1; c < num_classes; ++c) {
    float score = input[c * width * height + idx];
    if (score > max_score) {
      max_score = score;
      best_class = c;
    }
  }

  output[idx] = color_map[best_class];
}

void launch_decode_and_colorize_kernel(
  const float* input_gpu, uchar3* output_gpu,
  const uchar3* color_map_gpu, int width, int height, int num_classes,
  cudaStream_t stream)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y);

  decode_and_colorize_kernel<<<gridSize, blockSize, 0, stream>>>(
    input_gpu, output_gpu, color_map_gpu,
    width, height, num_classes);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_inferencer
