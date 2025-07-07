// local includes
#include "tensorrt_inferencer/exception.hpp"
#include "tensorrt_inferencer/decode_argmax_gpu.hpp"


namespace tensorrt_inferencer
{

__global__ void argmax_kernel(const float* input, unsigned char* output,
  int C, int H, int W)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // height

  if (x >= W || y >= H) return;

  int max_class = 0;
  int pixel_idx = y * W + x;
  float max_val = input[pixel_idx];  // c = 0

  // Loop through channels with stride for better memory access
  for (int c = 1; c < C; ++c) {
    int idx = c * H * W + pixel_idx;
    float val = input[idx];
    if (val > max_val) {
      max_val = val;
      max_class = c;
      }
  }

    output[pixel_idx] = static_cast<unsigned char>(max_class);
}

void decode_argmax_gpu(const cv::cuda::GpuMat& scores, cv::cuda::GpuMat& class_ids,
  int C, int H, int W, cudaStream_t stream)
{
  // Same validation as above...
  const float* input_ptr = scores.ptr<float>();
  //const float* input_ptr = reinterpret_cast<const float*>(scores.ptr<float>());
  class_ids.create(H, W, CV_8UC1);
  unsigned char* output_ptr = class_ids.ptr<unsigned char>();

  // Optimized launch configuration
  dim3 block(32, 8);  // Still 256 threads, but better for memory access
  dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

  argmax_kernel<<<grid, block, 0, stream>>>(input_ptr, output_ptr, C, H, W);

  CUDA_CHECK(cudaGetLastError());
  if (stream == 0) {
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

} // namespace tensorrt_inferencer
