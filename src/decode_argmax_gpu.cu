#include <iostream>
#include <vector>

// local includes
#include "tensorrt_inferencer/config.hpp"
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

// Optimized version using constant memory for Pascal VOC
__constant__ uchar3 voc_palette[21];

__global__ void colorize_kernel(const unsigned char* class_ids, uchar3* output,
  int H, int W, int num_classes)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // height

  if (x >= W || y >= H) return;

  int idx = y * W + x;
  unsigned char class_id = class_ids[idx];

  // Bounds checking for class_id
  if (class_id < num_classes) {
    output[idx] = voc_palette[class_id];
  } else {
    // Handle invalid class_id - set to black or first color
    output[idx] = make_uchar3(0, 0, 0);  // Black for invalid classes
  }
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
  CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for completion and check execution errors
//if (stream == 0) {
//  CUDA_CHECK(cudaDeviceSynchronize());
//}
}

void colorize_segmentation_gpu(const cv::cuda::GpuMat & class_ids,
  cv::cuda::GpuMat & color_mask, int C,
  cudaStream_t stream)
{
  // Input validation
  if (class_ids.empty()) {
    std::cerr << "Error: Input class_ids matrix is empty" << std::endl;
    return;
  }

  if (class_ids.type() != CV_8UC1) {
    std::cerr << "Error: Input class_ids must be of type CV_8UC1" << std::endl;
    return;
  }

  const int H = class_ids.rows;
  const int W = class_ids.cols;

  // Initialize Pascal VOC palette in constant memory (only once)
  static bool palette_initialized = false;
  if (!palette_initialized) {
    std::vector<uchar3> palette_data(C);
    for (size_t i = 0; i < config::PASCAL_VOC_COLORMAP.size(); ++i) {
      palette_data[i] = make_uchar3(config::PASCAL_VOC_COLORMAP[i][0],
        config::PASCAL_VOC_COLORMAP[i][1],
        config::PASCAL_VOC_COLORMAP[i][2]);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(voc_palette, palette_data.data(),
      C * sizeof(uchar3)));
    palette_initialized = true;
  }

  // Allocate output
  color_mask.create(H, W, CV_8UC3);

  // Launch configuration
  dim3 block(16, 16);
  dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

  colorize_kernel<<<grid, block, 0, stream>>>(
    class_ids.ptr<unsigned char>(),
    reinterpret_cast<uchar3*>(color_mask.ptr<unsigned char>()),
    H, W, C);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for completion and check execution errors
}

} // namespace tensorrt_inferencer
