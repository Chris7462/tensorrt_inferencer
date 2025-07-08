#pragma once

#include <opencv2/core/cuda.hpp>

namespace tensorrt_inferencer
{

void decode_argmax_gpu(
  const cv::cuda::GpuMat & scores, cv::cuda::GpuMat & class_ids,
  int C, int H, int W, cudaStream_t stream = 0);

void colorize_segmentation_gpu(
  const cv::cuda::GpuMat & class_ids,
  cv::cuda::GpuMat & color_mask, int C, cudaStream_t stream = 0);

} // namespace tensorrt_inferencer
