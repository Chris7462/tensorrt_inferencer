#pragma once

#include <cuda_runtime.h>


namespace tensorrt_inferencer
{
/**
 * @brief GPU accelerated segmentation decode and colorize kernel
 * @details GPU accelerated segmentation wrapper
 * @param input_gpu      Input logits on GPU: shape [num_classes, height, width]
 * @param output_gpu     Output buffer on GPU: shape [height * width], CV_8UC3
 * @param color_map_gpu  GPU colormap: size [num_classes], each is uchar3
 * @param width          Image width
 * @param height         Image height
 * @param num_classes    Number of segmentation classes
 * @param stream         CUDA stream to launch the kernel on
 */

void launch_decode_and_colorize_kernel(
  const float * input_gpu, uchar3 * output_gpu,
  const uchar3 * color_map_gpu, int width, int height, int num_classes,
  cudaStream_t stream);

} // namespace tensorrt_inferencer
