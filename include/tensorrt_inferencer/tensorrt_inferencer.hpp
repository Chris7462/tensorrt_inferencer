#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includeso
#include <opencv2/core.hpp>


// Normalization constants (ImageNet)
const std::vector<double> mean = {0.485, 0.456, 0.406};
const std::vector<double> stddev = {0.229, 0.224, 0.225};

// TensorRT Logger
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

// Optimized inference class with streaming
class TensorRTInferencer
{
public:
  TensorRTInferencer() = delete;
  explicit TensorRTInferencer(
    const std::string & engine_path, int height, int width, int classes);
  ~TensorRTInferencer();

  // Disable copy and move semantics
  TensorRTInferencer(const TensorRTInferencer &) = delete;
  TensorRTInferencer & operator=(const TensorRTInferencer &) = delete;
  TensorRTInferencer(TensorRTInferencer &&) = delete;
  TensorRTInferencer & operator=(TensorRTInferencer &&) = delete;

  std::vector<float> infer(const cv::Mat & image);

private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Multiple streams for pipelining
  static const int NUM_STREAMS = 2;
  cudaStream_t streams_[NUM_STREAMS];

  // Pinned memory for faster transfers
  float * pinned_input_;
  float * pinned_output_;

  // GPU memory buffers
  void * gpu_input_;
  void * gpu_output_;

  std::string input_name_, output_name_;
  size_t input_size_, output_size_;
  int input_height_, input_width_, num_classes_;

  // Stream index for round-robin
  int current_stream_;

private:
  // Helper functions
  void find_tensor_names();
  std::vector<uint8_t> load_engine_file(const std::string & engine_path);
  void warmup();
  void preprocess_image_optimized(const cv::Mat & image, float * output);
};
