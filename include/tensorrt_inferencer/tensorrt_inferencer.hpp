#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <cstdint>
#include <array>
#include <stdexcept>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>


namespace config
{
  // ImageNet normalization constants
  constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
  constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

  // Pascal VOC colors for visualization
  constexpr std::array<std::array<uint8_t, 3>, 21> PASCAL_VOC_COLORMAP = {{
    {0, 0, 0},       // Background
    {128, 0, 0},     // Aeroplane
    {0, 128, 0},     // Bicycle
    {128, 128, 0},   // Bird
    {0, 0, 128},     // Boat
    {128, 0, 128},   // Bottle
    {0, 128, 128},   // Bus
    {128, 128, 128}, // Car
    {64, 0, 0},      // Cat
    {192, 0, 0},     // Chair
    {64, 128, 0},    // Cow
    {192, 128, 0},   // Dining table
    {64, 0, 128},    // Dog
    {192, 0, 128},   // Horse
    {64, 128, 128},  // Motorbike
    {192, 128, 128}, // Person
    {0, 64, 0},      // Potted plant
    {128, 64, 0},    // Sheep
    {0, 192, 0},     // Sofa
    {128, 192, 0},   // Train
    {0, 64, 128},    // TV monitor
  }};
}

// Custom exception classes
class TensorRTException : public std::runtime_error
{
public:
  explicit TensorRTException(const std::string& message)
    : std::runtime_error("TensorRT Error: " + message) {}
};

class CudaException : public std::runtime_error
{
public:
  explicit CudaException(const std::string& message, cudaError_t error)
    : std::runtime_error("CUDA Error: " + message + " (" + cudaGetErrorString(error) + ")") {}
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      throw CudaException(#call, error); \
    } \
  } while(0)

  // TensorRT Logger with configurable severity
class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(Severity min_severity = Severity::kWARNING)
    : min_severity_(min_severity) {}

  void log(Severity severity, const char* msg) noexcept override;

private:
  Severity min_severity_;
};

// Memory management helper
class CudaMemoryManager
{
public:
  static void * allocate_device(size_t size);
  static void * allocate_host_pinned(size_t size);
  static void free_device(void* ptr);
  static void free_host_pinned(void* ptr);
};


//#pragma once

//// C++ standard library version: This project uses the C++17 standard library.
//#include <memory>
//#include <stdexcept>
//#include <string>
//#include <vector>

//// TensorRT includes
//#include <NvInfer.h>

//// OpenCV includeso
//#include <opencv2/core.hpp>


//// Normalization constants (ImageNet)
//const std::vector<double> mean = {0.485, 0.456, 0.406};
//const std::vector<double> stddev = {0.229, 0.224, 0.225};

//// TensorRT Logger
//class Logger : public nvinfer1::ILogger
//{
//public:
//  void log(Severity severity, const char * msg) noexcept override;
//};

//// Optimized inference class with streaming
//class TensorRTInferencer
//{
//public:
//  TensorRTInferencer() = delete;
//  explicit TensorRTInferencer(
//    const std::string & engine_path, int height, int width, int classes);
//  ~TensorRTInferencer();

//  // Disable copy and move semantics
//  TensorRTInferencer(const TensorRTInferencer &) = delete;
//  TensorRTInferencer & operator=(const TensorRTInferencer &) = delete;
//  TensorRTInferencer(TensorRTInferencer &&) = delete;
//  TensorRTInferencer & operator=(TensorRTInferencer &&) = delete;

//  std::vector<float> infer(const cv::Mat & image);

//private:
//  std::unique_ptr<nvinfer1::IRuntime> runtime_;
//  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
//  std::unique_ptr<nvinfer1::IExecutionContext> context_;

//  // Multiple streams for pipelining
//  static const int NUM_STREAMS = 2;
//  cudaStream_t streams_[NUM_STREAMS];

//  // Pinned memory for faster transfers
//  float * pinned_input_;
//  float * pinned_output_;

//  // GPU memory buffers
//  void * gpu_input_;
//  void * gpu_output_;

//  std::string input_name_, output_name_;
//  size_t input_size_, output_size_;
//  int input_height_, input_width_, num_classes_;

//  // Stream index for round-robin
//  int current_stream_;

//private:
//  // Helper functions
//  void find_tensor_names();
//  std::vector<uint8_t> load_engine_file(const std::string & engine_path);
//  void warmup();
//  void preprocess_image_optimized(const cv::Mat & image, float * output);
//};
