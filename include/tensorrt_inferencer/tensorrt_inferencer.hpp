#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <limits>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/core.hpp>


namespace tensorrt_inferencer
{

// Custom exception classes
class TensorRTException : public std::runtime_error
{
public:
  explicit TensorRTException(const std::string & message)
  : std::runtime_error("TensorRT Error: " + message) {}
};

class CudaException : public std::runtime_error
{
public:
  explicit CudaException(const std::string & message, cudaError_t error)
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

  void log(Severity severity, const char * msg) noexcept override;

private:
  Severity min_severity_;
};

// Memory management helper
class CudaMemoryManager
{
public:
  static void * allocate_device(size_t size);
  static void * allocate_host_pinned(size_t size);
  static void free_device(void * ptr);
  static void free_host_pinned(void * ptr);
};

// Optimized TensorRT inference class
class TensorRTInferencer
{
public:
  struct Config
  {
    /**
     * @brief Input image height
     */
    int height;

    /**
     * @brief Input image width
     */
    int width;
    /**
     * @brief Number of output classes
     * @details This should match the number of classes in your model.
     * - For Pascal VOC, this is 21 (plus background).
     */
    int num_classes;

    /**
     * @brief Number of CUDA streams used for pipelined execution
     * @details Determines how many CUDA streams are available for asynchronous inference.
     * - When using the synchronous `infer()` method, set this to 1, as images are processed sequentially.
     * - Increase to 2-4 streams when using `infer_async()` to:
     *   - Enable concurrent processing of multiple images.
     *   - Overlap preprocessing and postprocessing with inference for better performance.
     */
    int num_streams;

    /**
     * @brief Number of warmup iterations before timing starts
     * @details This is used to ensure that the CUDA kernels and GPU resources are properly initialized
     * and cached before actual inference timing begins. This helps to avoid cold start penalties.
     * - The first iteration initializes CUDA kernels and allocates any lazy GPU resources.
     * - The second iteration ensures everything is properly warmed up and gives more consistent timing.
     * - Set to 0 to disable warmup iterations.
     */
    int warmup_iterations;

    /**
     * @brief Log level for TensorRT messages
     * @details This controls the verbosity of TensorRT logging.
     */
    Logger::Severity log_level;

    /**
     * @brief Default constructor
     * @details Initializes the configuration with default values.
     */
    Config()
    : height(374), width(1238), num_classes(21), num_streams(1),
      warmup_iterations(2), log_level(Logger::Severity::kWARNING) {}
  };

  // Constructor with configuration
  explicit TensorRTInferencer(const std::string & engine_path, const Config & config = Config());

  // Destructor
  ~TensorRTInferencer();

  // Disable copy and move semantics - use std::unique_ptr for ownership transfer
  TensorRTInferencer(const TensorRTInferencer &) = delete;
  TensorRTInferencer & operator=(const TensorRTInferencer &) = delete;
  TensorRTInferencer(TensorRTInferencer &&) = delete;
  TensorRTInferencer & operator=(TensorRTInferencer &&) = delete;

  // Main inference method
  std::vector<float> infer(const cv::Mat & image);

    // Utility functions
  cv::Mat decode_segmentation(const std::vector<float> & output_data) const;
  cv::Mat create_overlay(
    const cv::Mat & original, const cv::Mat & segmentation,
    float alpha = 0.5f) const;

  // Performance monitoring
  struct PerformanceStats
  {
    double avg_inference_time_ms;
    double min_inference_time_ms;
    double max_inference_time_ms;
    size_t total_inferences;
    PerformanceStats()
    : avg_inference_time_ms(0.0),
      min_inference_time_ms(std::numeric_limits<double>::max()),
      max_inference_time_ms(0.0),
      total_inferences(0) {}
  };

  PerformanceStats get_performance_stats() const {return perf_stats_;}
  void reset_performance_stats();

private:
  // Initialization methods
  void initialize_engine(const std::string & engine_path);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();
  void warmup();

  // Memory management
  void cleanup();

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  cudaStream_t get_next_stream() const;
  void preprocess_image_optimized(const cv::Mat & image, float * output) const;
  void update_performance_stats(double inference_time_ms) const;

  // Validation methods
  void validate_image(const cv::Mat & image) const;

private:
  // Configuration
  Config config_;

  // TensorRT objects
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Tensor information
  std::string input_name_;
  std::string output_name_;
  size_t input_size_;
  size_t output_size_;

  // Memory buffers
  struct MemoryBuffers
  {
    float * pinned_input;
    float * pinned_output;
    void * device_input;
    void * device_output;

    MemoryBuffers()
    : pinned_input(nullptr), pinned_output(nullptr),
      device_input(nullptr), device_output(nullptr) {}
  } buffers_;

  // CUDA streams for pipelining
  std::vector<cudaStream_t> streams_;
  // Stream management
  mutable std::mutex stream_mutex_;
  mutable size_t current_stream_ = 0;

  // Performance tracking
  mutable PerformanceStats perf_stats_;
  mutable std::mutex perf_mutex_;
};

} // namespace tensorrt_inferencer
