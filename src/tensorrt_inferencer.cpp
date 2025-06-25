#include <iostream>
#include <fstream>
#include <chrono>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files: This project includes local header files.
#include "tensorrt_inferencer/config.hpp"
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"


// Logger implementation
void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= min_severity_) {
    const char * severity_str;
    switch (severity) {
      case Severity::kINTERNAL_ERROR: severity_str = "INTERNAL_ERROR"; break;
      case Severity::kERROR: severity_str = "ERROR"; break;
      case Severity::kWARNING: severity_str = "WARNING"; break;
      case Severity::kINFO: severity_str = "INFO"; break;
      case Severity::kVERBOSE: severity_str = "VERBOSE"; break;
      default: severity_str = "UNKNOWN"; break;
    }
    std::cerr << "[TensorRT " << severity_str << "] " << msg << std::endl;
  }
}

// CudaMemoryManager implementation
void * CudaMemoryManager::allocate_device(size_t size)
{
  void * ptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

void * CudaMemoryManager::allocate_host_pinned(size_t size)
{
  void * ptr;
  CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}

void CudaMemoryManager::free_device(void * ptr)
{
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

void CudaMemoryManager::free_host_pinned(void * ptr)
{
  if (ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
}

// TensorRTInferencer implementation
TensorRTInferencer::TensorRTInferencer(const std::string & engine_path, const Config & config)
: config_(config)
{
  try {
    // Initialize logger
    logger_ = std::make_unique<Logger>(config_.log_level);

    // Initialize engine and context
    initialize_engine(engine_path);

    // Find tensor information
    find_tensor_names();

    // Initialize memory and streams
    initialize_memory();
    initialize_streams();

    // Warm up the engine
    warmup();

  } catch (const std::exception & e) {
    cleanup();
    throw TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

TensorRTInferencer::~TensorRTInferencer()
{
  cleanup();
}

void TensorRTInferencer::initialize_engine(const std::string & engine_path)
{
  auto engine_data = load_engine_file(engine_path);

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw TensorRTException("Failed to create TensorRT runtime");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw TensorRTException("Failed to deserialize CUDA engine");
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext());
  if (!context_) {
    throw TensorRTException("Failed to create execution context");
  }
}

std::vector<uint8_t> TensorRTInferencer::load_engine_file(
  const std::string & engine_path) const
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read engine file: " + engine_path);
  }

  return buffer;
}

void TensorRTInferencer::find_tensor_names()
{
  bool found_input = false, found_output = false;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;
      found_input = true;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = tensor_name;
      found_output = true;
    }
  }

  if (!found_input || !found_output) {
    throw TensorRTException("Failed to find input or output tensor");
  }
}

void TensorRTInferencer::initialize_memory()
{
  // Calculate memory sizes
  input_size_ = 1 * 3 * config_.height * config_.width * sizeof(float);
  output_size_ = 1 * config_.classes * config_.height * config_.width * sizeof(float);

  // Allocate pinned host memory
  buffers_.pinned_input = static_cast<float *>(
    CudaMemoryManager::allocate_host_pinned(input_size_));
  buffers_.pinned_output = static_cast<float *>(
    CudaMemoryManager::allocate_host_pinned(output_size_));

  // Allocate device memory
  buffers_.device_input = CudaMemoryManager::allocate_device(input_size_);
  buffers_.device_output = CudaMemoryManager::allocate_device(output_size_);

  // Set tensor addresses
  if (!context_->setTensorAddress(input_name_.c_str(), buffers_.device_input)) {
    throw TensorRTException("Failed to set input tensor address");
  }
  if (!context_->setTensorAddress(output_name_.c_str(), buffers_.device_output)) {
    throw TensorRTException("Failed to set output tensor address");
  }
}

void TensorRTInferencer::initialize_streams()
{
  streams_.resize(config_.num_streams);
  for (int i = 0; i < config_.num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
}

void TensorRTInferencer::warmup()
{
  cv::Mat dummy_image = cv::Mat::zeros(config_.height, config_.width, CV_8UC3);

  for (int i = 0; i < config_.warmup_iterations; ++i) {
    infer(dummy_image);
  }

  // Reset performance stats after warmup
  reset_performance_stats();

  std::cout << "Engine warmed up with " << config_.warmup_iterations
            << " iterations" << std::endl;
}

void TensorRTInferencer::cleanup()
{
  // Destroy streams
  for (auto & stream : streams_) {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  streams_.clear();

  // Free memory
  CudaMemoryManager::free_host_pinned(buffers_.pinned_input);
  CudaMemoryManager::free_host_pinned(buffers_.pinned_output);
  CudaMemoryManager::free_device(buffers_.device_input);
  CudaMemoryManager::free_device(buffers_.device_output);

  buffers_ = {};
}

std::vector<float> TensorRTInferencer::infer(const cv::Mat & image)
{
  validate_image(image);

  auto start_time = std::chrono::high_resolution_clock::now();

  cudaStream_t stream = get_next_stream();

  // Preprocess directly into pinned memory
  preprocess_image_optimized(image, buffers_.pinned_input);

  // Async copy to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_input, buffers_.pinned_input,
    input_size_, cudaMemcpyHostToDevice, stream));

  // Run inference
  if (!context_->enqueueV3(stream)) {
    throw TensorRTException("Failed to enqueue inference");
  }

  // Async copy result back
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_output, buffers_.device_output,
    output_size_, cudaMemcpyDeviceToHost, stream));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
  update_performance_stats(duration.count());

  // Convert to vector
  size_t num_elements = output_size_ / sizeof(float);
  std::vector<float> result(buffers_.pinned_output,
    buffers_.pinned_output + num_elements);

  return result;
}

void TensorRTInferencer::validate_image(const cv::Mat & image) const
{
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }
  if (image.type() != CV_8UC3) {
    throw std::invalid_argument("Input image must be CV_8UC3 format");
  }
}

cudaStream_t TensorRTInferencer::get_next_stream() const
{
  std::lock_guard<std::mutex> lock(stream_mutex_);
  cudaStream_t stream = streams_[current_stream_];
  current_stream_ = (current_stream_ + 1) % config_.num_streams;
  return stream;
}

void TensorRTInferencer::preprocess_image_optimized(
  const cv::Mat & image, float * output) const
{
  cv::Mat img_resized;
  cv::resize(image, img_resized, cv::Size(config_.width, config_.height));

  // Convert to float and normalize in one step
  img_resized.convertTo(img_resized, CV_32FC3, 1.0f / 255.0f);

  // Split channels
  std::vector<cv::Mat> channels(3);
  cv::split(img_resized, channels);

  // Normalize each channel and copy to output buffer
  for (int c = 0; c < 3; ++c) {
    cv::Mat normalized = (channels[c] - config::MEAN[c]) / config::STDDEV[c];
    std::memcpy(output + c * config_.height * config_.width,
      normalized.data, config_.height * config_.width * sizeof(float));
  }
}

void TensorRTInferencer::reset_performance_stats()
{
  std::lock_guard<std::mutex> lock(perf_mutex_);
  perf_stats_ = PerformanceStats{};
}

void TensorRTInferencer::update_performance_stats(double inference_time_ms) const
{
  std::lock_guard<std::mutex> lock(perf_mutex_);

  perf_stats_.total_inferences++;
  perf_stats_.min_inference_time_ms =
    std::min(perf_stats_.min_inference_time_ms, inference_time_ms);
  perf_stats_.max_inference_time_ms =
    std::max(perf_stats_.max_inference_time_ms, inference_time_ms);

  // Update running average
  double delta = inference_time_ms - perf_stats_.avg_inference_time_ms;
  perf_stats_.avg_inference_time_ms += delta / perf_stats_.total_inferences;
}
