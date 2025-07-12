#include <iostream>
#include <fstream>
#include <chrono>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files: This project includes local header files.
#include "tensorrt_inferencer/config.hpp"
#include "tensorrt_inferencer/exception.hpp"
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"
#include "tensorrt_inferencer/normalize_kernel.hpp"


namespace tensorrt_inferencer
{

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
float * CudaMemoryManager::allocate_device(size_t size)
{
  float * ptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

float * CudaMemoryManager::allocate_host_pinned(size_t size)
{
  float * ptr;
  CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}

void CudaMemoryManager::free_device(float * ptr)
{
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
  }
}

void CudaMemoryManager::free_host_pinned(float * ptr)
{
  if (ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
    ptr = nullptr;
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
  output_size_ = 1 * config_.num_classes * config_.height * config_.width * sizeof(float);

  // Allocate pinned host memory
  buffers_.pinned_input = CudaMemoryManager::allocate_host_pinned(input_size_);
  buffers_.pinned_output = CudaMemoryManager::allocate_host_pinned(output_size_);

  // Allocate device memory
  buffers_.device_input = CudaMemoryManager::allocate_device(input_size_);
  buffers_.device_output = CudaMemoryManager::allocate_device(output_size_);

  // Allocate and initialize GPU memory for normalization constants
  buffers_.device_mean = CudaMemoryManager::allocate_device(3 * sizeof(float));
  buffers_.device_std = CudaMemoryManager::allocate_device(3 * sizeof(float));

  // Copy mean and std values to GPU (one-time initialization)
  float h_mean[3] = {config::MEAN[0], config::MEAN[1], config::MEAN[2]};
  float h_std[3] = {config::STDDEV[0], config::STDDEV[1], config::STDDEV[2]};

  CUDA_CHECK(cudaMemcpy(buffers_.device_mean, h_mean, 3 * sizeof(float),
    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers_.device_std, h_std, 3 * sizeof(float),
    cudaMemcpyHostToDevice));

  // Set tensor addresses
  if (!context_->setTensorAddress(input_name_.c_str(),
    static_cast<void *>(buffers_.device_input)))
  {
    throw TensorRTException("Failed to set input tensor address");
  }
  if (!context_->setTensorAddress(output_name_.c_str(),
    static_cast<void *>(buffers_.device_output)))
  {
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

  std::cout << "Engine warmed up with " << config_.warmup_iterations << " iterations" << std::endl;
}

void TensorRTInferencer::cleanup()
{
  // Free pinned host memory
  CudaMemoryManager::free_host_pinned(buffers_.pinned_input);
  CudaMemoryManager::free_host_pinned(buffers_.pinned_output);

  // Free device memory
  CudaMemoryManager::free_device(buffers_.device_input);
  CudaMemoryManager::free_device(buffers_.device_output);
  CudaMemoryManager::free_device(buffers_.device_mean);
  CudaMemoryManager::free_device(buffers_.device_std);

  // Destroy streams
  for (auto & stream : streams_) {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  streams_.clear();
}

std::vector<float> TensorRTInferencer::infer(const cv::Mat & image)
{
  cudaStream_t stream = get_next_stream();

  // Preprocess directly into pinned memory
  preprocess_image(image, buffers_.pinned_input);

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

  // Convert to vector
  size_t num_elements = output_size_ / sizeof(float);
  std::vector<float> result(buffers_.pinned_output,
    buffers_.pinned_output + num_elements);

  return result;
}

std::vector<float> TensorRTInferencer::infer_gpu(const cv::Mat & image)
{
  cudaStream_t stream = get_next_stream();

  // Preprocess directly into pinned memory
  preprocess_image_cuda(image, buffers_.device_input, stream);

  // Run inference
  if (!context_->enqueueV3(stream)) {
    throw TensorRTException("Failed to enqueue inference");
  }

  // Async copy result back
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_output, buffers_.device_output,
    output_size_, cudaMemcpyDeviceToHost, stream));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Convert to vector
  size_t num_elements = output_size_ / sizeof(float);
  std::vector<float> result(buffers_.pinned_output,
    buffers_.pinned_output + num_elements);

  return result;
}

cv::Mat TensorRTInferencer::decode_segmentation(const std::vector<float> & output_data) const
{
  cv::Mat seg_map(config_.height, config_.width, CV_8UC3);
  const float * data = output_data.data();

  // Optimized argmax with vectorization hints
  for (int y = 0; y < config_.height; ++y) {
    for (int x = 0; x < config_.width; ++x) {
      int pixel_idx = y * config_.width + x;

      // Find class with maximum probability
      int max_class = 0;
      float max_val = data[pixel_idx];

      for (int c = 1; c < config_.num_classes; ++c) {
        float val = data[c * config_.height * config_.width + pixel_idx];
        if (val > max_val) {
          max_val = val;
          max_class = c;
        }
      }

      // Apply colormap
      const auto & color = config::PASCAL_VOC_COLORMAP[max_class];
      seg_map.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]); // BGR
    }
  }

  return seg_map;
}

cv::Mat TensorRTInferencer::create_overlay(
  const cv::Mat & original, const cv::Mat & segmentation, float alpha) const
{
  cv::Mat overlay;
  cv::Mat seg_resized;

  // Resize segmentation to match original image size
  cv::resize(segmentation, seg_resized, original.size(), 0, 0, cv::INTER_NEAREST);

  // Create weighted overlay
  cv::addWeighted(original, 1.0f - alpha, seg_resized, alpha, 0, overlay);

  return overlay;
}

cudaStream_t TensorRTInferencer::get_next_stream() const
{
  std::lock_guard<std::mutex> lock(stream_mutex_);
  cudaStream_t stream = streams_[current_stream_];
  current_stream_ = (current_stream_ + 1) % config_.num_streams;
  return stream;
}

void TensorRTInferencer::preprocess_image(
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

// Much simpler CUDA preprocessing - follows the same pattern as CPU version
void TensorRTInferencer::preprocess_image_cuda(
  const cv::Mat & image, float * output, cudaStream_t stream) const
{
  // Step 1: Resize image using OpenCV (on CPU)
  cv::Mat img_resized;
  cv::resize(image, img_resized, cv::Size(config_.width, config_.height));

  // Step 2: Convert to float (on CPU)
  img_resized.convertTo(img_resized, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  size_t image_size = img_resized.rows * img_resized.cols * img_resized.channels() * sizeof(float);
  float * gpu_image_data = CudaMemoryManager::allocate_device(image_size);
  CUDA_CHECK(cudaMemcpyAsync(gpu_image_data, img_resized.data,
    image_size, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch simple normalization kernel
  launch_normalize_kernel(
    gpu_image_data,
    output,
    config_.width, config_.height,
    buffers_.device_mean, buffers_.device_std,
    stream);

  // Step 5: Cleanup temporary image data
  CudaMemoryManager::free_device(gpu_image_data);
}

} // namespace tensorrt_inferencer
