#include <iostream>
#include <fstream>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"


// TensorRT Logger
void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::cerr << msg << std::endl;
  }
}

TensorRTInferencer::TensorRTInferencer(
  const std::string & engine_path, int height, int width, int classes)
: input_height_(height), input_width_(width), num_classes_(classes), current_stream_(0)
{
  Logger logger;

  // Load engine
  auto engine_data = load_engine_file(engine_path);
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

  // Find tensor names
  find_tensor_names();

  // Calculate sizes
  input_size_ = 1 * 3 * input_height_ * input_width_ * sizeof(float);
  output_size_ = 1 * num_classes_ * input_height_ * input_width_ * sizeof(float);

  // Create streams
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamCreate(&streams_[i]);
  }

  // Allocate pinned memory for faster CPU-GPU transfers
  cudaMallocHost((void **)&pinned_input_, input_size_);
  cudaMallocHost((void **)&pinned_output_, output_size_);

  // Allocate GPU memory
  cudaMalloc(&gpu_input_, input_size_);
  cudaMalloc(&gpu_output_, output_size_);

  // Set tensor addresses
  context_->setTensorAddress(input_name_.c_str(), gpu_input_);
  context_->setTensorAddress(output_name_.c_str(), gpu_output_);

  // Warm up the engine
  warmup();
}

TensorRTInferencer::~TensorRTInferencer()
{
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamDestroy(streams_[i]);
  }
  cudaFreeHost(pinned_input_);
  cudaFreeHost(pinned_output_);
  cudaFree(gpu_input_);
  cudaFree(gpu_output_);
}

std::vector<uint8_t> TensorRTInferencer::load_engine_file(const std::string & engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  file.read(reinterpret_cast<char *>(buffer.data()), size);
  return buffer;
}

void TensorRTInferencer::find_tensor_names()
{
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = tensor_name;
    }
  }
}

void TensorRTInferencer::warmup()
{
  // Create dummy input
  cv::Mat dummy_image = cv::Mat::zeros(374, 1238, CV_8UC3);

  // Run a few inferences to warm up
  for (int i = 0; i < 3; ++i) {
    infer(dummy_image);
  }
  std::cout << "Engine warmed up" << std::endl;
}

std::vector<float> TensorRTInferencer::infer(const cv::Mat & image)
{
  cudaStream_t stream = streams_[current_stream_];
  current_stream_ = (current_stream_ + 1) % NUM_STREAMS;

  // Preprocess directly into pinned memory
  preprocess_image_optimized(image, pinned_input_);

  // Async copy to GPU
  cudaMemcpyAsync(gpu_input_, pinned_input_, input_size_, cudaMemcpyHostToDevice, stream);

  // Run inference
  context_->enqueueV3(stream);

  // Async copy result back
  cudaMemcpyAsync(pinned_output_, gpu_output_, output_size_, cudaMemcpyDeviceToHost, stream);

  // Wait for completion
  cudaStreamSynchronize(stream);

  // Convert to vector
  std::vector<float> result(pinned_output_, pinned_output_ + (output_size_ / sizeof(float)));

  return result;
}

void TensorRTInferencer::preprocess_image_optimized(const cv::Mat & image, float * output)
{
  cv::Mat img_resized;
  cv::resize(image, img_resized, cv::Size(input_width_, input_height_));

  // Convert to float and normalize in one step
  img_resized.convertTo(img_resized, CV_32FC3, 1.0 / 255.0);

  // Split channels
  std::vector<cv::Mat> channels(3);
  cv::split(img_resized, channels);

  // Normalize and copy directly to output buffer
  for (int c = 0; c < 3; ++c) {
    cv::Mat normalized = (channels[c] - mean[c]) / stddev[c];
    std::memcpy(output + c * input_height_ * input_width_,
      normalized.data, input_height_ * input_width_ * sizeof(float));
  }
}
