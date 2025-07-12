//// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/cudacodec.hpp>  // Temporary

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "tensorrt_inferencer/config.hpp"
#include "tensorrt_inferencer/decode_and_colorize_kernel.hpp"  // Temporary

#define private public
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"
#undef private


class TensorRTInferencerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Configure the inferencer
    tensorrt_inferencer::TensorRTInferencer::Config conf;
    conf.height = input_height_;
    conf.width = input_width_;
    conf.num_classes = num_classes_;
    conf.num_streams = 1;
    conf.warmup_iterations = 2;
    conf.log_level = tensorrt_inferencer::Logger::Severity::kINFO;

    try {
      inferencer_ = std::make_unique<tensorrt_inferencer::TensorRTInferencer>(engine_path_, conf);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize TensorRT inferencer: " << e.what();
    }
  }

  void TearDown() override
  {
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  void save_results(
    const cv::Mat & original, const cv::Mat & segmentation,
    const cv::Mat & overlay, const std::string & suffix = "")
  {
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_segmentation" + suffix + ".png", segmentation);
    cv::imwrite("test_output_overlay" + suffix + ".png", overlay);
  }

  std::shared_ptr<tensorrt_inferencer::TensorRTInferencer> inferencer_;

public:
  const int input_width_ = 1238;
  const int input_height_ = 374;
  const int num_classes_ = 21;

private:
  const std::string engine_path_ = "fcn_resnet50_1238x374.trt";
  const std::string image_path_ = "image_000.png";
};


TEST_F(TensorRTInferencerTest, TestBasicInference)
{
  cv::Mat image = load_test_image();
  EXPECT_EQ(image.cols, 1238);
  EXPECT_EQ(image.rows, 374);
  EXPECT_EQ(image.type(), CV_8UC3);

  auto start = std::chrono::high_resolution_clock::now();
  auto output = inferencer_->infer(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "Single inference time: " << duration.count() << " ms" << std::endl;

  // Validate output
  size_t expected_size = inferencer_->config_.num_classes * inferencer_->config_.height *
    inferencer_->config_.width;
  EXPECT_EQ(output.size(), expected_size);

  // Check that output contains valid probabilities
  bool has_valid_values = std::any_of(output.begin(), output.end(),
      [](float val) {return std::isfinite(val);});
  EXPECT_TRUE(has_valid_values);
}

TEST_F(TensorRTInferencerTest, TestSegmentationDecoding)
{
  cv::Mat image = load_test_image();
  auto start = std::chrono::high_resolution_clock::now();
  auto output = inferencer_->infer(image);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "CPU infer: " << duration.count() << std::endl;

  // Decode segmentation
  auto start1 = std::chrono::high_resolution_clock::now();
  cv::Mat segmentation = inferencer_->decode_segmentation(output);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration<double, std::milli>(end1 - start1);
  std::cout << "Decode time on CPU: " << duration1.count() << std::endl;

  EXPECT_EQ(segmentation.rows, inferencer_->config_.height);
  EXPECT_EQ(segmentation.cols, inferencer_->config_.width);
  EXPECT_EQ(segmentation.type(), CV_8UC3);

  // Create overlay
  cv::Mat overlay = inferencer_->create_overlay(image, segmentation, 0.5f);

  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, segmentation, overlay);

  // Optional: Display results (comment out for automated testing)
  /*
  cv::imshow("Original", image);
  cv::imshow("Segmentation", segmentation);
  cv::imshow("Overlay", overlay);
  cv::waitKey(0);
  cv::destroyAllWindows();
  */
}

TEST_F(TensorRTInferencerTest, TestGPUSegmentationDecoding)
{
  cv::Mat image = load_test_image();

  auto start = std::chrono::high_resolution_clock::now();
  auto output = inferencer_->infer_gpu(image);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "GPU infer: " << duration.count() << std::endl;

  // Decode segmentation
  cv::Mat segmentation = inferencer_->decode_segmentation(output);

  EXPECT_EQ(segmentation.rows, inferencer_->config_.height);
  EXPECT_EQ(segmentation.cols, inferencer_->config_.width);
  EXPECT_EQ(segmentation.type(), CV_8UC3);

  // Create overlay
  cv::Mat overlay = inferencer_->create_overlay(image, segmentation, 0.5f);

  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, segmentation, overlay, "_gpu");

  // Optional: Display results (comment out for automated testing)
  /*
  cv::imshow("Original", image);
  cv::imshow("Segmentation", segmentation);
  cv::imshow("Overlay", overlay);
  cv::waitKey(0);
  cv::destroyAllWindows();
  */
}

TEST_F(TensorRTInferencerTest, TestSegmentationDecodingGPU)
{
  cv::Mat image = load_test_image();
  // Run inference normally (CPU vector result unused here)
  inferencer_->infer(image);

  auto start = std::chrono::high_resolution_clock::now();
  // Allocate GPU buffer for decoded RGB mask
  uchar3 * decoded_mask_gpu = nullptr;
  size_t mask_bytes = input_height_ * input_width_ * sizeof(uchar3);
  cudaMalloc(&decoded_mask_gpu, mask_bytes);

  // Upload Pascal VOC colormap to GPU
  constexpr auto & cmap = config::PASCAL_VOC_COLORMAP;
  std::vector<uchar3> cmap_vec;
  for (const auto & rgb : cmap) {
    cmap_vec.push_back({rgb[2], rgb[1], rgb[0]});  // Convert to uchar3 (BGR)
  }

  uchar3 * color_map_gpu = nullptr;
  cudaMalloc(&color_map_gpu, cmap_vec.size() * sizeof(uchar3));
  cudaMemcpyAsync(color_map_gpu, cmap_vec.data(), cmap_vec.size() * sizeof(uchar3),
    cudaMemcpyHostToDevice, inferencer_->get_next_stream());

  // Launch GPU decode using raw GPU output from the buffers
  tensorrt_inferencer::launch_decode_and_colorize_kernel(
    static_cast<const float *>(inferencer_->buffers_.device_output),
    decoded_mask_gpu,
    color_map_gpu,
    input_width_, input_height_,
    num_classes_,
    inferencer_->get_next_stream()
  );

  // Download result to CPU
  cv::cuda::GpuMat gpu_mask(input_height_, input_width_, CV_8UC3, decoded_mask_gpu);
  cv::Mat segmentation;
  gpu_mask.download(segmentation);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "Decode time on GPU: " << duration.count() << std::endl;

  // Validate output
  EXPECT_EQ(segmentation.rows, input_height_);
  EXPECT_EQ(segmentation.cols, input_width_);
  EXPECT_EQ(segmentation.type(), CV_8UC3);

  // Create overlay
  cv::Mat overlay = inferencer_->create_overlay(image, segmentation, 0.5f);
  EXPECT_EQ(overlay.size(), image.size());

  save_results(image, segmentation, overlay, "_gpu_decode");

  // Cleanup
  cudaFree(decoded_mask_gpu);
  cudaFree(color_map_gpu);
}

TEST_F(TensorRTInferencerTest, TestMultipleInferences)
{
  cv::Mat image = load_test_image();

  const int num_iterations = 10;
  std::vector<double> inference_times;

  for (int i = 0; i < num_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto output = inferencer_->infer(image);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(end - start);
    inference_times.push_back(duration.count());

    // Validate output consistency
    EXPECT_EQ(output.size(),
      inferencer_->config_.num_classes * inferencer_->config_.height * inferencer_->config_.width);
  }

  // Calculate statistics
  double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
    inference_times.size();
  double min_time = *std::min_element(inference_times.begin(), inference_times.end());
  double max_time = *std::max_element(inference_times.begin(), inference_times.end());

  std::cout << "Multiple inference statistics:" << std::endl;
  std::cout << "  Average: " << avg_time << " ms" << std::endl;
  std::cout << "  Min: " << min_time << " ms" << std::endl;
  std::cout << "  Max: " << max_time << " ms" << std::endl;

  // Performance expectations (adjust based on your hardware)
  EXPECT_LT(avg_time, 100.0); // Should be less than 100ms on decent hardware
}

TEST_F(TensorRTInferencerTest, TestBenchmarkInference)
{
  cv::Mat image = load_test_image();

  const int warmup_iterations = 10;
  const int benchmark_iterations = 100;

  // Warmup
  for (int i = 0; i < warmup_iterations; ++i) {
    inferencer_->infer(image);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < benchmark_iterations; ++i) {
    inferencer_->infer(image);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration<double, std::milli>(end - start);

  double avg_time = total_duration.count() / benchmark_iterations;
  double fps = 1000.0 / avg_time;

  std::cout << "Benchmark Results:" << std::endl;
  std::cout << "Iterations: " << benchmark_iterations << std::endl;
  std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
  std::cout << "Average time per inference: " << avg_time << " ms" << std::endl;
  std::cout << "Throughput: " << fps << " FPS" << std::endl;
}

// Test with multiple different images (if available)
TEST_F(TensorRTInferencerTest, DISABLED_TestMultipleImages)
{
  std::vector<std::string> test_images = {
    "image_000.png",
    "image_001.png",
    "image_002.png"
  };

  int successful_tests = 0;

  for (const auto & image_path : test_images) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cout << "Skipping missing image: " << image_path << std::endl;
      continue;
    }

    // Resize to expected dimensions if needed
    if (image.rows != input_height_ || image.cols != input_width_) {
      cv::resize(image, image, cv::Size(input_width_, input_height_));
    }

    try {
      auto output = inferencer_->infer(image);
      auto segmentation = inferencer_->decode_segmentation(output);
      auto overlay = inferencer_->create_overlay(image, segmentation);

      // Save results with image-specific suffix
      std::string suffix = "_" + std::to_string(successful_tests);
      save_results(image, segmentation, overlay, suffix);

      successful_tests++;

    } catch (const std::exception & e) {
      FAIL() << "Failed to process image " << image_path << ": " << e.what();
    }
  }

  EXPECT_GT(successful_tests, 0) << "No test images were successfully processed";
  std::cout << "Successfully processed " << successful_tests << " test images" << std::endl;
}
