//// C++ standard library includes
//#include <chrono>
//#include <iostream>
//#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "tensorrt_inferencer/config.hpp"

#define private public
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"
#undef private


class TensorRTInferencerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Configure the inferencer
    TensorRTInferencer::Config conf;
    conf.height = input_height_;
    conf.width = input_width_;
    conf.classes = num_classes_;
    conf.num_streams = 1;
    conf.warmup_iterations = 2;
    conf.log_level = Logger::Severity::kINFO;

    try {
      inferencer_ = std::make_unique<TensorRTInferencer>(engine_path_, conf);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize TensorRT inferencer: " << e.what();
    }
  }

  void TearDown() override
  {
    // Performance stats will be printed automatically
    if (inferencer_) {
      auto stats = inferencer_->get_performance_stats();
      std::cout << "\n=== Performance Statistics ===" << std::endl;
      std::cout << "Total inferences: " << stats.total_inferences << std::endl;
      std::cout << "Average time: " << stats.avg_inference_time_ms << " ms" << std::endl;
      std::cout << "Min time: " << stats.min_inference_time_ms << " ms" << std::endl;
      std::cout << "Max time: " << stats.max_inference_time_ms << " ms" << std::endl;
    }
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

  std::shared_ptr<TensorRTInferencer> inferencer_;

private:
  const std::string engine_path_ = "fcn_resnet50_1238x374.trt";
  const std::string image_path_ = "image_000.png";
  const int input_width_ = 1238;
  const int input_height_ = 374;
  const int num_classes_ = 21;
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
  size_t expected_size = inferencer_->config_.classes * inferencer_->config_.height *
    inferencer_->config_.width;
  EXPECT_EQ(output.size(), expected_size);

  // Check that output contains valid probabilities
  bool has_valid_values = std::any_of(output.begin(), output.end(),
      [](float val) {return std::isfinite(val);});
  EXPECT_TRUE(has_valid_values);

//  // Decode and visualize
//cv::Mat seg_color = decode_segmentation_fast(output, input_height, input_width, num_classes);
//cv::resize(seg_color, seg_color, image.size(), 0, 0, cv::INTER_NEAREST);

//cv::Mat overlay;
//cv::addWeighted(image, 0.5, seg_color, 0.5, 0, overlay);

//cv::imshow("Original", image);
//cv::imshow("Segmentation", seg_color);
//cv::imshow("Overlay", overlay);
//cv::waitKey(0);
}
