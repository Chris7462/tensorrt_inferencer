// C++ standard library includes
#include <chrono>
#include <iostream>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"


class TensorRTInferencerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    inferencer_ = std::make_shared<TensorRTInferencer>(
      engine_path, input_height, input_width, num_classes);
  }

  void TearDown() override
  {
  }

  std::shared_ptr<TensorRTInferencer> inferencer_;
  const std::string engine_path = "fcn_resnet50_model_1238x374.trt";
  const std::string image_path = "image_000.png";
  const int input_width = 1238;
  const int input_height = 374;
  const int num_classes = 21;
};

// Pascal VOC colormap (optimized with lookup table)
cv::Mat decode_segmentation_fast(
  const std::vector<float> & output_data,
  int height, int width, int num_classes = 21)
{
  static const std::vector<cv::Vec3b> colormap = {
    {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0}, {0, 0, 128},
    {128, 0, 128}, {0, 128, 128}, {128, 128, 128}, {64, 0, 0}, {192, 0, 0},
    {64, 128, 0}, {192, 128, 0}, {64, 0, 128}, {192, 0, 128}, {64, 128, 128},
    {192, 128, 128}, {0, 64, 0}, {128, 64, 0}, {0, 192, 0}, {128, 192, 0},
    {0, 64, 128}
  };

  cv::Mat seg_map(height, width, CV_8UC3);
  const float * data = output_data.data();

  // Vectorized argmax operation
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int pixel_idx = y * width + x;

      // Find max class efficiently
      int max_class = 0;
      float max_val = data[pixel_idx];

      for (int c = 1; c < num_classes; ++c) {
        float val = data[c * height * width + pixel_idx];
        if (val > max_val) {
          max_val = val;
          max_class = c;
        }
      }

      seg_map.at<cv::Vec3b>(y, x) = colormap[max_class];
    }
  }

  return seg_map;
}

TEST_F(TensorRTInferencerTest, TestInitialization)
{
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    throw std::runtime_error("Failed to load image: " + image_path);
  }
  EXPECT_EQ(image.cols, 1238);
  EXPECT_EQ(image.rows, 374);

  auto start = std::chrono::high_resolution_clock::now();
  auto output = inferencer_->infer(image);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start);

  std::cout << "Single inference time: " << duration.count() << " ms" << std::endl;

    // Decode and visualize
  cv::Mat seg_color = decode_segmentation_fast(output, input_height, input_width, num_classes);
  cv::resize(seg_color, seg_color, image.size(), 0, 0, cv::INTER_NEAREST);

  cv::Mat overlay;
  cv::addWeighted(image, 0.5, seg_color, 0.5, 0, overlay);

  cv::imshow("Original", image);
  cv::imshow("Segmentation", seg_color);
  cv::imshow("Overlay", overlay);
  cv::waitKey(0);
}
