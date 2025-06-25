cmake_minimum_required(VERSION 3.8)
project(tensorrt_inferencer VERSION 1.0.0)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Find TensorRT
set(TENSORRT_ROOT /usr/local/tensorrt)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

# Check if TensorRT was found
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT to the TensorRT installation directory.")
endif()

# Define the library target
add_library(tensorrt_inferencer STATIC
  src/tensorrt_inferencer.cpp)

# Specify include directories for the target
target_include_directories(tensorrt_inferencer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TENSORRT_INCLUDE_DIRS}  # Make this PUBLIC so downstream packages can use it
    ${OpenCV_INCLUDE_DIRS})
    #${CUDAToolkit_INCLUDE_DIRS}  # Not needed as CUDA::cudart in the target_link_libraries will provide CUDA includes for Modern CMake

# Link OpenCV libraries to the target
target_link_libraries(tensorrt_inferencer
  PUBLIC
    CUDA::cudart  # This automatically provides CUDA includes. Make this PUBLIC so downstream packages can use it
  PRIVATE
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${OpenCV_LIBS})

target_compile_features(tensorrt_inferencer PUBLIC cxx_std_17)

# Optionally install the library
install(TARGETS tensorrt_inferencer
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV CUDAToolkit)

# Export include directories for downstream packages
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_tensorrt_inferencer
    test/test_tensorrt_inferencer.cpp)

  target_include_directories(test_tensorrt_inferencer PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

  target_link_libraries(test_tensorrt_inferencer
    tensorrt_inferencer
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    CUDA::cudart
    gtest_main)

endif()

ament_package()
