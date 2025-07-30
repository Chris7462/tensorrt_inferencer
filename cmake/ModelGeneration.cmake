# ModelGeneration.cmake
# This file handles automatic ONNX and TensorRT engine generation

# Create directories for models and engines
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/engines)

# Find Python3
find_program(PYTHON3_EXECUTABLE python3 REQUIRED)
if(NOT PYTHON3_EXECUTABLE)
  message(FATAL_ERROR "Python3 not found. Please install Python3.")
endif()

# Find trtexec
find_program(TRTEXEC_EXECUTABLE trtexec
  HINTS ${TENSORRT_ROOT}/bin ${CUDA_TOOLKIT_ROOT_DIR}/bin
  PATHS /usr/local/bin /usr/bin)
if(NOT TRTEXEC_EXECUTABLE)
  message(FATAL_ERROR "trtexec not found. Please ensure TensorRT is properly installed and trtexec is in PATH.")
endif()

# Custom target to generate ONNX model
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/fcn_resnet101_374x1238.onnx
  COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/script/export_fcn_to_onnx.py
          --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/models
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/script/export_fcn_to_onnx.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating ONNX model..."
  VERBATIM
)

# Custom target to generate TensorRT engine
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/engines/fcn_resnet101_374x1238.engine
  COMMAND ${TRTEXEC_EXECUTABLE}
          --onnx=${CMAKE_CURRENT_SOURCE_DIR}/models/fcn_resnet101_374x1238.onnx
          --saveEngine=${CMAKE_CURRENT_SOURCE_DIR}/engines/fcn_resnet101_374x1238.engine
          --memPoolSize=workspace:4096
          --fp16
          --verbose
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/models/fcn_resnet101_374x1238.onnx
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating TensorRT engine..."
  VERBATIM
)

# Create custom targets that can be built
add_custom_target(generate_onnx
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/models/fcn_resnet101_374x1238.onnx
)

add_custom_target(generate_engine
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/engines/fcn_resnet101_374x1238.engine
)

# Function to add model generation dependency to a target
function(add_model_generation_dependency target_name)
  add_dependencies(${target_name} generate_engine)
endfunction()

# Install generated models and engines
# Check if directories exist before installing
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models/
    DESTINATION share/${PROJECT_NAME}/models
    FILES_MATCHING PATTERN "*.onnx")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/engines)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/engines/
    DESTINATION share/${PROJECT_NAME}/engines
    FILES_MATCHING PATTERN "*.engine")
endif()
