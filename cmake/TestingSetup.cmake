# TestingSetup.cmake
# This file handles creating symbolic link to engine and image file for testing

# Create symbolic link to engine file in build directory for testing
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fcn_resnet101_374x1238.engine
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/engines/fcn_resnet101_374x1238.engine
          ${CMAKE_CURRENT_BINARY_DIR}/fcn_resnet101_374x1238.engine
  DEPENDS fcn_trt_backend  # Depend on the main library target so it won't build before the engine is generated
  COMMENT "Creating symbolic link to engine file for testing..."
)

# Custom target for the symbolic link
add_custom_target(test_engine_link
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fcn_resnet101_374x1238.engine
)

# List of image files to link
set(TEST_IMAGE_FILES
  image_000.png
  image_001.png
  image_002.png
)

# Initialize an empty list to collect output files
set(TEST_IMAGE_OUTPUTS)

# Loop to create a symbolic link for each image file in build directory for testing
foreach(image_file IN LISTS TEST_IMAGE_FILES)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/test/${image_file})
  set(dst ${CMAKE_CURRENT_BINARY_DIR}/${image_file})

  add_custom_command(
    OUTPUT ${dst}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dst}
    DEPENDS fcn_trt_backend
    COMMENT "Creating symbolic link to ${image_file} for testing..."
  )

  list(APPEND TEST_IMAGE_OUTPUTS ${dst})
endforeach()

# Custom target that depends on all symbolic links
add_custom_target(test_image_link
  DEPENDS ${TEST_IMAGE_OUTPUTS}
)

# Function to add model generation dependency to a target
function(add_testing_dependency target_name)
  add_dependencies(${target_name} test_engine_link test_image_link)
endfunction()
