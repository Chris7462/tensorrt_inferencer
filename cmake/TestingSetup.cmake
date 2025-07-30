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

# Create symbolic link to image file in build directory for testing
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/image_000.png
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/test/image_000.png
          ${CMAKE_CURRENT_BINARY_DIR}/image_000.png
  DEPENDS fcn_trt_backend # Same as above
  COMMENT "Creating symbolic link to image file for testing..."
)

# Custom target for the symbolic link
add_custom_target(test_image_link
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/image_000.png
)
