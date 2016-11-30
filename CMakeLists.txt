cmake_minimum_required(VERSION 3.4)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(
  APPEND CMAKE_CXX_FLAGS
  "-std=c++11"
)

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/lib
)

link_directories(
  ${CMAKE_SOURCE_DIR}/build/lib
)

add_subdirectory(common)
add_subdirectory(test-common)
add_subdirectory(cuda)
add_subdirectory(openmp)
