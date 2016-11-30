cmake_minimum_required(VERSION 3.4)

find_package(CUDA QUIET REQUIRED)

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -gencode arch=compute_60,code=sm_60
)

include_directories(
  ${CMAKE_SOURCE_DIR}/cuda/neural-networks
  lib
)

set(CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(neural-networks)
add_subdirectory(runner)
add_subdirectory(test)
