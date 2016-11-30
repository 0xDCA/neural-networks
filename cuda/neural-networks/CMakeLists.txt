cmake_minimum_required(VERSION 3.4)

project(cuda-nn-lib)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/cuda-nn-lib)

cuda_add_library(
  cuda-nn-lib
  cuda-helpers.h
  cuda-helpers.cu
  CudaFeedforwardNeuralNetwork.h
  CudaFeedforwardNeuralNetwork.cpp
)

target_link_libraries(
  cuda-nn-lib
  nn-common
)
