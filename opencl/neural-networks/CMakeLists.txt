cmake_minimum_required(VERSION 3.4)

project(opencl-nn-lib)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/opencl-nn-lib)

cuda_add_library(
  opencl-nn-lib
  OpenclNeuralNetworkHelper.h
  OpenclNeuralNetworkHelper.cpp
  OpenclFeedforwardNeuralNetwork.h
  OpenclFeedforwardNeuralNetwork.cpp
)

target_link_libraries(
  opencl-nn-lib
  nn-common
)
