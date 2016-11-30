cmake_minimum_required(VERSION 3.5.1)
project(openmp-nn-lib)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
)

set(SOURCE_FILES OpenmpFeedforwardNeuralNetwork.cpp OpenmpFeedforwardNeuralNetwork.h
  eigen-util.h eigen-util.cpp)

add_library(openmp-nn-lib ${SOURCE_FILES})

target_link_libraries(
openmp-nn-lib
nn-common
)
