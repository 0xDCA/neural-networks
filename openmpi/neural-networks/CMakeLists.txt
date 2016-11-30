cmake_minimum_required(VERSION 3.5.1)
project(openmpi-nn-lib)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
)

set(SOURCE_FILES OpenmpiFeedforwardNeuralNetwork.cpp OpenmpiFeedforwardNeuralNetwork.h
  eigen-util.h eigen-util.cpp)

add_library(openmpi-nn-lib ${SOURCE_FILES})

target_link_libraries(
  openmpi-nn-lib
  nn-common
  boost_mpi
  boost_serialization
)
