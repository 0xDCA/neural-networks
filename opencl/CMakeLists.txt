cmake_minimum_required(VERSION 3.4)

set(PLATFORM $(shell uname -s))
if ($(PLATFORM) EQUAL Darwin)
    set(LIBS "-framework OpenCL")
endif()

list(
  APPEND CMAKE_CXX_FLAGS
  ${LIBS}
)

include_directories(
  ${CMAKE_SOURCE_DIR}/opencl/neural-networks
  lib
)

add_subdirectory(neural-networks)
add_subdirectory(runner)
add_subdirectory(test)
