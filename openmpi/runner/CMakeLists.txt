cmake_minimum_required(VERSION 3.5.1)
project(openmpi-nn-runner)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/openmpi/neural-networks
)

set(SOURCE_FILES main.cpp)
add_executable(openmpi-nn-runner ${SOURCE_FILES})

target_link_libraries(
    openmpi-nn-runner
    openmpi-nn-lib
    nn-common
)

add_dependencies(openmpi-nn-runner openmpi-nn-lib)
