cmake_minimum_required(VERSION 3.5.1)
project(openmp-nn-runner)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/openmp/neural-networks
)

set(SOURCE_FILES main.cpp)
add_executable(openmp-nn-runner ${SOURCE_FILES})

target_link_libraries(
    openmp-nn-runner
    openmp-nn-lib
    nn-common
)

add_dependencies(openmp-nn-runner openmp-nn-lib)
