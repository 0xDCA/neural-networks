cmake_minimum_required(VERSION 3.4)

project(cuda-nn-test)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/cuda-nn-test)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --cudart static
)

include_directories(
  ${CMAKE_SOURCE_DIR}/test-common
)

cuda_add_executable(
    cuda-nn-test
    main.cpp
)

target_link_libraries(
    cuda-nn-test
    cuda-nn-lib
    nn-test-common
    ${CUDA_LIBRARY}
)

add_dependencies(cuda-nn-test cuda-nn-lib)
