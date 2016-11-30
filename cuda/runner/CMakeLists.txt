cmake_minimum_required(VERSION 3.4)

project(cuda-nn-runner)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/cuda-nn-runner)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --cudart static
)

cuda_add_executable(
    cuda-nn-runner
    main.cpp
)

target_link_libraries(
    cuda-nn-runner
    cuda-nn-lib
    ${CUDA_LIBRARY}
)

add_dependencies(cuda-nn-runner cuda-nn-lib)
