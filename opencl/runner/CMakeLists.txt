cmake_minimum_required(VERSION 3.4)

project(opencl-nn-runner)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/opencl-nn-runner)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --cudart static
)

cuda_add_executable(
    opencl-nn-runner
    main.cpp
)

target_link_libraries(
    opencl-nn-runner
    opencl-nn-lib
    ${CUDA_LIBRARY}
)

add_dependencies(opencl-nn-runner opencl-nn-lib)
