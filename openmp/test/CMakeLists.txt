cmake_minimum_required(VERSION 3.4)

project(openmp-nn-test)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/openmp/neural-networks
    ${CMAKE_SOURCE_DIR}/test-common
)

add_executable(
    openmp-nn-test
    main.cpp
)

target_link_libraries(
    openmp-nn-test
    openmp-nn-lib
    nn-common
    nn-test-common
)

add_dependencies(openmp-nn-test openmp-nn-lib nn-common nn-test-common)
