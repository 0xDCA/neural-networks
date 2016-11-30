cmake_minimum_required(VERSION 3.5.1)
project(nn-common)

add_library(nn-test-common
  ApproxMatrixMatcher.h
  ApproxMatrixMatcher.cpp
  nn-test.h
  nn-test.cpp
)
