cmake_minimum_required(VERSION 3.5.1)
project(nn-common)

add_library(nn-common
  data-util.h
  data-util.cpp
  int-util.h
  int-util.cpp
  ManagedMatrix.h
  TrainResult.h
  TrainSettings.h
  FeedforwardNeuralNetwork.h
  FeedforwardNeuralNetwork.cpp
  AbstractFeedforwardNeuralNetworkFactory.h
  FeedforwardNeuralNetworkClassFactory.h
  runner-common.h
  runner-common.cpp
)
