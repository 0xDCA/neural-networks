#ifndef ABSTRACTFEEDFORWARDNEURALNETWORKFACTORY_H_
#define ABSTRACTFEEDFORWARDNEURALNETWORKFACTORY_H_

#include "FeedforwardNeuralNetwork.h"
#include <vector>

class AbstractFeedforwardNeuralNetworkFactory {
public:
  virtual ~AbstractFeedforwardNeuralNetworkFactory() {}

  virtual FeedforwardNeuralNetwork* create_neural_network(const std::vector<int>& layer_neurons) const = 0;
};

#endif /* ABSTRACTFEEDFORWARDNEURALNETWORKFACTORY_H_ */
