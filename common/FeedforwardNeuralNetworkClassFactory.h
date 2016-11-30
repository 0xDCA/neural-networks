#ifndef FEEDFORWARDNEURALNETWORKCLASSFACTORY_H_
#define FEEDFORWARDNEURALNETWORKCLASSFACTORY_H_

#include "FeedforwardNeuralNetwork.h"
#include "AbstractFeedforwardNeuralNetworkFactory.h"
#include <vector>

template <class T>
class FeedforwardNeuralNetworkClassFactory : public AbstractFeedforwardNeuralNetworkFactory {
public:
  virtual FeedforwardNeuralNetwork* create_neural_network(const std::vector<int>& layer_neurons) const {
    return new T(layer_neurons);
  }
};

#endif /* FEEDFORWARDNEURALNETWORKCLASSFACTORY_H_ */
