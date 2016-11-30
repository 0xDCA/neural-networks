#ifndef RUNNERCOMMON_H_
#define RUNNERCOMMON_H_

#include "AbstractFeedforwardNeuralNetworkFactory.h"

void run_neural_network(const AbstractFeedforwardNeuralNetworkFactory& nn_factory,
   int argc, char* argv[]);

#endif /* RUNNERCOMMON_H_ */
