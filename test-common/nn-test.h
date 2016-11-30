#ifndef NNTEST_H_
#define NNTEST_H_

#include "AbstractFeedforwardNeuralNetworkFactory.h"
#include "TrainSettings.h"

void run_common_nn_tests(const AbstractFeedforwardNeuralNetworkFactory& nn_factory, const TrainSettings& train_settings);

#endif /* NNTEST_H_ */
