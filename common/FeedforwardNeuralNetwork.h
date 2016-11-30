#ifndef FEEDFORWARDNEURALNETWORK_H_
#define FEEDFORWARDNEURALNETWORK_H_

#include <vector>
#include <stdexcept>
#include <exception>
#include <random>
#include "ManagedMatrix.h"
#include "TrainSettings.h"
#include "TrainResult.h"

struct TrainSettings;

class FeedforwardNeuralNetwork {
public:
    virtual ~FeedforwardNeuralNetwork();
    virtual std::vector<std::vector<double> > get_activations(const std::vector<double>& input) const = 0;

    virtual std::vector<double> predict(const std::vector<double>& input) const = 0;

    virtual std::vector<dc::ManagedMatrix<double> > compute_weights_error(const std::vector<double>& input,
    		const std::vector<double>& output) const = 0;

    virtual TrainResult train(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
    		const TrainSettings& train_settings) = 0;

    virtual void set_weights(int source_layer, const dc::ManagedMatrix<double> &weights) = 0;

    virtual dc::ManagedMatrix<double> get_weights(int source_layer) const = 0;

    virtual double compute_error(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
    		double regularization_term) = 0;
};

#endif /* FEEDFORWARDNEURALNETWORK_H_ */
