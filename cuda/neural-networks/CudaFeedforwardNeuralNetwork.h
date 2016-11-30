#ifndef CUDAFEEDFORWARDNEURALNETWORK_H_
#define CUDAFEEDFORWARDNEURALNETWORK_H_

#include <vector>
#include <stdexcept>
#include <exception>
#include <random>
#include "ManagedMatrix.h"
#include "TrainSettings.h"
#include "TrainResult.h"
#include "FeedforwardNeuralNetwork.h"

struct TrainSettings;

class CudaFeedforwardNeuralNetwork : public FeedforwardNeuralNetwork {
public:
    CudaFeedforwardNeuralNetwork(const std::vector<int>& layer_neurons);

    std::vector<std::vector<double> > get_activations(const std::vector<double>& input) const;

    std::vector<double> predict(const std::vector<double>& input) const;

    std::vector<dc::ManagedMatrix<double> > compute_weights_error(const std::vector<double>& input,
    		const std::vector<double>& output) const;

    TrainResult train(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
    		const TrainSettings& train_settings);

    void set_weights(int source_layer, const dc::ManagedMatrix<double> &weights);

    dc::ManagedMatrix<double> get_weights(int source_layer) const;

    double compute_error(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
    		double regularization_term);

    double compute_error_gpu(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
			double regularization_term, int blocks, int threads_per_block);
private:
    std::vector<int> layer_neurons;

    // weights[i] has dimensions layer_neurons[i + 1] * (layer_neurons[i] + 1)
    std::vector<dc::ManagedMatrix<double> > weights;

    static void fill_matrix_randomly(dc::ManagedMatrix<double>& matrix,
    		double epsilon, std::mt19937& generator);
};

#endif /* CUDAFEEDFORWARDNEURALNETWORK_H_ */
