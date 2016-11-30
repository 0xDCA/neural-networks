#include "CudaFeedforwardNeuralNetwork.h"
#include <stdexcept>
#include <algorithm>
#include "cuda-helpers.h"


using dc::ManagedMatrix;
using std::runtime_error;

CudaFeedforwardNeuralNetwork::CudaFeedforwardNeuralNetwork(const std::vector<int>& layer_neurons) :
	layer_neurons(layer_neurons) {
	if (layer_neurons.size() < 2) {
		throw runtime_error("Neural networks need at least an input layer and an output layer");
	}

	if (!std::all_of(layer_neurons.begin(), layer_neurons.end(), [](int element) { return element > 0; })) {
		throw runtime_error("All layers must have a positive number of units");
	}

	weights.reserve(layer_neurons.size() - 1);
	for (int i = 0; i < static_cast<int>(layer_neurons.size()) - 1; ++i) {
		weights.push_back(ManagedMatrix<double>(layer_neurons[i + 1], layer_neurons[i] + 1));
	}
}

std::vector<std::vector<double> > CudaFeedforwardNeuralNetwork::get_activations(const std::vector<double>& input) const {
	return dc_internal::forward_propagate(layer_neurons, input, weights);
}

std::vector<double> CudaFeedforwardNeuralNetwork::predict(const std::vector<double>& input) const {
	return get_activations(input).back();
}

std::vector<dc::ManagedMatrix<double> > CudaFeedforwardNeuralNetwork::compute_weights_error(
		const std::vector<double>& input, const std::vector<double>& output) const {
	return dc_internal::back_propagate(layer_neurons, input, output, weights);
}

TrainResult CudaFeedforwardNeuralNetwork::train(const dc::ManagedMatrix<double>& x,
		const dc::ManagedMatrix<double>& y, const TrainSettings& train_settings) {
	if (x.get_columns() != y.get_columns()) {
		throw runtime_error("x must have the same examples as y");
	}

	if (x.get_rows() != layer_neurons[0]) {
		throw runtime_error("x must have as much rows as neurons in the input layer");
	}

	if (y.get_rows() != layer_neurons[layer_neurons.size() - 1]) {
		throw runtime_error("y must have as much rows as neurons in the output layer");
	}

	train_settings.validate();

	if (train_settings.initialize_weights_randomly) {
		// Randomly initialize weights
		for(size_t i = 0; i < weights.size(); ++i) {
			fill_matrix_randomly(weights[i], train_settings.random_epsilon, *train_settings.generator);
		}
	}

	/*std::cout << "Before:\n";
	for(auto w : weights) {
		std::cout << w;
	}*/

	auto result = dc_internal::train(layer_neurons, x, y, train_settings, weights);

	/*std::cout << "After:\n";
	for(auto w : weights) {
		std::cout << w;
	}*/

	return result;
}

dc::ManagedMatrix<double> CudaFeedforwardNeuralNetwork::get_weights(int source_layer) const {
	if (source_layer < 0 || source_layer > layer_neurons.size()) {
		throw std::runtime_error("Invalid layer number");
	}

	return weights[source_layer];
}

void CudaFeedforwardNeuralNetwork::set_weights(int source_layer, const dc::ManagedMatrix<double> &weights) {
	if (source_layer < 0 || source_layer > layer_neurons.size()) {
		throw std::runtime_error("Invalid layer number");
	}

	int cols = this->weights[source_layer].get_columns();
	int rows = this->weights[source_layer].get_rows();
	if (weights.get_columns() != cols || weights.get_rows() != rows) {
		throw std::runtime_error("Weights dimensions mismatch");
	}

	this->weights[source_layer] = weights;
}

void CudaFeedforwardNeuralNetwork::fill_matrix_randomly(dc::ManagedMatrix<double>& matrix,
		double epsilon, std::mt19937& generator) {
	std::uniform_real_distribution<> distribution(-epsilon, epsilon);

	for(int i = 0; i < matrix.get_columns(); ++i) {
		for(int j = 0; j < matrix.get_rows(); ++j) {
			matrix.set(j, i, distribution(generator));
		}
	}
}

double CudaFeedforwardNeuralNetwork::compute_error(const dc::ManagedMatrix<double>& x,
		const dc::ManagedMatrix<double>& y, double regularization_term) {
	if (x.get_columns() != y.get_columns()) {
		throw runtime_error("Different number of x and y examples");
	}

	if (x.get_rows() != layer_neurons[0]) {
		throw runtime_error("x must have as much rows as neurons in the input layer");
	}

	if (y.get_rows() != layer_neurons[layer_neurons.size() - 1]) {
		throw runtime_error("y must have as much rows as neurons in the output layer");
	}

	double error = 0;
	int examples = x.get_columns();
	for(int i = 0; i < examples; ++i) {
		const double* xi_pointer = x.get_internal_data() + x.get_rows() * i;
		std::vector<double> xi(xi_pointer, xi_pointer + x.get_rows());
		auto prediction = predict(xi);

		double norm = 0;
		for(int j = 0; j < prediction.size(); ++j) {
			norm += (y.get(j, i) - prediction[j]) * (y.get(j, i) - prediction[j]);
		}

		error += norm;
	}

	double complexity_term = 0;
	for (size_t i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights[i].get_rows(); ++j) {
			for (int k = 0; k < weights[i].get_columns(); ++k) {
				complexity_term += weights[i].get(j, k) * weights[i].get(j, k);
			}
		}
	}

	return 1.0 / examples * error + regularization_term / (2.0 * examples) * complexity_term;
}

double CudaFeedforwardNeuralNetwork::compute_error_gpu(const dc::ManagedMatrix<double>& x,
		const dc::ManagedMatrix<double>& y, double regularization_term, int blocks, int threads_per_block) {
	if (x.get_columns() != y.get_columns()) {
		throw runtime_error("Different number of x and y examples");
	}

	if (x.get_rows() != layer_neurons[0]) {
		throw runtime_error("x must have as much rows as neurons in the input layer");
	}

	if (y.get_rows() != layer_neurons[layer_neurons.size() - 1]) {
		throw runtime_error("y must have as much rows as neurons in the output layer");
	}

	return dc_internal::compute_error(layer_neurons, x, y, weights, regularization_term,
			blocks, threads_per_block);
}
