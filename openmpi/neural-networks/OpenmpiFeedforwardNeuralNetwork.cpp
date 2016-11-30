#include "OpenmpiFeedforwardNeuralNetwork.h"
#include "TrainSettings.h"
#include "eigen-util.h"
#include "util.h"
#include "serialization.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include <boost/serialization/vector.hpp>
#include <boost/mpi.hpp>

using std::runtime_error;
using std::cout;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace mpi = boost::mpi;

OpenmpiFeedforwardNeuralNetwork::OpenmpiFeedforwardNeuralNetwork(const std::vector<int>& layers) : layers(layers) {
    if (layers.size() < 2) {
        throw runtime_error("Neural networks need at least an input layer and an output layer");
    }

    if (!std::all_of(layers.begin(), layers.end(), [](int element) { return element > 0; })) {
        throw runtime_error("All layers must have a positive number of units");
    }

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        weight_list.push_back(MatrixXd::Zero(layers[i + 1], layers[i] + 1));
    }
}

void OpenmpiFeedforwardNeuralNetwork::back_propagation(const std::vector<Eigen::VectorXd> &fp_results,
                                                const Eigen::VectorXd &y,
                                                const std::vector<Eigen::MatrixXd> &weight_list,
                                                std::vector<Eigen::MatrixXd> &out) {
    VectorXd delta_next = fp_results[fp_results.size() - 1] - y;

    for (int i = (int)fp_results.size() - 2; i >= 0; --i) {
        out[i] += delta_next * fp_results[i].transpose();

        MatrixXd current = (weight_list[i].transpose() * delta_next);

        delta_next = current.cwiseProduct(fp_results[i])
                .cwiseProduct(fp_results[i].unaryExpr([](double element) { return 1 - element; }));
        delta_next = delta_next.bottomRows(delta_next.rows() - 1);
    }
}

void OpenmpiFeedforwardNeuralNetwork::back_propagation(const std::vector<Eigen::VectorXd>& fp_results,
                                                const Eigen::VectorXd& y,
                                                std::vector<Eigen::MatrixXd>& out) const {
    back_propagation(fp_results, y, weight_list, out);
}

std::vector<Eigen::VectorXd> OpenmpiFeedforwardNeuralNetwork::forward_propagation(const Eigen::VectorXd &input,
                                                                           const std::vector<Eigen::MatrixXd> &weight_list) {
    if (input.rows() != weight_list[0].cols() - 1){
        std::cout << "Input rows: " << input.rows() << ". Expected: " << (weight_list[0].cols() - 1) <<'\n';
        throw runtime_error("Invalid input size");
    }

    std::vector<VectorXd> result(weight_list.size() + 1);

    VectorXd extended_vector(input.rows() + 1);
    extended_vector << 1, input;
    result[0] = extended_vector;

    for (int i = 0; i < (int)weight_list.size() - 1; ++i) {
        extended_vector = VectorXd(weight_list[i].rows() + 1);
        extended_vector << 1, (weight_list[i] * result[i]).unaryExpr(&sigmoid);
        result[i + 1] = extended_vector;
    }

    int last = (int)result.size() - 1;
    result[last] = (weight_list[last - 1] * result[last - 1]).unaryExpr(&sigmoid);

    return result;
}

std::vector<Eigen::VectorXd> OpenmpiFeedforwardNeuralNetwork::forward_propagation(const Eigen::VectorXd &input) const {
    return forward_propagation(input, this->weight_list);
}

void OpenmpiFeedforwardNeuralNetwork::set_weights_eigen(int source_layer, const Eigen::MatrixXd &weights) {
    if (source_layer < 0 || source_layer >= static_cast<int>(weight_list.size())) {
        throw runtime_error("Invalid layer index");
    }

    if (weights.rows() != weight_list[source_layer].rows() || weights.cols() != weight_list[source_layer].cols()) {
        throw runtime_error("Invalid weight matrix dimension");
    }

    weight_list[source_layer] = weights;
}

Eigen::MatrixXd OpenmpiFeedforwardNeuralNetwork::get_weights_eigen(int source_layer) const {
    if (source_layer < 0 || source_layer >= static_cast<int>(weight_list.size())) {
        throw runtime_error("Invalid layer index");
    }

    return weight_list[source_layer];
}

Eigen::VectorXd OpenmpiFeedforwardNeuralNetwork::predict(const Eigen::VectorXd &input) const {
    std::vector<Eigen::VectorXd> all_results = forward_propagation(input);

    return all_results[all_results.size() - 1];
}

Eigen::MatrixXd OpenmpiFeedforwardNeuralNetwork::random_matrix(int rows, int cols, double epsilon, std::mt19937& generator) {
  std::uniform_real_distribution<> distribution(-epsilon, epsilon);
  MatrixXd result(rows, cols);

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < cols; ++j) {
      result(i, j) = distribution(generator);
    }
  }

  return result;
}

TrainResult OpenmpiFeedforwardNeuralNetwork::train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                                            const TrainSettings& train_settings) {
    if (x.rows() != y.rows()) {
        throw runtime_error("Invalid training data");
    }

    train_settings.validate();

    int rank = world.rank();
    double error;
    int threads = world.size();

    if (rank == 0) {
      // Randomly initialize vectors
      if (train_settings.initialize_weights_randomly) {
        for(size_t i = 0; i < weight_list.size(); ++i) {
          set_weights_eigen(i, OpenmpiFeedforwardNeuralNetwork::random_matrix(
            weight_list[i].rows(), weight_list[i].cols(), train_settings.random_epsilon,
            *train_settings.generator));
          }
        }

        error = compute_error(x, y, train_settings.regularization_term);
    }

    std::random_device rd;

    // Gradient-descent
    for(int it = 1; it <= train_settings.iterations; ++it) {
      std::vector<MatrixXd> new_weights;
      if (rank == 0) {
        new_weights = weight_list;
      }

      broadcast(world, new_weights, 0);
      broadcast(world, error, 0);
      if (error <= train_settings.target_error) {
        return TrainResult(it, error);
      }

      WorkerParams params;
      params.weights = &new_weights;
      params.x = &x;
      params.y = &y;
      params.train_settings = &train_settings;
      params.seed = rd() + rank;

      do_gradient_descent(params);

      if (rank == 0) {
        std::vector<MatrixXd> old_weights = weight_list;
        std::vector<std::vector<MatrixXd> > worker_weights;
        gather(world, new_weights, worker_weights, 0);

        for(int i = 0; i < threads; ++i) {
          for (size_t j = 0; j < weight_list.size(); ++j) {
            weight_list[j] += 1.0 / threads * (worker_weights[i][j] - old_weights[j]);
          }
        }

        error = compute_error(x, y, train_settings.regularization_term);
      } else {
        gather(world, new_weights, 0);
      }
    }

    return TrainResult(train_settings.iterations, error);
}

void OpenmpiFeedforwardNeuralNetwork::do_gradient_descent(const WorkerParams& params) {
    std::vector<MatrixXd> &weights = *params.weights;
    std::vector<MatrixXd> deltas(weights.size()); // layers - 1
    std::vector<MatrixXd> gradients(weights.size());

    for(size_t i = 0; i < gradients.size(); ++i) {
        gradients[i] = MatrixXd::Zero(weights[i].rows(), weights[i].cols());
    }

    int steps = params.train_settings->inner_steps;

    static thread_local std::mt19937 generator(params.seed);
    std::uniform_int_distribution<int> distribution(0, params.x->rows() - 1);

    for (int t = 0; t < steps; ++t) {

        for (size_t k = 0; k < deltas.size(); ++k) {
            deltas[k] = MatrixXd::Zero(weights[k].rows(), weights[k].cols());
        }

        int example = distribution(generator);
        VectorXd xi = params.x->row(example).transpose();
        VectorXd yi = params.y->row(example).transpose();

        std::vector<Eigen::VectorXd> fp_results = forward_propagation(xi, weights);
        back_propagation(fp_results, yi, weights, deltas);

        for (size_t i = 0; i < deltas.size(); ++i) {
            gradients[i] = params.train_settings->momentum * gradients[i] + (1.0 / params.x->rows()) * deltas[i];
            gradients[i].rightCols(gradients[i].cols() - 1) += (params.train_settings->regularization_term / params.x->rows()) * weights[i].rightCols(weights[i].cols() - 1);
            weights[i] -= params.train_settings->step_factor * gradients[i];
        }
    }
}

double OpenmpiFeedforwardNeuralNetwork::compute_error(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double regularization_term,
  const std::vector<Eigen::MatrixXd>& weight_list) {
    if (x.rows() != y.rows()) {
        throw runtime_error("Different number of x and y examples");
    }

    if (y.cols() != weight_list[weight_list.size() - 1].rows()) {
        throw runtime_error("Unexpected number of output classes");
    }

    double error = 0;
    for(int i = 0; i < x.rows(); ++i) {
        std::vector<Eigen::VectorXd> all_results = forward_propagation((x.row(i).transpose()), weight_list);

        auto prediction = all_results[all_results.size() - 1];

        error += pow((y.row(i).transpose() - prediction).norm(), 2);
    }

    double complexity_term = 0;
    for (size_t i = 0; i < weight_list.size(); ++i) {
        for (int j = 0; j < weight_list[i].rows(); ++j) {
            for (int k = 0; k < weight_list[i].cols(); ++k) {
                complexity_term += weight_list[i](j, k) * weight_list[i](j, k);
            }
        }
    }

    return 1.0 / x.rows() * error + regularization_term / (2 * x.rows()) * complexity_term;
}

double OpenmpiFeedforwardNeuralNetwork::compute_error(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                                               double regularization_term) const {
    return compute_error(x, y, regularization_term, this->weight_list);
}

std::vector<Eigen::MatrixXd> OpenmpiFeedforwardNeuralNetwork::compute_weights_error(const Eigen::VectorXd &x,
    const Eigen::VectorXd &y) const {
    auto fp_results = forward_propagation(x);

    std::vector<Eigen::MatrixXd> result(weight_list.size());
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = Eigen::MatrixXd::Zero(weight_list[i].rows(), weight_list[i].cols());
    }

    back_propagation(fp_results, y, result);

    return result;
}

std::vector<std::vector<double> > OpenmpiFeedforwardNeuralNetwork::get_activations(const std::vector<double>& input) const {
  std::vector<Eigen::VectorXd> original_result = forward_propagation(stdVectorToEigenVector(input));
  std::vector<std::vector<double> > result;

  for (int i = 0; i < original_result.size(); ++i) {
    result.push_back(eigenVectorToStdVector(original_result[i]));
  }

  return result;
}

std::vector<double> OpenmpiFeedforwardNeuralNetwork::predict(const std::vector<double>& input) const {
  return eigenVectorToStdVector(predict(stdVectorToEigenVector(input)));
}

std::vector<dc::ManagedMatrix<double> > OpenmpiFeedforwardNeuralNetwork::compute_weights_error(const std::vector<double>& input,
    const std::vector<double>& output) const {
  std::vector<Eigen::MatrixXd> original_results = compute_weights_error(stdVectorToEigenVector(input),
    stdVectorToEigenVector(output));

  std::vector<dc::ManagedMatrix<double> > result;

  for(int i = 0; i < original_results.size(); ++i) {
    result.push_back(eigenMatrixToManagedMatrix(original_results[i]));
  }

  return result;
}

TrainResult OpenmpiFeedforwardNeuralNetwork::train(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
    const TrainSettings& train_settings) {
  return train(managedMatrixToEigenMatrix(x).transpose(), managedMatrixToEigenMatrix(y).transpose(), train_settings);
}

void OpenmpiFeedforwardNeuralNetwork::set_weights(int source_layer, const dc::ManagedMatrix<double> &weights) {
  set_weights_eigen(source_layer, managedMatrixToEigenMatrix(weights));
}

dc::ManagedMatrix<double> OpenmpiFeedforwardNeuralNetwork::get_weights(int source_layer) const {
  return eigenMatrixToManagedMatrix(get_weights_eigen(source_layer));
}

double OpenmpiFeedforwardNeuralNetwork::compute_error(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
    double regularization_term) {
  return compute_error(managedMatrixToEigenMatrix(x).transpose(), managedMatrixToEigenMatrix(y).transpose(), regularization_term);
}
