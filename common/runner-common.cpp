#include "runner-common.h"

#include <iostream>
#include <random>
#include <cxxopts.hpp>
#include <memory>
#include "ManagedMatrix.h"
#include "TrainSettings.h"
#include "FeedforwardNeuralNetwork.h"
#include "data-util.h"
#include "int-util.h"

using dc::ManagedMatrix;
using std::cout;
using std::unique_ptr;

void run_neural_network(const AbstractFeedforwardNeuralNetworkFactory& nn_factory, int argc, char* argv[]) {
  cxxopts::Options options("runner",
			"A PoC implementation of a parallel Feed-forward neural network");

	options.add_options()
	  ("t,threads", "Number of threads to use", cxxopts::value<int>()->default_value("1"))
	  ("b,blocks", "Number of blocks to use", cxxopts::value<int>()->default_value("1"))
	  ("i,iterations", "Max iterations", cxxopts::value<int>()->default_value("1"))
	  ("s,steps", "Inner steps (steps per thread)", cxxopts::value<int>()->default_value("1"))
	  ("r,regularization_term", "Regularization term", cxxopts::value<double>()->default_value("0.0"))
	  ("m,momentum", "Momentum", cxxopts::value<double>()->default_value("0.0"))
	  ("l,learning_rate", "Learning rate", cxxopts::value<double>()->default_value("0.1"))
	  ("e,epsilon", "During training, weights will be initialized between [-e, e]", cxxopts::value<double>()->default_value("10"))
	  ("error", "Target error", cxxopts::value<double>()->default_value("0.00001"))
	  ("h,help", "Print help")
	  ;

	options.parse(argc, argv);

	if (options.count("help")) {
		std::cout << options.help({"", "Group"}) << std::endl;
		exit(0);
	}

  std::random_device rd;
  std::mt19937 generator(rd());

  unique_ptr<FeedforwardNeuralNetwork> network(nn_factory.create_neural_network({2, 8, 1}));
  //unique_ptr<FeedforwardNeuralNetwork> network(nn_factory.create_neural_network({3, 8, 1}));

  /*MatrixXd weights(2, 3);
  weights << -30, 20, 20, 10, -20, -20;
  MatrixXd weights2(1, 3);
  weights2 << -10, 20, 20;
  network.set_weights(0, weights);
  network.set_weights(1, weights2);*/

  ManagedMatrix<double> sample_x(4, 2);
  sample_x.set_all_row_wise({
  	0, 0,
  	0, 1,
  	1, 0,
  	1, 1
  });

  sample_x = sample_x.get_transposed();


  ManagedMatrix<double> sample_y(4, 1);
  sample_y.set_all_row_wise({
  	0,
  	1,
  	1,
  	0
  });

  sample_y = sample_y.get_transposed();

  /*auto mnist_data = read_iris_database("iris.data");
  ManagedMatrix<double>& sample_x = mnist_data.first;
  ManagedMatrix<double>& sample_y = mnist_data.second;*/

  ManagedMatrix<double> test_sample_x = sample_x;
  ManagedMatrix<double> test_sample_y = sample_y;

  /*auto training_data = generate_data(1000, generator);
  auto test_data = generate_data(100, generator);
  ManagedMatrix<double> sample_x = training_data.first.get_transposed();
  ManagedMatrix<double> sample_y = training_data.second.get_transposed();
  ManagedMatrix<double> test_sample_x = test_data.first.get_transposed();
  ManagedMatrix<double> test_sample_y = test_data.second.get_transposed();*/

  TrainSettings train_settings;
  train_settings.threads = options["threads"].as<int>();
	train_settings.blocks = options["blocks"].as<int>();
	train_settings.generator = &generator;
  train_settings.inner_steps = options["steps"].as<int>();
  train_settings.iterations = options["iterations"].as<int>();
  train_settings.initialize_weights_randomly = true;
  train_settings.regularization_term = options["regularization_term"].as<double>();
	train_settings.momentum = options["momentum"].as<double>();
	train_settings.step_factor = options["learning_rate"].as<double>();
	train_settings.random_epsilon = options["epsilon"].as<double>();
	train_settings.target_error = options["error"].as<double>();

  /*train_settings.regularization_term = 0.0;
  train_settings.momentum = 0.9;
  train_settings.step_factor = 1.0;
  train_settings.random_epsilon = 10;
  train_settings.target_error = 0.00000001;*/

  /*train_settings.regularization_term = 0.1;
	train_settings.momentum = 0.6;
	train_settings.step_factor = 0.06;
	train_settings.random_epsilon = 10;
	train_settings.target_error = 0.001;*/

  train_settings.validate();

  auto result = network->train(sample_x, sample_y, train_settings);

  /*for(int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
          std::vector<double> input = {(double)i, (double)j};
          auto result = network.predict(input);
          cout << "x: [" << i << ", " << j << "] => " << result[0] << '\n';
      }
  }*/

  cout << "Actual iterations: " << result.iterations << '\n';

  cout << "Training error: " << network->compute_error(sample_x,
                                                      sample_y,
                                                      train_settings.regularization_term) << "\n";
  /*cout << "Test error: " << network.compute_error_gpu(test_sample_x,
                                                  test_sample_y,
                                                  train_settings.regularization_term, error_blocks, error_threads) << "\n";*/
}
