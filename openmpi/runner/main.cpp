#include "runner-common.h"
#include "OpenmpiFeedforwardNeuralNetwork.h"
#include "FeedforwardNeuralNetworkClassFactory.h"
#include <iostream>
#include <random>
#include <cxxopts.hpp>
#include <memory>
#include "ManagedMatrix.h"
#include "TrainSettings.h"
#include "FeedforwardNeuralNetwork.h"
#include "data-util.h"
#include "int-util.h"
#include <boost/mpi.hpp>

using dc::ManagedMatrix;
using std::cout;

int main(int argc, char* argv[])
{
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

  OpenmpiFeedforwardNeuralNetwork network({2, 8, 1});

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

  ManagedMatrix<double> test_sample_x = sample_x;
  ManagedMatrix<double> test_sample_y = sample_y;

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


  train_settings.validate();

	boost::mpi::environment env;
	boost::mpi::communicator world;
  auto result = network.train(sample_x, sample_y, train_settings);

	if (world.rank() == 0) {
	  cout << "Actual iterations: " << result.iterations << '\n';

	  cout << "Training error: " << network.compute_error(sample_x,
	                                                      sample_y,
	                                                      train_settings.regularization_term) << "\n";
	}
}
