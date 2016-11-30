#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "OpenclFeedforwardNeuralNetwork.h"
#include "FeedforwardNeuralNetworkClassFactory.h"
#include "ManagedMatrix.h"
#include "nn-test.h"

#include <vector>

using dc::ManagedMatrix;

TEST_CASE("calculates the error correctly on GPU", "[error]") {
	SECTION("simple neural network") {
		OpenclFeedforwardNeuralNetwork network({2, 2, 1});

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

		ManagedMatrix<double> weights_0(2, 3);
		weights_0.set_all_row_wise({
			1, -5, -5,
			-5, 10, -10
		});

		ManagedMatrix<double> weights_1(1, 3);
		weights_1.set_all_row_wise({
			-5, 1, 1
		});

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		const double expected_error = 8.06265;
		int blocks = 2;
		int threads_per_block = 2;
		double actual_error = network.compute_error_gpu(sample_x, sample_y, 0.2, blocks, threads_per_block);

		REQUIRE(actual_error == Approx(expected_error));
	}
}

TEST_CASE("passes neural network common tests", "[common]") {
	run_common_nn_tests(FeedforwardNeuralNetworkClassFactory<OpenclFeedforwardNeuralNetwork>());
}
