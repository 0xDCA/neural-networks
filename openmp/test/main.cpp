#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include "catch.hpp"
#include "OpenmpFeedforwardNeuralNetwork.h"
#include "FeedforwardNeuralNetworkClassFactory.h"
#include "nn-test.h"
#include "TrainSettings.h"

TEST_CASE("passes neural network common tests", "[common]") {
	TrainSettings base_settings;
	base_settings.threads = 8;
	base_settings.blocks = 1;
	run_common_nn_tests(FeedforwardNeuralNetworkClassFactory<OpenmpFeedforwardNeuralNetwork>(), base_settings);
}
