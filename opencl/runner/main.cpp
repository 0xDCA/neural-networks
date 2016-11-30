#include "runner-common.h"
#include "OpenclFeedforwardNeuralNetwork.h"
#include "FeedforwardNeuralNetworkClassFactory.h"

using dc::ManagedMatrix;
using std::cout;

int main(int argc, char* argv[])
{
	run_neural_network(FeedforwardNeuralNetworkClassFactory<OpenclFeedforwardNeuralNetwork>(),
		argc, argv);
}
