#include "runner-common.h"
#include "CudaFeedforwardNeuralNetwork.h"
#include "FeedforwardNeuralNetworkClassFactory.h"

using dc::ManagedMatrix;
using std::cout;

int main(int argc, char* argv[])
{
	run_neural_network(FeedforwardNeuralNetworkClassFactory<CudaFeedforwardNeuralNetwork>(),
		argc, argv);
}
