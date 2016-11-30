/*
 * cuda-helpers.h
 *
 *  Created on: 7/10/2016
 *      Author: drcastroa
 */

#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <vector>
#include "ManagedMatrix.h"
#include "TrainResult.h"

struct TrainSettings;

namespace dc_internal {
	std::vector<std::vector<double> > forward_propagate(const std::vector<int>& layer_neurons,
			const std::vector<double>& input,
			const std::vector<dc::ManagedMatrix<double> >& weights);

	std::vector<dc::ManagedMatrix<double> > back_propagate(const std::vector<int>& layer_neurons,
			const std::vector<double>& input, const std::vector<double>& output,
			const std::vector<dc::ManagedMatrix<double> >& weights);

	TrainResult train(const std::vector<int>& layer_neurons,
			const dc::ManagedMatrix<double>& x,
			const dc::ManagedMatrix<double>& y, const TrainSettings& train_settings,
			std::vector<dc::ManagedMatrix<double> >& weights);

	double compute_error(const std::vector<int>& layer_neurons,
			const dc::ManagedMatrix<double>& x,
			const dc::ManagedMatrix<double>& y,
			std::vector<dc::ManagedMatrix<double> >& weights,
			double regularization_term, int blocks, int threads_per_block);
}


#endif /* CUDA_HELPERS_H_ */
