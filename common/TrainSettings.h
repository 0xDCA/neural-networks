/*
 * TrainSettings.h
 *
 *  Created on: 7/10/2016
 *      Author: drcastroa
 */

#ifndef TRAINSETTINGS_H_
#define TRAINSETTINGS_H_

#include <random>

struct TrainSettings {
	int iterations;
    double regularization_term;
    double step_factor;
    double momentum;
    int inner_steps;
    int threads;
    int blocks;
    std::mt19937* generator;
    double random_epsilon;
    bool initialize_weights_randomly = true;
    double target_error;

    void validate() const {
        if (threads <= 0) {
            throw std::runtime_error("Invalid threads");
        }

        if (blocks <= 0) {
					throw std::runtime_error("Invalid blocks");
				}

        if (inner_steps <= 0) {
            throw std::runtime_error("Invalid inner_steps");
        }

        if (iterations <= 0) {
            throw std::runtime_error("Invalid iterations");
        }

        if (generator == nullptr) {
        	throw std::runtime_error("Invalid generator");
        }

        if (target_error < 0) {
        	throw std::runtime_error("Invalid target_error");
				}
    }
};

#endif /* TRAINSETTINGS_H_ */
