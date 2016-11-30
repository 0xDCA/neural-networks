/*
 * util.h
 *
 *  Created on: 15/10/2016
 *      Author: drcastroa
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "ManagedMatrix.h"
#include <random>
#include <utility>

std::pair<dc::ManagedMatrix<double>, dc::ManagedMatrix<double> > generate_data(int n, std::mt19937& generator);

std::pair<dc::ManagedMatrix<double>, dc::ManagedMatrix<double> > read_mnist_database(
		const std::string& image_file_name, const std::string& label_file_name);

std::pair<dc::ManagedMatrix<double>, dc::ManagedMatrix<double> > read_iris_database(
		const std::string& file_name);

#endif /* UTIL_H_ */
