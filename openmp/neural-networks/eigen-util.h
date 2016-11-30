#ifndef EIGENUTIL_H
#define EIGENUTIL_H

#include <Eigen/Dense>
#include "ManagedMatrix.h"

Eigen::VectorXd stdVectorToEigenVector(const std::vector<double>& v);

std::vector<double> eigenVectorToStdVector(const Eigen::VectorXd& v);

dc::ManagedMatrix<double> eigenMatrixToManagedMatrix(const Eigen::MatrixXd& m);

Eigen::MatrixXd managedMatrixToEigenMatrix(const dc::ManagedMatrix<double>& m);

#endif
