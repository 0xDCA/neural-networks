#include "eigen-util.h"

Eigen::VectorXd stdVectorToEigenVector(const std::vector<double>& v) {
  Eigen::VectorXd result(v.size());

  for (int i = 0; i < v.size(); ++i) {
    result(i) = v[i];
  }

  return result;
}

std::vector<double> eigenVectorToStdVector(const Eigen::VectorXd& v) {
  std::vector<double> result;

  for (int i = 0; i < v.rows(); ++i) {
    result.push_back(v(i));
  }

  return result;
}

dc::ManagedMatrix<double> eigenMatrixToManagedMatrix(const Eigen::MatrixXd& m) {
  dc::ManagedMatrix<double> result(m.rows(), m.cols());

  for(int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      result.set(i, j, m(i, j));
    }
  }

  return result;
}

Eigen::MatrixXd managedMatrixToEigenMatrix(const dc::ManagedMatrix<double>& m) {
  Eigen::MatrixXd result(m.get_rows(), m.get_columns());

  for(int i = 0; i < m.get_rows(); ++i) {
    for (int j = 0; j < m.get_columns(); ++j) {
      result(i, j) = m.get(i, j);
    }
  }

  return result;
}
