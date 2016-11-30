/*
 * ApproxMatrixMatcher.cpp
 *
 *  Created on: 14/10/2016
 *      Author: drcastroa
 */

#include "ApproxMatrixMatcher.h"

#include <sstream>

ApproxMatrixMatcher::ApproxMatrixMatcher(const dc::ManagedMatrix<double>& data, double epsilon, double scale) :
		m_data(data), m_epsilon(epsilon), m_scale(scale) {

}

ApproxMatrixMatcher::ApproxMatrixMatcher(const ApproxMatrixMatcher& other) :
		m_data(other.m_data), m_epsilon(other.m_epsilon), m_scale(other.m_scale) {

}

ApproxMatrixMatcher::~ApproxMatrixMatcher() {

}

bool ApproxMatrixMatcher::match(dc::ManagedMatrix<double> const& expr) const {
	if (expr.get_columns() != m_data.get_columns() || expr.get_rows() != m_data.get_rows()) {
		return false;
	}

	for (int i = 0; i < m_data.get_columns(); ++i) {
		for(int j = 0; j < m_data.get_rows(); ++j) {
			if (expr.get(j, i) != Approx(m_data.get(j, i)).epsilon(m_epsilon).scale(m_scale)) {
				return false;
			}
		}
	}

	return true;
}

std::string ApproxMatrixMatcher::toString() const {
	std::ostringstream oss;

	oss << "Approx: \n";
	oss << m_data;
	oss << '\n';
	oss << "scale = " << m_scale << ", epsilon = " << m_epsilon << '\n';

	return oss.str();
}
