/*
 * ApproxMatrixMatcher.h
 *
 *  Created on: 14/10/2016
 *      Author: drcastroa
 */

#ifndef APPROXMATRIXMATCHER_H_
#define APPROXMATRIXMATCHER_H_

#include "ManagedMatrix.h"
#include <catch.hpp>
#include <string>
#include <limits>

class ApproxMatrixMatcher: public Catch::Matchers::Impl::MatcherImpl<ApproxMatrixMatcher, dc::ManagedMatrix<double> > {
public:
	ApproxMatrixMatcher(const dc::ManagedMatrix<double>& data,
			double epsilon = std::numeric_limits<float>::epsilon() * 100, double scale = 1.0);

	virtual ~ApproxMatrixMatcher();
	explicit ApproxMatrixMatcher(const ApproxMatrixMatcher& other);

	virtual bool match(dc::ManagedMatrix<double> const& expr) const;
	virtual std::string toString() const;

private:
	dc::ManagedMatrix<double> m_data;
	double m_epsilon, m_scale;
};

#endif /* APPROXMATRIXMATCHER_H_ */
