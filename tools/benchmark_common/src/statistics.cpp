/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "statistics.hpp"

#include <cmath>

double perflibs::statistics::mean_incremental(normal_distribution in, double x) {
	return in.mean*in.n/(in.n+1) + x/(in.n+1);
}

double perflibs::statistics::sample_stddev_incremental(normal_distribution in, double m, double x) {
	double var = in.stddev*in.stddev;
	var = (m - in.mean)*(m - in.mean) +
		(in.n > 0 ? (var*(in.n-1) + (x - m)*(x - m))/in.n : (double) 0);
	return std::sqrt(var);
}

perflibs::statistics::normal_distribution perflibs::statistics::sample_normal_incremental(normal_distribution in, double x) {
	const double m = mean_incremental(in, x);
	const double stddev = sample_stddev_incremental(in, m, x);
	return { m, stddev, in.n + 1 };
}

perflibs::statistics::normal_distribution operator/(const perflibs::statistics::normal_distribution &a, double b) {
	return { a.mean/b, a.stddev/b, a.n };
}

perflibs::statistics::normal_distribution operator/(double a, const perflibs::statistics::normal_distribution &b) {
	const double stddev_frac = b.stddev / b.mean;
	const double mean = a/b.mean;
	return { mean, mean*stddev_frac, b.n };
}
