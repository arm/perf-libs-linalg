/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "statistics.hpp"
#include "perflibs_numeric_utils.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>

double perflibs::statistics::mean(const std::vector<double> &values) {
	double sum = 0;
	for (double v : values) {
		sum += v;
	}
	return sum / values.size();
}

double perflibs::statistics::mean_incremental(normal_distribution in, double x) {
	return in.mean*in.n/(in.n+1) + x/(in.n+1);
}

double perflibs::statistics::sample_variance(const std::vector<double> &values, double m) {
	if (values.size() <= 1) {
		return std::numeric_limits<double>::infinity();
	}
	double sum = 0;
	for (double v : values) {
		sum += (v-m)*(v-m);
	}
	return sum / (values.size()-1);
}

double perflibs::statistics::sample_variance(const std::vector<double> &values) {
	return sample_variance(values, mean(values));
}

double perflibs::statistics::sample_stddev(const std::vector<double> &values, double m) {
	return std::sqrt(sample_variance(values, m));
}

double perflibs::statistics::sample_stddev(const std::vector<double> &values) {
	return std::sqrt(sample_variance(values));
}

double perflibs::statistics::sample_stddev_incremental(normal_distribution in, double m, double x) {
	double var = in.stddev*in.stddev;
	var = (m - in.mean)*(m - in.mean) +
		(in.n > 0 ? (var*(in.n-1) + (x - m)*(x - m))/in.n : (double) 0);
	return std::sqrt(var);
}

perflibs::statistics::normal_distribution perflibs::statistics::sample_normal(const std::vector<double> &values) {
	double m = mean(values);
	double stddev = sample_stddev(values, m);
	return { m, stddev, (double) values.size() };
}

perflibs::statistics::normal_distribution perflibs::statistics::sample_normal_incremental(normal_distribution in, double x) {
	double m = mean_incremental(in, x);
	double stddev = sample_stddev_incremental(in, m, x);
	return { m, stddev, in.n + 1 };
}

perflibs::statistics::normal_distribution perflibs::statistics::combine_normals(const normal_distribution &dist1, const normal_distribution &dist2) {
	double n_total = dist1.n + dist2.n;
	double m = 0, var = 0;
	for (const auto &dist : { dist1, dist2 }) {
		double d_var = dist.stddev*dist.stddev;
		m += (dist.n/n_total)*dist.mean;
		var += ((dist.n-1)/(n_total-2))*d_var;
	}
	return { m, std::sqrt(var), n_total };
}

perflibs::statistics::normal_distribution perflibs::statistics::combine_normals(const std::vector<normal_distribution> &dists) {
	double n_total = 0, m = 0, var = 0;
	for (const auto &dist : dists) {
		n_total += dist.n;
	}
	for (const auto &dist : dists) {
		double d_var = dist.stddev*dist.stddev;
		m += (dist.n/n_total)*dist.mean;
		var += ((dist.n-1)/(n_total-dists.size()))*d_var;
	}
	return { m, std::sqrt(var), n_total };
}

perflibs::statistics::t_test_result perflibs::statistics::welch_t_test(normal_distribution n1, normal_distribution n2) {
	double n1_s2 = n1.stddev*n1.stddev, n2_s2 = n2.stddev*n2.stddev;
	double s_delta = n1_s2/n1.n + n2_s2/n2.n;
	double t = (n1.mean - n2.mean)/std::sqrt(s_delta);
	double v = s_delta*s_delta / ((n1_s2*n1_s2/(n1.n*(n1.n-1))) +
			(n2_s2*n2_s2/(n2.n*(n2.n-1))));
	double v1_p = 0.5 + (1.0/perflibs::consts::pi<double>)*std::atan(t);
	return { t, v, v1_p };
}

perflibs::statistics::t_test_result perflibs::statistics::welch_t_test(const std::vector<double> &v1, const std::vector<double> &v2) {
	auto n1 = sample_normal(v1), n2 = sample_normal(v2);
	return welch_t_test(n1, n2);
}

perflibs::statistics::normal_distribution operator*(const perflibs::statistics::normal_distribution &a, double b) {
	return { a.mean*b, a.stddev*b, a.n };
}

perflibs::statistics::normal_distribution operator*(double a, const perflibs::statistics::normal_distribution &b) {
	return { b.mean*a, b.stddev*a, b.n };
}

perflibs::statistics::normal_distribution operator/(const perflibs::statistics::normal_distribution &a, double b) {
	return { a.mean/b, a.stddev/b, a.n };
}

perflibs::statistics::normal_distribution operator/(double a, const perflibs::statistics::normal_distribution &b) {
	double stddev_frac = b.stddev / b.mean;
	double mean = a/b.mean;
	return { mean, mean*stddev_frac, b.n };
}
