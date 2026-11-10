/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_BENCHMARK_STATISTICS_HPP
#define PERFLIBS_BENCHMARK_STATISTICS_HPP

namespace perflibs::statistics {
	struct normal_distribution {
		double mean = 0;
		double stddev = 1;
		double n = 0;

		normal_distribution() = default;
		normal_distribution(double mean, double stddev, double n)
			: mean(mean), stddev(stddev), n(n) {
		}
	};

	double mean_incremental(normal_distribution in, double x);
	double sample_stddev_incremental(normal_distribution in, double m, double x);
	normal_distribution sample_normal_incremental(normal_distribution in, double x);
}

perflibs::statistics::normal_distribution operator/(const perflibs::statistics::normal_distribution &a, double b);
perflibs::statistics::normal_distribution operator/(double a, const perflibs::statistics::normal_distribution &b);

#endif
