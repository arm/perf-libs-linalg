/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_FFT_STATISTICS_HPP
#define PERFLIBS_FFT_STATISTICS_HPP

#include <algorithm>
#include <random>
#include <vector>

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

	double mean(const std::vector<double> &values);
	double mean_incremental(normal_distribution in, double x);

	double sample_variance(const std::vector<double> &values, double m);
	double sample_variance(const std::vector<double> &values);

	double sample_stddev(const std::vector<double> &values, double m);
	double sample_stddev(const std::vector<double> &values);
	double sample_stddev_incremental(normal_distribution in, double m, double x);

	normal_distribution sample_normal(const std::vector<double> &values);
	normal_distribution sample_normal_incremental(normal_distribution in, double x);

	normal_distribution combine_normals(const normal_distribution &dist1, const normal_distribution &dist2);
	normal_distribution combine_normals(const std::vector<normal_distribution> &dists);

	template<typename Compare = std::less<double>>
	[[maybe_unused]]
	static double exact_test(const std::vector<double> &v1, const std::vector<double> &v2, Compare compare = Compare()) {
		double m_diff = mean(v1) - mean(v2);
		std::vector<double> v_comb(v1.size() + v2.size());
		std::copy(v1.begin(), v1.end(), v_comb.begin());
		std::copy(v2.begin(), v2.end(), v_comb.begin() + v1.size());
		std::vector<bool> v_sel(v_comb.size());
		std::fill(v_sel.end() - v1.size(), v_sel.end(), true);
		std::vector<double> v1_shuffle(v1.size());
		std::vector<double> v2_shuffle(v2.size());
		size_t diff_cmp_count = 0, total_count = 0;
		do {
			for (size_t i1=0, i2=0, in=0; in<v_sel.size(); ++in) {
				if (v_sel[in]) {
					v1_shuffle[i1++] = v_comb[in];
				} else {
					v2_shuffle[i2++] = v_comb[in];
				}
			}
			double m_shuffle_diff = mean(v1_shuffle) - mean(v2_shuffle);
			if (compare(m_shuffle_diff, m_diff)) {
				++diff_cmp_count;
			}
			++total_count;
		} while (std::next_permutation(v_sel.begin(), v_sel.end()));
		return (double) diff_cmp_count / total_count;
	}

	template<typename Compare = std::less<double>>
	[[maybe_unused]]
	static double shuffle_test(const std::vector<double> &v1, const std::vector<double> &v2,
			size_t limit, Compare compare = Compare()) {
		double m_diff = mean(v1) - mean(v2);
		std::vector<double> v_comb(v1.size() + v2.size());
		std::copy(v1.begin(), v1.end(), v_comb.begin());
		std::copy(v2.begin(), v2.end(), v_comb.begin() + v1.size());
		std::vector<bool> v_sel(v_comb.size());
		std::fill(v_sel.end() - v1.size(), v_sel.end(), true);
		std::vector<double> v1_shuffle(v1.size());
		std::vector<double> v2_shuffle(v2.size());
		std::mt19937 gen;
		size_t diff_cmp_count = 0, total_count = 0;
		while (total_count < limit) {
			std::shuffle(v_sel.begin(), v_sel.end(), gen);
			for (size_t i1=0, i2=0, in=0; in<v_sel.size(); ++in) {
				if (v_sel[in]) {
					v1_shuffle[i1++] = v_comb[in];
				} else {
					v2_shuffle[i2++] = v_comb[in];
				}
			}
			double m_shuffle_diff = mean(v1_shuffle) - mean(v2_shuffle);
			if (compare(m_shuffle_diff, m_diff)) {
				++diff_cmp_count;
			}
			++total_count;
		}
		return (double) diff_cmp_count / (double) total_count;
	}

	struct t_test_result {
		double t;
		double v;
		double v1_p; // probability if v=1
	};

	t_test_result welch_t_test(normal_distribution n1, normal_distribution n2);
	t_test_result welch_t_test(const std::vector<double> &v1, const std::vector<double> &v2);
}

perflibs::statistics::normal_distribution operator*(const perflibs::statistics::normal_distribution &a, double b);
perflibs::statistics::normal_distribution operator*(double a, const perflibs::statistics::normal_distribution &b);
perflibs::statistics::normal_distribution operator/(const perflibs::statistics::normal_distribution &a, double b);
perflibs::statistics::normal_distribution operator/(double a, const perflibs::statistics::normal_distribution &b);

#endif
