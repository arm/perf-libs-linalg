/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_BENCHMARK_MEASUREMENT_HPP
#define PERFLIBS_BENCHMARK_MEASUREMENT_HPP

#include "statistics.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

namespace perflibs::linalg {

struct measurement_settings {
	std::int64_t pre_runs; // Individual warm-up calls, excluded from statistics.
	std::int64_t nruns; // Number of measured batches to collect.
	std::int64_t niters; // Requested calls per measured batch.
	double target_secs; // Measurement-loop budget in seconds; zero means unlimited.
};

struct measurement_result {
	bool success = true;
	bool out_of_time = false;
	// Statistics across batch averages, expressed in seconds per invocation.
	statistics::normal_distribution time;
	// Minimum batch average; remains infinite when no measurement is accepted.
	double min_time = std::numeric_limits<double>::infinity();
};

// prepare() runs before each invoke(); invoke() returns execution success.
// Callers supply operation-specific behavior.
template<typename Clock = std::chrono::steady_clock, typename Prepare, typename Invoke>
measurement_result measure(const measurement_settings& settings, const Prepare& prepare, const Invoke& invoke) {
	measurement_result result;
	// Warm-up calls are outside both the statistics and the measurement-loop budget.
	for(std::int64_t i = 0; i != settings.pre_runs; ++i) {
		prepare();
		result.success &= invoke();
	}

	std::int64_t num_measurements = 0;
	constexpr auto max_niters = std::numeric_limits<std::int64_t>::max();
	// effective_niters != settings.niters because it might get adjusted later on.
	auto effective_niters = std::max<std::int64_t>(1, settings.niters);

	const auto runs_start = Clock::now();

	for(; result.success && !result.out_of_time && num_measurements != settings.nruns;) {
		std::int64_t i = 0;
		// TODO: Exclude preparation timing from total_measurement_time
		const auto iter_start = Clock::now();
		for(; result.success && i != effective_niters; ++i) {
			prepare();
			result.success &= invoke();
		}
		const auto iter_end = Clock::now();
		const auto elapsed = std::chrono::duration<double>(iter_end - iter_start).count();
		// Check between batches and discard the batch that exceeds the budget.
		// This cannot interrupt an invocation that is still running.
		if(settings.target_secs > 0 && std::chrono::duration<double>(iter_end - runs_start).count() > settings.target_secs) {
			result.out_of_time = true;
			break;
		}
		// In case the operations happen very fast, we need to increase the number
		// of iterations. We do this by doubling this number (making sure we don't
		// overflow this value).
		const auto time_per_iter = elapsed / i;
		if(time_per_iter <= 0) {
			if(effective_niters > max_niters / 2) {
				result.success = false;
				break;
			}
			effective_niters *= 2;
			continue;
		}
		result.min_time = std::min(result.min_time, time_per_iter);
		// Each accepted batch contributes one average, regardless of its call count.
		result.time = statistics::sample_normal_incremental(result.time, time_per_iter);
		++num_measurements;
	}
	return result;
}

} // namespace perflibs::linalg

#endif
