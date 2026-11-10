/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_BENCHMARK_IO_HPP
#define PERFLIBS_PCTX_BENCHMARK_IO_HPP

#include "measurement.hpp"
#include <nlohmann/json.hpp>
#include "statistics.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace perflibs::linalg {

struct bench_settings {
	measurement_settings measurement;
	std::vector<bool> ignore;
	std::size_t first_strategy;
	std::size_t last_strategy;
};

struct bench_result {
	bool success;
	bool out_of_time;
	statistics::normal_distribution time;
	statistics::normal_distribution mflops;
	double max_mflops;
};

bench_settings parse_bench_settings(const nlohmann::json& j, std::size_t n_strategies);
bench_result make_bench_result(const measurement_result& measurement, double nflops);
nlohmann::json make_bench_output(const nlohmann::json& j);
void append_bench_result(nlohmann::json& output, std::size_t index, std::string_view name,
	const bench_result& result);
void print_bench_output(nlohmann::json& output, const std::vector<bool>& ignore);

} // namespace perflibs::linalg

#endif
