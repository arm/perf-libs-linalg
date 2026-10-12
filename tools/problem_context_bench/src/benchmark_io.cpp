/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "benchmark_io.hpp"
#include "json_optional.hpp"
#include "statistics_json.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

namespace perflibs::linalg {

bench_settings parse_bench_settings(const nlohmann::json& j, std::size_t n_strategies) {
	const auto pre_runs    = from_json_or(j, "pre_runs",    1_ki               );
	const auto nruns       = from_json_or(j, "nruns",       1_ki               );
	const auto niters      = from_json_or(j, "niters",      1_ki               );
	const auto target_secs = from_json_or(j, "target_secs", 0.                 );
	      auto ignore      = from_json_or(j, "ignore",      std::vector<bool>{});

	ignore.resize(n_strategies, false);

	const auto [first_strategy, last_strategy] = [&] {
		if(const auto si_type = perflibs::from_json<std::string>(j, "strategy"); si_type && *si_type == "sweep") {
			return std::pair { 0uz, n_strategies };
		}

		const auto strategy_index = from_json_or(j, "strategy", 0uz);

		return std::pair { strategy_index, strategy_index + 1uz };
	}();

	return { pre_runs, nruns, niters, target_secs, std::move(ignore), first_strategy, last_strategy };
}

nlohmann::json make_bench_output(const nlohmann::json& j) {
	auto j2 = j;
	j2["results"] = nlohmann::json::array();
	return j2;
}

void append_bench_result(nlohmann::json& output, std::size_t index, std::string_view name,
	const bench_result& result) {
	nlohmann::json entry {
		{ "strategy_index", index },
		{ "strategy_name", std::string { name } },
		{ "compute_success", result.success },
		{ "out_of_time", result.out_of_time },
	};

	if(result.success) {
		entry["max_mflops"] = result.max_mflops;
		entry["time"]       = to_json(result.time);
		entry["mflops"]     = to_json(result.mflops);
	}

	output["results"].push_back(entry);
}

void print_bench_output(nlohmann::json& output, const std::vector<bool>& ignore) {
	output["ignore"] = to_json(ignore);
	std::cout << std::setw(4) << output << std::endl;
}

} // namespace perflibs::linalg
