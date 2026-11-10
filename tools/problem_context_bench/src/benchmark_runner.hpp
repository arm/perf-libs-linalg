/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_BENCHMARK_RUNNER_HPP
#define PERFLIBS_PCTX_BENCHMARK_RUNNER_HPP

/*
 * Run the selected strategies for a context and print their results.
 */

#include "benchmark_io.hpp"
#include "context_flops.hpp"
#include "measurement.hpp"
#include "supported_compute.hpp"

#include <nlohmann/json.hpp>

namespace perflibs::linalg {

struct null_prepare {
	template<typename ProblemContextBase>
	inline
	void operator()(const ProblemContextBase&) const { }
};

// prepare(pctx_base) runs before each warm-up and measured call. Use null_prepare
// when inputs need no reset. Keep any data used by prepare alive until bench() returns.
template<typename ProblemContext, typename Prepare>
void bench(const nlohmann::json& j, const ProblemContext& pctx_base, const Prepare& prepare=null_prepare { } ) {
	auto settings = parse_bench_settings(j, get_n_strategies(pctx_base));
	auto output = make_bench_output(j);

	for(auto si = settings.first_strategy; si != settings.last_strategy; ++si) {
		if (settings.ignore[si]) {
			continue;
		}

		const auto nflops = get_dnflops(pctx_base);
		// Warm-up calls do not contribute to the timing results.
		// Measured time includes both prepare(pctx_base) and compute_index().
		const auto measurement = measure(settings.measurement,
			[&] { prepare(pctx_base); },
			[&] { return compute_index(pctx_base, si); });
		const auto result = make_bench_result(measurement, nflops);

		append_bench_result(output, si, get_strategy_name(pctx_base, si), result);
		if (result.out_of_time) {
			settings.ignore[si] = true;
		}
	}

	print_bench_output(output, settings.ignore);
}

} // namespace perflibs::linalg

#endif
