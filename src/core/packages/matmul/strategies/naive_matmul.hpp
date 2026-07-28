/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_NAIVE_MATMUL_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_NAIVE_MATMUL_HPP

#include "framework/linalg_util.hpp"
#include "naive/out_of_place_matmul.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class out_of_place_naive_matmul {
public:
	static constexpr std::string_view name() { return "out_of_place_naive_matmul"; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		auto driver = naive_out_of_place_matmul{};
		const compute_position pos{ 0, 0, 0, 0 };
		driver(pctx.a, pctx.b, pctx.c, pctx.d, pos, pctx.alpha, pctx.beta);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return true;
	}

}; // class out_of_place_naive_matmul
} // namespace perflibs::linalg::matmul
#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_NAIVE_MATMUL_HPP
