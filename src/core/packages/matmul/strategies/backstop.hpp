/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_BACKSTOP_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_BACKSTOP_HPP

#include "naive/gemm.hpp"
#include "framework/compute_position.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

template<typename ProblemContext>
concept has_general_c_matrix_type = is_general_matrix_v<typename ProblemContext::c_matrix_type>;

//falls back to using the naive gemm
class backstop {
public:
	static constexpr std::string_view name() { return "backstop"; }

	template<typename ProblemContext>
	requires has_general_c_matrix_type<ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using architecture_spec_type = typename ProblemContext::architecture_spec_type;

		if( !this->can_compute(pctx) ) return false;

		auto driver = naive_gemm<architecture_spec_type> { pctx.beta_zero_mode };

		driver(pctx.a, pctx.b, pctx.c, compute_position { 0, 0, 0, 0 }, pctx.alpha, pctx.beta);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires has_general_c_matrix_type<ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class backstop

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_BACKSTOP_HPP
