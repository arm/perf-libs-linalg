/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_COMPRESSED_TRIANGULAR_SOLVE_VECTOR_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_COMPRESSED_TRIANGULAR_SOLVE_VECTOR_HPP

#include "packages/solve/problem_context_bases.hpp"

#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include "blas/compressed_solve.hpp"

#include <string_view>

namespace perflibs::linalg::solve {

class compressed_triangular_solve_vector {
public:
	static constexpr std::string_view name() { return "compressed_triangular_solve_vector"; }


	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_triangular_solve_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		const auto spec = spec::get_spec(spec::strategy_tag<compressed_triangular_solve_vector>{}, pctx);
		auto driver = tri_band_sv{ spec.kernel_axpby, spec.kernel_dot };
		driver(pctx.a, pctx.b);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_triangular_solve_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class compressed_triangular_solve_vector
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SOLVE_STRATEGIES_COMPRESSED_TRIANGULAR_SOLVE_VECTOR_HPP
