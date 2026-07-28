/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_SOLVE_PROBLEM_CONTEXT_BASES_HPP

#include <utility>

#include "spec/problem_context.hpp"

namespace perflibs::linalg::solve {

template<
	typename AMatrixType,
	typename BMatrixType,
	typename ScalarType
>
struct solve {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using scalar_type   = ScalarType;

	//TODO: this is compatibility for specs generation
	struct c_matrix_type { using value_type = typename b_matrix_type::value_type; };

	//TODO: the code in the LINALG should be able to infer
	// the parameters below from the matrix objects
	perflibs_side side;
	perflibs_trans transa;

	a_matrix_type a;
	b_matrix_type b;

	scalar_type   alpha;

	zero_mode beta_zero_mode { zero_mode::scale };

	solve(perflibs_side side, perflibs_trans transa,
		AMatrixType a_, BMatrixType b_, ScalarType alpha_)
	:	side   { side }
	,	transa { transa }
	,	a      { std::move(a_) }
	,	b      { std::move(b_) }
	,	alpha  { std::move(alpha_) }
	{	}
}; //struct solve

template<
	typename AMatrixType,
	typename BMatrixType,
	typename ScalarType
>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const solve<AMatrixType, BMatrixType, ScalarType>& pctx) {
	return pctx.a.cntg() * pctx.b.cntg() * pctx.b.strd();
}

} //namespace perflibs::linalg::solve

namespace perflibs::linalg::spec {

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<solve::solve<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<solve::solve<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::b_matrix_type::value_type;
};

} // /namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_SOLVE_PROBLEM_CONTEXT_BASES_HPP
