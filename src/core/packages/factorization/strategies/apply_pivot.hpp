/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_STRATEGIES_APPLY_PIVOT_HPP
#define PERFLIBS_LINALG_STRATEGIES_APPLY_PIVOT_HPP

#include "matrix/matrix.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/misc/interfaces/swap.hpp"


namespace perflibs::linalg::factorization {

class apply_pivot {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using pctx_t = spec::problem_context<
		lu_apply_pivot<general_matrix<matrix_base<ADataType>>, IntType>,
	    ArchitectureSpec
	>;

public:
	template<typename ADatatype, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const pctx_t<ADatatype, IntType, ArchitectureSpec>& pctx) const {

		// Todo: instantiate get_tuned_routine_spec for apply_pivot to pass swap as kernel

		const IntType inc = pctx.a.strd_step();
		const IntType n   = pctx.a.strd();
		for (IntType i = 0 ; i < static_cast<IntType>(pctx.pivot.size()); i++) {
			// pivot entries are converted from Fortran base to C++ indexing
			if (i != pctx.pivot[i]){
				perflibs::linalg::swap<false, IntType, ADatatype, ArchitectureSpec>(&n, get_cntg_panel(pctx.a, i, 1).data(), &inc, get_cntg_panel(pctx.a, pctx.pivot[i], 1).data(), &inc);
			}
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }
}; // class apply_pivot
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_STRATEGIES_APPLY_PIVOT_HPP