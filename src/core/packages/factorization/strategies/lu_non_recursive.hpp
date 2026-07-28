/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_NON_RECURSIVE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_NON_RECURSIVE_HPP

#include "matrix/matrix.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/kernels/lu_unblocked_direct_kernel.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class lu_non_recursive {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using lu_non_recursive_pctx_t = spec::problem_context<
		lu_factorization<general_matrix<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "lu_non_recursive"; }

	template<typename ADatatype, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const lu_non_recursive_pctx_t<ADatatype, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Todo: Get lu_non_recursive_kernel from spec as kernel
		lu_unblocked_direct_kernel<ArchitectureSpec>(pctx.a.cntg(), pctx.a.strd(), pctx.a.data(), pctx.a.strd_step(), pctx.pivot.data(), pctx.info);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename ADatatype, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const lu_non_recursive_pctx_t<ADatatype, IntType, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class lu_non_recursive
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_NON_RECURSIVE_HPP
