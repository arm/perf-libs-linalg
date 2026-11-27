/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LDL_NO_PIVOT_PACKED_UNBLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LDL_NO_PIVOT_PACKED_UNBLOCKED_HPP

#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/kernels/ldl_packed_no_pivot_kernel.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class ldl_no_pivot_packed_unblocked {

	template<typename DataType, typename ArchitectureSpec>
	using ldl_packed_pctx_t = spec::problem_context<ldl_factorization_no_pivot_packed<packed_matrix_base<DataType>>, ArchitectureSpec>;

public:
	static constexpr std::string_view name() { return "ldl_no_pivot_packed_unblocked"; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ldl_packed_pctx_t<DataType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		ldl_packed_no_pivot_kernel<ArchitectureSpec>(pctx.a.cntg(), pctx.ncolm, pctx.a.data(), pctx.output);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ldl_packed_pctx_t<DataType, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class ldl_no_pivot_packed_unblocked
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LDL_NO_PIVOT_PACKED_UNBLOCKED_HPP
