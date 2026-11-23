/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_NO_PIVOT_RECURSIVE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_NO_PIVOT_RECURSIVE_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/fwd.hpp"
#include "matrix/adaptors.hpp"

#include "perflibs_util.hpp"

#include <span>
#include <string_view>

namespace perflibs::linalg::factorization {

/*
 * Recursive LU factorization without pivoting.
 *
 *	[ A11 A12 ]   1. factor the left half [A11; A21] (recursively)
 *	[ A21 A22 ]   2. A12 = L11^-1 A12 and A22 = A22 - L21 A12 (lu_panel_update with no pivots)
 *	              3. factor A22 (recursively)
 * A zero pivot in step 1 at column s limits step 2 to the s factored columns and skips step 3.
 */
class lu_no_pivot_recursive {

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	using lu_no_pivot_pctx_t = spec::problem_context<
		lu_factorization_no_pivot<general_matrix<matrix_base<ADataType>>, IntType>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "lu_no_pivot_recursive"; }

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const lu_no_pivot_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		auto a                  = pctx.a;
		const kernel_inttype m  = a.cntg();
		const kernel_inttype n  = a.strd();
		const kernel_inttype n1 = min(m, n) / 2;
		const std::span<IntType> no_pivot{};

		// 1. factor the left half
		auto left      = get_strd_panel(a, 0, n1);
		IntType iinfo  = 0;
		auto left_pctx = spec::problem_context{
			lu_factorization_no_pivot{ left, iinfo },
			pctx.architecture_spec
		};
		compute(left_pctx);

		// 2. update the right half with the factored columns
		const kernel_inttype factored = iinfo > 0 ? kernel_inttype(iinfo - 1) : n1;
		auto right = get_strd_panel(a, n1, n - n1);
		if (factored > 0) {
			auto factored_panel = get_strd_panel(a, 0, factored);
			auto update_pctx    = spec::problem_context{
				lu_panel_update{ factored_panel, right, no_pivot },
				pctx.architecture_spec
			};
			compute(update_pctx);
		}

		if (iinfo > 0) {
			if (pctx.info == 0) {
				pctx.info = iinfo;
			}
			return true;
		}

		// 3. factor the trailing matrix
		if (m > n1) {
			auto trailing      = a.sub_matrix(n1, m - n1, n1, n - n1);
			IntType tinfo      = 0;
			auto trailing_pctx = spec::problem_context{
				lu_factorization_no_pivot{ trailing, tinfo },
				pctx.architecture_spec
			};
			compute(trailing_pctx);
			if (pctx.info == 0 && tinfo > 0) {
				pctx.info = tinfo + n1;
			}
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename ADataType, typename IntType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const lu_no_pivot_pctx_t<ADataType, IntType, ArchitectureSpec>& pctx) const {
		// problems with at most min_recursive_size columns to eliminate use lu_no_pivot_non_recursive
		const auto spec = get_spec(spec::strategy_tag<lu_factorization_no_pivot_generic>{}, pctx);
		return min(pctx.a.cntg(), pctx.a.strd()) > spec.min_recursive_size;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class lu_no_pivot_recursive
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LU_NO_PIVOT_RECURSIVE_HPP
