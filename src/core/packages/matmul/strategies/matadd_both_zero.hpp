/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD_BOTH_ZERO_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD_BOTH_ZERO_HPP

#include "framework/linalg_util.hpp"
#include "matrix/operations.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "spec/problem_context.hpp"

#include <algorithm>
#include <string_view>

namespace perflibs::linalg::matmul {

class matadd_both_zero {
public:
	static constexpr std::string_view name() { return "matadd_both_zero"; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (
		const spec::problem_context<matadd2<Ts...>, ArchitectureSpec>& pctx
	) const {
		if (!this->can_compute(pctx)) return false;

		impl(pctx.c);
		return true;
	}

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (
		const spec::problem_context<matadd3<Ts...>, ArchitectureSpec>& pctx
	) const {
		if (!this->can_compute(pctx)) return false;

		impl(pctx.c);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(
		const spec::problem_context<matadd2<Ts...>, ArchitectureSpec>& pctx
	) const {
		using  scalar_type =  typename spec::problem_context<matadd2<Ts...>, ArchitectureSpec>::scalar_type;

		return pctx.alpha  == zero<scalar_type>
		    && pctx.beta   == zero<scalar_type>;
	}

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(
		const spec::problem_context<matadd3<Ts...>, ArchitectureSpec>& pctx
	) const {
		using scalar_type = typename spec::problem_context<matadd3<Ts...>, ArchitectureSpec>::scalar_type;

		return pctx.alpha == zero<scalar_type>
		    && pctx.beta  == zero<scalar_type>;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

private:
	template<typename Matrix>
	PERFLIBS_LINALG_INLINE
	void impl(const Matrix& dst) const {
		using value_type = typename Matrix::value_type;

		const bool cntg_contiguous =  dst.cntg_step() == 1
		                           && dst.strd_step() == dst.cntg();

		const bool strd_contiguous =  dst.strd_step() == 1
		                           && dst.cntg_step() == dst.strd();

		if (cntg_contiguous || strd_contiguous) {
			std::fill_n(
				dst.data(),
				dst.cntg() * dst.strd(),
				zero<value_type>
			);
		}
		else {
			set(zero<value_type>, dst);
		}
	}
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD_BOTH_ZERO_HPP
