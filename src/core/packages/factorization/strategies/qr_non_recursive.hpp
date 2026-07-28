/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_NON_RECURSIVE_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_NON_RECURSIVE_HPP

#include "matrix/matrix.hpp"
#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/kernels/qr_non_recursive_kernel.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class qr_non_recursive {

	template<typename ADataType, typename ArchitectureSpec>
	using qr_non_recursive_pctx_t = spec::problem_context<
		qr_factorization<general_matrix<matrix_base<ADataType>>>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "qr_non_recursive"; }

	template<typename ADatatype, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const qr_non_recursive_pctx_t<ADatatype, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		// Todo: Get qr_non_recursive_kernel from spec as kernel
		qr_non_recursive_kernel<ArchitectureSpec>(
			pctx.a.cntg(), pctx.a.strd(), pctx.a.data(),
			pctx.a.strd_step(), pctx.tau.data(), pctx.work.data()
		);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename ADatatype, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE constexpr bool
	can_compute(const qr_non_recursive_pctx_t<ADatatype, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class qr_non_recursive
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_QR_NON_RECURSIVE_HPP
