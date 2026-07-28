/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_TRMM_REFERENCE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_TRMM_REFERENCE_HPP

#include "spec/problem_context.hpp"
#include "spec/problem_context_helpers.hpp"
#include "packages/matmul/problem_context_bases.hpp"

#include "matmul_references.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class trmm_reference {
	template<typename DataType, typename ArchitectureSpec>
	using problem_context_t = spec::problem_context<
		matmul2<
			triangular_matrix<matrix_base<const DataType>>,
			general_matrix<   matrix_base<      DataType>>,
			DataType
		>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "trmm_reference"; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const problem_context_t<DataType, ArchitectureSpec>& pctx) const {

		if (! this->can_compute(pctx) ) return false;

		const bool is_left =
			is_cntg_contig(pctx.b) && (pctx.b.cntg() <= 1 || !is_strd_contig(pctx.b));

		auto perflibs_side = is_left ? PERFLIBS_LEFT : PERFLIBS_RIGHT;

		const auto side            = side_to_c(perflibs_side);
		const bool is_linalg_trans_a = is_cntg_contig(pctx.a);
		const bool is_trans_a      = is_linalg_trans_a ^ (!is_left);
		const auto transa          = is_trans_a ? pctx.a.is_conj() ? 'C' : 'T' : 'N';
		const auto perflibs_uplo      = is_linalg_trans_a ? pctx.a.uplo() : lower_flip(pctx.a.uplo());
		const auto uplo            = uplo_to_c(perflibs_uplo);
		const auto diag            = diag_to_c(pctx.a.diag());

		pl_linalg_int_t m              = is_left         ? pctx.a.strd()      : pctx.b.strd();
		pl_linalg_int_t n              = is_left         ? pctx.b.strd()      : pctx.a.strd();
		const pl_linalg_int_t lda      = is_linalg_trans_a ? pctx.a.strd_step() : pctx.a.cntg_step();
		const pl_linalg_int_t ldb      = is_left         ? pctx.b.strd_step() : pctx.b.cntg_step();

		reference::trmm<DataType>(&side, &uplo, &transa, &diag, &m, &n, &pctx.alpha, pctx.a.data(), &lda, pctx.b.data(), &ldb);
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext& pctx) const { return false; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const problem_context_t<DataType, ArchitectureSpec>& pctx) const {
		return ( pctx.a.strd_step() == 1 || pctx.a.cntg_step() == 1)
		    && ( pctx.b.strd_step() == 1 || pctx.b.cntg_step() == 1);
	}
}; //class trmm_reference

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_TRMM_REFERENCE_HPP
