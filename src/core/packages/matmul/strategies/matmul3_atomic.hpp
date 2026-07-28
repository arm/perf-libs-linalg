/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_ATOMIC_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_ATOMIC_HPP

#include "spec/problem_context_helpers.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul3_atomic {
public:
	static constexpr std::string_view name() { return "matmul3_atomic"; }

	template<typename DataType, typename ArchitectureSpec>
	using pctx_t = spec::problem_context<
		matmul3<
			general_matrix<matrix_base<const DataType>>,
			general_matrix<matrix_base<const DataType>>,
			general_matrix<matrix_base<      DataType>>,
			DataType
		>,
		ArchitectureSpec
	>;

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const pctx_t<DataType, ArchitectureSpec>& pctx) const {
		if (! can_compute(pctx) ) return false;

		const auto transa = c_to_trans(spec::gemm_transa(pctx));
		const auto transb = c_to_trans(spec::gemm_transb(pctx));
		const auto kernel = get_atomic_dispatch_kernel(pctx);

		if(pctx.c.cntg_step() == 1) {
			return kernel(
				transa, transb,
				pctx.a.strd(), pctx.b.strd(), pctx.a.cntg(),
				pctx.a.data(), spec::gemm_lda(pctx),
				pctx.b.data(), spec::gemm_ldb(pctx),
				pctx.c.data(), pctx.c.strd_step(),
				pctx.alpha, pctx.beta);
		}
		/*
		 * If C is a vector, rather than a matrix, and it has strides along the
		 * cntg dimension, then we can swap A & B around and re-present those strides
		 * down cntg as lda as cntg becomes strd
		 *
		 * Note, because we have to transpose the trans option,
		 * if either of the input matrices are CONJTRANS  then we can not continue
		 */
		else if(pctx.a.strd() == 1 && !pctx.a.is_conj() && !pctx.b.is_conj()) {
			return kernel(
				transpose(transb), transpose(transa),
				pctx.b.strd(), pctx.a.strd(), pctx.a.cntg(),
				pctx.b.data(), spec::gemm_ldb(pctx),
				pctx.a.data(), spec::gemm_lda(pctx),
				pctx.c.data(), pctx.c.cntg_step(),
				pctx.alpha, pctx.beta);
		}
		else {
			return false;
		}
	}

	template<typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const pctx_t<r16, ArchitectureSpec>& pctx) const { return false; }

	template<typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const pctx_t<bf16, ArchitectureSpec>& pctx) const { return false; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const pctx_t<DataType, ArchitectureSpec>& pctx) const {
		return pctx.alpha          != zero<DataType>
		    && pctx.beta_zero_mode != zero_mode::scale
		    && pctx.a.cntg_step() >= 1 && pctx.a.strd_step() >= 1
		    && pctx.b.cntg_step() >= 1 && pctx.b.strd_step() >= 1
		    && pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }
}; // class matmul3_atomic

} // namespace perflibs::linalg::matmul
#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_ATOMIC_HPP
