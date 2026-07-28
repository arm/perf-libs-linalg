/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_GEMM_REFERENCE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_GEMM_REFERENCE_HPP

#include "spec/problem_context.hpp"
#include "spec/problem_context_helpers.hpp"
#include "packages/matmul/problem_context_bases.hpp"

#include "matmul_references.hpp"

#include "perflibs_assert.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul3_gemm_reference {
	template<typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
	using problem_context_t = spec::problem_context<
		matmul3<
			general_matrix<matrix_base<const AType>>,
			general_matrix<matrix_base<const BType>>,
			general_matrix<matrix_base<      CType>>,
			ScalarType
		>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "matmul3_gemm_reference"; }

	template<typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const problem_context_t<AType, BType, CType, ScalarType, ArchitectureSpec>& pctx) const {
		if constexpr( reference::gemm<AType, BType, CType, ScalarType> == nullptr  ) {
			return false;
		}

		if (! this->can_compute(pctx) ) return false;

		const auto transa     = gemm_transa(pctx);
		const auto transb     = gemm_transb(pctx);
		const pl_linalg_int_t m   = pctx.a.strd();
		const pl_linalg_int_t n   = pctx.b.strd();
		const pl_linalg_int_t k   = pctx.a.cntg();
		const auto alpha      = pctx.alpha;
		const auto a          = pctx.a.data();
		const pl_linalg_int_t lda = gemm_lda(pctx);
		const auto b          = pctx.b.data();
		const pl_linalg_int_t ldb = gemm_ldb(pctx);
		const auto beta       = pctx.beta;
		      auto c          = pctx.c.data();
		//if we've got here from a vector interface like GEMV) then we strd_step may be set to 1 or 0
		const pl_linalg_int_t ldc = n == 1 ? m : pctx.c.strd_step();

		if(pctx.c.cntg_step() == 1) {
			reference::gemm<AType, BType, CType, ScalarType>(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}
		else {
			PERFLIBS_ASSERT(false); //ensure this is actually correct
			//reference::gemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext& pctx) const { return false; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

	template<typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const problem_context_t<AType, BType, CType, ScalarType, ArchitectureSpec>& pctx) const {
		if constexpr( reference::gemm<AType, BType, CType, ScalarType> == nullptr  ) {
			return false;
		}

		return ( ( pctx.a.cntg_step() == 1 && pctx.a.strd_step() >= 1 ) || ( pctx.a.strd_step() == 1 && pctx.a.cntg_step() >= 1 ) )
		    && ( ( pctx.b.cntg_step() == 1 && pctx.b.strd_step() >= 1 ) || ( pctx.b.strd_step() == 1 && pctx.b.cntg_step() >= 1 ) )
		    &&   ( pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1 )
		    && pctx.beta_zero_mode == zero_mode::set
		    // BLAS interface cannot encode conjugate non-transpose cases.
		    && !( (gemm_transa(pctx) == 'N' && pctx.a.is_conj())
		       || (gemm_transb(pctx) == 'N' && pctx.b.is_conj()) );
	}
}; //class matmul3_gemm_reference

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_GEMM_REFERENCE_HPP
