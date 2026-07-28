/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_SYMM_HEMM_R_REFERENCE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_SYMM_HEMM_R_REFERENCE_HPP

#include "spec/problem_context.hpp"
#include "spec/problem_context_helpers.hpp"
#include "packages/matmul/problem_context_bases.hpp"

#include "matmul_references.hpp"
#include "perflibs_assert.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul3_symm_hemm_r_reference {
	template<typename DataType, typename ArchitectureSpec>
	using sym_problem_context_t = spec::problem_context<
		matmul3<
			general_matrix  <matrix_base<const DataType>>,
			symmetric_matrix<matrix_base<const DataType>>,
			general_matrix  <matrix_base<      DataType>>,
			DataType
		>,
		ArchitectureSpec
	>;

	template<typename DataType, typename ArchitectureSpec>
	using her_problem_context_t = spec::problem_context<
		matmul3<
			general_matrix  <matrix_base<const DataType>>,
			hermitian_matrix<matrix_base<const DataType>>,
			general_matrix  <matrix_base<      DataType>>,
			DataType
		>,
		ArchitectureSpec
	>;

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool impl(const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		using b_data_type = typename ProblemContext::b_matrix_type::value_type;

		if(pctx.c.cntg_step() == 1) {
			const auto side       = side_to_c(PERFLIBS_RIGHT);
			const auto uplo       = uplo_to_c(pctx.b.uplo());
			const pl_linalg_int_t m   = pctx.a.strd();
			const pl_linalg_int_t n   = pctx.b.strd();
			const auto alpha      = pctx.alpha;
			const auto a          = pctx.b.data();
			const pl_linalg_int_t lda = pctx.b.strd_step();
			const auto b          = pctx.a.data();
			const pl_linalg_int_t ldb = pctx.a.cntg_step();
			const auto beta       = pctx.beta;
			      auto c          = pctx.c.data();
			const pl_linalg_int_t ldc = pctx.c.strd_step();

			if constexpr(is_symmetric_matrix_v<typename ProblemContext::b_matrix_type>) {
				reference::symm<b_data_type>(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
			}
			else {
				reference::hemm<b_data_type>(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
			}
		}
		else {
			PERFLIBS_ASSERT(false);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool can_compute_impl(const ProblemContext& pctx) const {
		return false;
		/* TODO doesn't work for symm R
		return ( ( pctx.a.cntg_step() == 1 && pctx.a.strd_step() >= 1 ) || ( pctx.a.strd_step() == 1 && pctx.a.cntg_step() >= 1 ) )
		    && ( ( pctx.b.cntg_step() == 1 && pctx.b.strd_step() >= 1 ) || ( pctx.b.strd_step() == 1 && pctx.b.cntg_step() >= 1 ) )
		    &&   ( pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1 )
		    && pctx.beta_zero_mode == zero_mode::set;
		*/
	}

public:
	static constexpr std::string_view name() { return "matmul3_symm_hemm_r_reference"; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext& pctx) const {
		return false;
	}

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const her_problem_context_t<DataType, ArchitectureSpec>& pctx) const {
		return impl(pctx);
	}

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const sym_problem_context_t<DataType, ArchitectureSpec>& pctx) const {
		return impl(pctx);
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return false;
	}

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const sym_problem_context_t<DataType, ArchitectureSpec>& pctx) const {
		return can_compute_impl(pctx);
	}

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const her_problem_context_t<DataType, ArchitectureSpec>& pctx) const {
		return can_compute_impl(pctx);
	}
}; //class matmul3_symm_hemm_reference_l

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_SYMM_HEMM_R_REFERENCE_HPP
