/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_SYR2K_HER2K_REFERENCE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_SYR2K_HER2K_REFERENCE_HPP

#include "spec/problem_context.hpp"
#include "spec/problem_context_helpers.hpp"
#include "packages/matmul/problem_context_bases.hpp"

#include "matmul_references.hpp"
#include "perflibs_assert.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class syr2k_her2k_reference {
	template<typename DataType, typename ArchitectureSpec>
	using sym_problem_context_t = spec::problem_context<
		rank_update_2k<
			general_matrix  <matrix_base<const DataType>>,
			general_matrix  <matrix_base<const DataType>>,
			symmetric_matrix<matrix_base<      DataType>>,
			DataType
		>,
		ArchitectureSpec
	>;

	template<typename DataType, typename ArchitectureSpec>
	using her_problem_context_t = spec::problem_context<
		rank_update_2k<
			general_matrix  <matrix_base<const DataType>>,
			general_matrix  <matrix_base<const DataType>>,
			hermitian_matrix<matrix_base<      DataType>>,
			DataType
		>,
		ArchitectureSpec
	>;

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool impl(const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		using c_data_type = typename ProblemContext::c_matrix_type::value_type;

		const auto uplo       = uplo_to_c(pctx.c.uplo());
		const auto transa     = pctx.a.cntg_step() == 0 || is_strd_contig(pctx.a)
		                      ? 'N'
		                      : pctx.a.is_conj()
		                      ? 'C'
		                      : 'T';
		const pl_linalg_int_t n   = pctx.a.strd();
		const pl_linalg_int_t k   = pctx.a.cntg();
		const auto alpha      = pctx.alpha;
		const auto a          = pctx.a.data();
		const pl_linalg_int_t lda = syrk_lda(pctx);
		const auto b          = pctx.b.data();
		const pl_linalg_int_t ldb = syr2k_ldb(pctx);
		      auto c          = pctx.c.data();
		const pl_linalg_int_t ldc = pctx.c.strd_step();


		if constexpr(is_symmetric_matrix_v<typename ProblemContext::c_matrix_type>) {
			const auto beta  = pctx.beta;
			reference::syr2k<c_data_type>(&uplo, &transa, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}
		else {
			PERFLIBS_ASSERT(pctx.beta.imag()  == zero<typename ProblemContext::scalar_type::value_type>, "her2k_reference narrows complex beta to real");

			const auto beta  = pctx.beta.real();

			reference::her2k<c_data_type>(&uplo, &transa, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool can_compute_impl(const ProblemContext& pctx) const {
		return ( ( pctx.a.cntg_step() == 1 && pctx.a.strd_step() >= 1 ) || ( pctx.a.strd_step() == 1 && pctx.a.cntg_step() >= 1 ) )
		    && ( ( pctx.b.cntg_step() == 1 && pctx.b.strd_step() >= 1 ) || ( pctx.b.strd_step() == 1 && pctx.b.cntg_step() >= 1 ) )
		    &&   ( pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1 );
	}

public:
	static constexpr std::string_view name() { return "syr2k_her2k_reference"; }

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
}; //class syr2k_her2k_reference

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_SYR2K_HER2K_REFERENCE_HPP
