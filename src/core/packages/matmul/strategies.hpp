/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_MATMUL_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_MATMUL_STRATEGIES_HPP

#include "framework/compute.hpp"

#include "packages/matmul/strategies/backstop.hpp"

#include "packages/matmul/strategies/matmul3_atomic.hpp"
#include "packages/matmul/strategies/matmul3_vector_scalar.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_basic.hpp"
#include "packages/matmul/strategies/matmul3_inner_product.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_large.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_large_no_sync.hpp"
#include "packages/matmul/strategies/matmul3_matrix_vector.hpp"
#include "packages/matmul/strategies/matmul3_outer_product.hpp"

#include "packages/matmul/strategies/matmul3_gemm_reference.hpp"
#include "packages/matmul/strategies/matmul3_symm_hemm_l_reference.hpp"
#include "packages/matmul/strategies/matmul3_symm_hemm_r_reference.hpp"

#include "packages/matmul/strategies/matmul3_interleaved_sequential.hpp"
#include "packages/matmul/strategies/matmul3_unpacked.hpp"



#include "packages/matmul/strategies/compressed_general_matrix_vector.hpp"
#include "packages/matmul/strategies/compressed_rank_one_update.hpp"
#include "packages/matmul/strategies/compressed_rank_two_update.hpp"
#include "packages/matmul/strategies/compressed_symmetric_matrix_vector.hpp"
#include "packages/matmul/strategies/compressed_triangular_matrix_vector.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_tri_gen_gen.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_gen_tri_gen.hpp"
#include "packages/matmul/strategies/syrk_herk_reference.hpp"
#include "packages/matmul/strategies/syr2k_her2k_reference.hpp"
#include "packages/matmul/strategies/trmm_reference.hpp"
#include "packages/matmul/strategies/set_or_scale.hpp"
#include "packages/matmul/strategies/symmetric_matrix_vector.hpp"
#include "packages/matmul/strategies/rank_k_update_basic.hpp"
#include "packages/matmul/strategies/rank_k_update_large.hpp"
#include "packages/matmul/strategies/rank_one_update.hpp"
#include "packages/matmul/strategies/rank_update_2k_interleaved.hpp"
#include "packages/matmul/strategies/rank_two_update.hpp"
#include "packages/matmul/strategies/inplace_matmul_large.hpp"
#include "packages/matmul/strategies/inplace_matmul_vector.hpp"
#include "packages/matmul/strategies/naive_matmul.hpp"
#include "packages/matmul/strategies/matadd2_buffer_naive.hpp"
#include "packages/matmul/strategies/matadd3_naive.hpp"
#include "packages/matmul/strategies/matadd3_zero_scalar.hpp"

#include "packages/matmul/strategies/matmul4_vector_scalar.hpp"

namespace perflibs::linalg {

//all of the strategies available to use to compute a GEMM-like routine
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<matmul::matmul4<T...>, ArchitectureSpec>> = std::tuple {
	matmul::matmul4_vector_scalar_strategy { },
	matmul::out_of_place_naive_matmul { },
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<matmul::matmul3<T...>, ArchitectureSpec>> = std::tuple {
	matmul::set_or_scale { },
	matmul::compressed_general_matrix_vector { },
	matmul::symmetric_matrix_vector { },
	matmul::compressed_symmetric_matrix_vector { },
	matmul::compressed_rank_one_update { },
	matmul::matmul3_interleaved_tri_gen_gen { },
	matmul::matmul3_interleaved_gen_tri_gen { },
	matmul::matmul3_atomic { },
	matmul::matmul3_inner_product { },
	matmul::matmul3_vector_scalar { },
	matmul::matmul3_matrix_vector { },
	matmul::matmul3_outer_product { },
	matmul::matmul3_unpacked { },
	matmul::matmul3_interleaved_basic { },
	matmul::matmul3_interleaved_sequential { },
	matmul::matmul3_interleaved_large { },
	matmul::matmul3_interleaved_large_no_sync<false> { },
	matmul::rank_k_update_large { },
	matmul::rank_k_update_basic { },
	matmul::rank_one_update { },
	matmul::matmul3_gemm_reference { },
	matmul::matmul3_symm_hemm_l_reference { },
	matmul::matmul3_symm_hemm_r_reference { },
	matmul::syrk_herk_reference { },
	matmul::backstop { },
	matmul::matmul3_interleaved_large_no_sync<true> { }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<matmul::rank_update_2k<T...>, ArchitectureSpec>> = std::tuple {
	/*
	 * When the scalars and dimensions of the matrix
	 * mean there is actually no need to compute dot products
	 * out of A & B, then we can get away with just scaling or setting
	 * the values of the output matrix C
	 */
	matmul::set_or_scale { },
	/*
	 * Packed symmetric / Hermitian rank one update
	 */
	matmul::compressed_rank_two_update { },
	/*
	 * Parallel strategy for large rank Two updates
	 */
    matmul::rank_two_update { },
	/*
	 * Parallel strategy for large rank 2k updates
	 */
    matmul::rank_update_2k_interleaved { },
	/*
	 * If execution reaches this point, no optimized strategy matched
	 * and if it has come via the BLAS
	 * interface then we can use the BLAS reference to do the compute
	 */
	matmul::syr2k_her2k_reference { }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<matmul::matmul2<T...>, ArchitectureSpec>> = std::tuple {
	matmul::set_or_scale { },
	matmul::inplace_matmul_vector { },
	matmul::inplace_matmul_large { },
	matmul::compressed_triangular_matrix_vector { },
	matmul::trmm_reference { }
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<matmul::matadd2<T...>, ArchitectureSpec>> = std::tuple {
	/*
	 * Buffered in-place matadd fallback
	 */
	matmul::matadd2_buffer_naive { },
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<matmul::matadd3<T...>, ArchitectureSpec>> = std::tuple {
	/*
	 * Zero-scalar degenerate omatadd cases
	 */
	matmul::matadd3_zero_scalar { },
	/*
	 * Standard out-of-place matadd
	 */
	matmul::matadd3_naive { },
};


namespace matmul {

template<typename ProblemContext>
void compute_impl(ProblemContext& pctx) {

	set_or_scale sos { };

	if(sos(pctx)) {
		return;
	}

	::perflibs::linalg::compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<matmul::matmul2<T...>, ArchitectureSpec>& pctx) {
	matmul::compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<matmul::matmul3<T...>, ArchitectureSpec>& pctx) {
	matmul::compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<matmul::matmul4<T...>, ArchitectureSpec>& pctx) {
	matmul::compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<matmul::rank_update_2k<T...>, ArchitectureSpec>& pctx) {
	matmul::compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<matmul::matadd2<T...>, ArchitectureSpec>& pctx) {
	matmul::compute_impl(pctx);
}

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<matmul::matadd3<T...>, ArchitectureSpec>& pctx) {
	matmul::compute_impl(pctx);
}

} //namespace matmul

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PACKAGE_MATMUL_STRATEGIES_HPP
