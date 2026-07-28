/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_GEMV_KERNELS_HPP
#define PERFLIBS_LINALG_GEMV_KERNELS_HPP

#include "framework/linalg_util.hpp"
#include "framework/gemv_fallback_kernels.hpp"

#include "detect/os.hpp"
#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"

#include "detect/cpu_info.hpp" //features.neon_bf16

namespace perflibs::linalg {
/**
 * Legacy GEMV kernel interface which adheres to the netlib interface
 */
template<typename AType, typename XType, typename YType, typename ScalarType>
using gemv_kernel_t =
	void(const kernel_inttype m, const kernel_inttype n, const ScalarType alpha, const AType *a,
		 const kernel_inttype lda, const XType *x, const kernel_inttype incx, const ScalarType beta,
		 YType *y, const kernel_inttype incy);

/**
 * New GEMV kernel interface, a_strd is the shared dim of A & Y, cntg is the shared dim of A & X
 * Transposition is determined by the multi-stide step parameters of A
 */
template<typename AType, typename BType, typename CType,
         typename ScalarType = promote_t<AType, BType, CType>>
using gemv_new_kernel_t =
	 void(
		const kernel_inttype a_cntg, const kernel_inttype a_strd,
		const ScalarType alpha,
		const AType *a, const kernel_inttype a_cntg_step, const kernel_inttype a_strd_step,
		const BType *x, const kernel_inttype x_cntg_step,
		const ScalarType beta,
		      CType *y, const kernel_inttype y_cntg_step);
/**
 * Declarations of asm kernels
 */
extern "C" {
	perflibs::linalg::gemv_kernel_t<bf16, bf16, bf16, bf16> bgemv_n_bfmlal_16x8;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, bf16, bf16> bgemv_n_fp32_fmla;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, bf16, bf16> bgemv_t_bfdot_4x24;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, bf16, bf16> bgemv_t_fp32_fmla_4x12;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, r32,  r32>  sbgemv_n_bfmlal_16x8;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, r32,  r32>  sbgemv_n_fp32_fmla;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, r32,  r32>  sbgemv_t_bfdot_4x24;
	perflibs::linalg::gemv_kernel_t<bf16, bf16, r32,  r32>  sbgemv_t_fp32_fmla_4x12;
	perflibs::linalg::gemv_kernel_t<r32,  r32,  r32,  r32>  sgemv_t_fmla_4x12;

	perflibs::linalg::gemv_kernel_t<r32, r32, r32, r32> sgemv_t_neon_kernel;
	perflibs::linalg::gemv_kernel_t<r32, r32, r32, r32> sgemv_n_neon_kernel;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_t_neon_kernel_serial;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_t_neon_kernel_parallel;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_n_neon_kernel;
	perflibs::linalg::gemv_kernel_t<c32, c32, c32, c32> cgemv_c_neon_kernel;
	perflibs::linalg::gemv_kernel_t<c32, c32, c32, c32> cgemv_t_neon_kernel;
	perflibs::linalg::gemv_kernel_t<c32, c32, c32, c32> cgemv_n_neon_kernel;
	perflibs::linalg::gemv_kernel_t<c64, c64, c64, c64> zgemv_c_neon_kernel;
	perflibs::linalg::gemv_kernel_t<c64, c64, c64, c64> zgemv_t_neon_kernel;
	perflibs::linalg::gemv_kernel_t<c64, c64, c64, c64> zgemv_n_neon_kernel;

	//TODO: we need to test whether they are actually useful
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_inc2_n_neon_kernel;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_inc2_t_neon_kernel_serial;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_inc2_t_neon_kernel_parallel;

	perflibs::linalg::gemv_kernel_t<r32, r32, r32, r32> sgemv_t_sve_kernel;
	perflibs::linalg::gemv_kernel_t<r32, r32, r32, r32> sgemv_n_sve_kernel;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_t_sve_kernel;
	perflibs::linalg::gemv_kernel_t<r64, r64, r64, r64> dgemv_n_sve_kernel;
	perflibs::linalg::gemv_kernel_t<c32, c32, c32, c32> cgemv_c_sve_kernel;
	perflibs::linalg::gemv_kernel_t<c32, c32, c32, c32> cgemv_t_sve_kernel;
	perflibs::linalg::gemv_kernel_t<c32, c32, c32, c32> cgemv_n_sve_kernel;
	perflibs::linalg::gemv_kernel_t<c64, c64, c64, c64> zgemv_c_sve_kernel;
	perflibs::linalg::gemv_kernel_t<c64, c64, c64, c64> zgemv_t_sve_kernel;
	perflibs::linalg::gemv_kernel_t<c64, c64, c64, c64> zgemv_n_sve_kernel;
} //extern "C"


template<typename AType, typename BType=AType, typename CType=AType,
         typename ScalarType = promote_t<AType, BType, CType>>
struct gemv_kernel_spec {
	gemv_new_kernel_t<AType, BType, CType, ScalarType> *gemv_kernel;
	kernel_inttype num_cols;
	bool scale_b;
	bool apply_beta;

	constexpr
	gemv_kernel_spec() = default;

	constexpr
	gemv_kernel_spec(gemv_new_kernel_t<AType, BType, CType, ScalarType> *kernel, kernel_inttype num_cols, bool scale_b, bool apply_beta)
	:	gemv_kernel { kernel }
	,	num_cols    { num_cols }
	,	scale_b     { scale_b }
	,	apply_beta  { apply_beta }
	{	}
}; //struct gemv_kernel_spec

namespace {

/**
 * The Netlib GEMV interface is inconsistent with other BLAS routines.
 * When trans changes, the meaning of M and N also changes.
 * Historically this treated M as row count and N as column count.
 * This differs from GEMM, where M and N are fixed for the problem.
 *
 * In newly written kernels (and the surrounding LINALG framework), M is the
 * shared dimension of A and C (Y), which we call a_strd.
 *
 * N is the shared dimension of A and B (X) -- which we call cntg.
 *
 * To transpose A, a_cntg_step and a_strd_step are flipped.
 *
 * Legacy kernels adapt kernels written against the Netlib format to the
 * current LINALG style.
 *
 * New GEMV kernels should not use this shim.
 * Implement kernels directly in the LINALG-native interface.
 */
template<bool IsTrans, typename AType, typename XType, typename YType,
         typename ScalarType, gemv_kernel_t<AType, XType, YType, ScalarType> GemvFunc>
PERFLIBS_LINALG_INLINE
void gemv_shim(
	const kernel_inttype a_cntg, const kernel_inttype a_strd,
	const ScalarType alpha,
	const AType *a, const kernel_inttype a_cntg_step, const kernel_inttype a_strd_step,
	const XType *x, const kernel_inttype x_cntg_step,
	const ScalarType beta,
	      YType *y, const kernel_inttype y_cntg_step) {

	if constexpr(IsTrans) {
		PERFLIBS_ASSERT(a_cntg_step == 1, "gemv_shim:trans has been invoked a_cntg_step != 1");

		GemvFunc(
			a_cntg, a_strd,
			alpha,
			a, a_strd_step,
			x, x_cntg_step,
			beta,
			y, y_cntg_step);
	}
	else {
		PERFLIBS_ASSERT(a_strd_step == 1, "gemv_shim:notrans has been invoked a_strd_step != 1");

		GemvFunc(
			a_strd, a_cntg,
			alpha,
			a, a_cntg_step,
			x, x_cntg_step,
			beta,
			y, y_cntg_step);
	}
}


constexpr gemv_kernel_spec bgemv_n_fp32_fmla_spec  { &gemv_shim<false, bf16, bf16, bf16, bf16, bgemv_n_fp32_fmla>,      16, false,  true };
constexpr gemv_kernel_spec bgemv_n_bfmlal_spec     { &gemv_shim<false, bf16, bf16, bf16, bf16, bgemv_n_bfmlal_16x8>,    16, false,  true };
constexpr gemv_kernel_spec bgemv_t_bfdot_spec      { &gemv_shim<true,  bf16, bf16, bf16, bf16, bgemv_t_bfdot_4x24>,      4, false, false };
constexpr gemv_kernel_spec bgemv_t_fp32_fmla_spec  { &gemv_shim<true,  bf16, bf16, bf16, bf16, bgemv_t_fp32_fmla_4x12>,  4, false, false };

constexpr gemv_kernel_spec sbgemv_n_fp32_fmla_spec { &gemv_shim<false, bf16, bf16,  r32,  r32, sbgemv_n_fp32_fmla>,     16, false,  true };
constexpr gemv_kernel_spec sbgemv_n_bfmlal_spec    { &gemv_shim<false, bf16, bf16,  r32,  r32, sbgemv_n_bfmlal_16x8>,   16, false,  true };
constexpr gemv_kernel_spec sbgemv_t_bfdot_spec     { &gemv_shim<true,  bf16, bf16,  r32,  r32, sbgemv_t_bfdot_4x24>,     4, false, false };
constexpr gemv_kernel_spec sbgemv_t_fp32_fmla_spec { &gemv_shim<true,  bf16, bf16,  r32,  r32, sbgemv_t_fp32_fmla_4x12>, 4, false, false };
constexpr gemv_kernel_spec sgemv_t_fmla_spec       { &gemv_shim<true,  r32,  r32,   r32,  r32, sgemv_t_fmla_4x12>,       4, false, false };

constexpr gemv_kernel_spec<r32, r32, r32, r32> sgemv_n_neon_spec {
	&gemv_shim<false, r32, r32, r32, r32, sgemv_n_neon_kernel>, 14, true, true
};

constexpr gemv_kernel_spec<r32, r32, r32, r32> sgemv_t_neon_spec {
	&gemv_shim<true, r32, r32, r32, r32, sgemv_t_neon_kernel>, 14, false, true
};
constexpr gemv_kernel_spec<r64, r64, r64, r64> dgemv_n_neon_spec {
	&gemv_shim<false, r64, r64, r64, r64, dgemv_n_neon_kernel>, 14, true, true
};
constexpr gemv_kernel_spec<r64, r64, r64, r64> dgemv_t_neon_parallel_spec {
	&gemv_shim<true, r64, r64, r64, r64, dgemv_t_neon_kernel_parallel>, 8, false, true
};
constexpr gemv_kernel_spec<r64, r64, r64, r64> dgemv_t_neon_serial_spec {
	&gemv_shim<true, r64, r64, r64, r64, dgemv_t_neon_kernel_serial>, 14, false, true
};
constexpr gemv_kernel_spec<c32, c32, c32, c32> cgemv_n_neon_spec {
	&gemv_shim<false, c32, c32, c32, c32, cgemv_n_neon_kernel>, 7, true, true
};
constexpr gemv_kernel_spec<c32, c32, c32, c32> cgemv_t_neon_spec {
	&gemv_shim<true, c32, c32, c32, c32, cgemv_t_neon_kernel>, 7, false, true
};
constexpr gemv_kernel_spec<c32, c32, c32, c32> cgemv_c_neon_spec {
	&gemv_shim<true, c32, c32, c32, c32, cgemv_c_neon_kernel>, 7, false, true
};
constexpr gemv_kernel_spec<c64, c64, c64, c64> zgemv_n_neon_spec {
	&gemv_shim<false, c64, c64, c64, c64, zgemv_n_neon_kernel>, 7, true, true
};
constexpr gemv_kernel_spec<c64, c64, c64, c64> zgemv_t_neon_spec {
	&gemv_shim<true, c64, c64, c64, c64, zgemv_t_neon_kernel>, 7, false, true
};
constexpr gemv_kernel_spec<c64, c64, c64, c64> zgemv_c_neon_spec {
	&gemv_shim<true, c64, c64, c64, c64, zgemv_c_neon_kernel>, 7, false, true
};
///SVE
constexpr gemv_kernel_spec<r32, r32, r32, r32> sgemv_n_sve_spec {
	&gemv_shim<false, r32, r32, r32, r32, sgemv_n_sve_kernel>, 14, true, true
};
constexpr gemv_kernel_spec<r32, r32, r32, r32> sgemv_t_sve_spec {
	&gemv_shim<true, r32, r32, r32, r32, sgemv_t_sve_kernel>, 14, false, true
};
constexpr gemv_kernel_spec<r64, r64, r64, r64> dgemv_n_sve_spec {
	&gemv_shim<false, r64, r64, r64, r64, dgemv_n_sve_kernel>, 14, true, true
};
constexpr gemv_kernel_spec<r64, r64, r64, r64> dgemv_t_sve_spec {
	&gemv_shim<true, r64, r64, r64, r64, dgemv_t_sve_kernel>, 14, false, true
};
constexpr gemv_kernel_spec<c32, c32, c32, c32> cgemv_n_sve_spec {
	&gemv_shim<false, c32, c32, c32, c32, cgemv_n_sve_kernel>, 7, true, true
};
constexpr gemv_kernel_spec<c32, c32, c32, c32> cgemv_t_sve_spec {
	&gemv_shim<true, c32, c32, c32, c32, cgemv_t_sve_kernel>, 7, false, true
};
constexpr gemv_kernel_spec<c32, c32, c32, c32> cgemv_c_sve_spec {
	&gemv_shim<true, c32, c32, c32, c32, cgemv_c_sve_kernel>, 7, false, true
};
constexpr gemv_kernel_spec<c64, c64, c64, c64> zgemv_n_sve_spec {
	&gemv_shim<false, c64, c64, c64, c64, zgemv_n_sve_kernel>, 7, true, true
};
constexpr gemv_kernel_spec<c64, c64, c64, c64> zgemv_t_sve_spec {
	&gemv_shim<true, c64, c64, c64, c64, zgemv_t_sve_kernel>, 7, false, true
};
constexpr gemv_kernel_spec<c64, c64, c64, c64> zgemv_c_sve_spec {
	&gemv_shim<true, c64, c64, c64, c64, zgemv_c_sve_kernel>, 7, false, true
};

template<typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
constexpr gemv_kernel_spec<AType, BType, CType, ScalarType> gemv_n_fallback_spec {
	&gemv_a_strd_first<false, ArchitectureSpec, AType, BType, CType, ScalarType>, 1, true, false
};

template<typename AType, typename CType, typename ArchitectureSpec>
constexpr gemv_kernel_spec<AType, bf16, CType, r32> gemv_n_fallback_spec<AType, bf16, CType, r32, ArchitectureSpec> {
	&gemv_a_strd_first<false, ArchitectureSpec, AType, bf16, CType, r32>, 1, false, false
};

template<bool IsConj, typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
constexpr gemv_kernel_spec<AType, BType, CType, ScalarType> gemv_t_fallback_spec {
	&gemv_a_cntg_first<IsConj, ArchitectureSpec, AType, BType, CType, ScalarType>, 1, false, false
};

template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType>
using gemv_problem_kernel_spec_t =
	gemv_kernel_spec<
		std::remove_cv_t<typename AMatrixType::value_type>,
		std::remove_cv_t<typename BMatrixType::value_type>,
		std::remove_cv_t<typename CMatrixType::value_type>,
		ScalarType>;

template<
	typename AMatrixType,
	typename BMatrixType,
	typename CMatrixType,
	typename ScalarType,
	typename ArchitectureSpec
>
PERFLIBS_LINALG_INLINE
gemv_problem_kernel_spec_t<AMatrixType, BMatrixType, CMatrixType, ScalarType> get_gemv_fallback_spec(
	const spec::problem_context<matmul::matmul3<
		AMatrixType, BMatrixType, CMatrixType,
		ScalarType>, ArchitectureSpec>& pctx) {

	using a_data_type = std::remove_cv_t<typename AMatrixType::value_type>;
	using b_data_type = std::remove_cv_t<typename BMatrixType::value_type>;
	using c_data_type = std::remove_cv_t<typename CMatrixType::value_type>;

	if constexpr(is_complex_v<ScalarType>) {
		if(pctx.a.is_conj()) {
			return gemv_t_fallback_spec<true, a_data_type, b_data_type, c_data_type, ScalarType, ArchitectureSpec>;
		}
	}

	return is_cntg_contig(pctx.a)
		 ? gemv_t_fallback_spec<false, a_data_type, b_data_type, c_data_type, ScalarType, ArchitectureSpec>
		 : gemv_n_fallback_spec<a_data_type, b_data_type, c_data_type, ScalarType, ArchitectureSpec>;
}

template <typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_gemv_neon_kernel(const ProblemContext& pctx, kernel_inttype max_threads) {

	if constexpr (os::windows_arm64ec) {
		return get_gemv_fallback_spec(pctx);
	}

	using scalar_type = typename ProblemContext::scalar_type;
	using a_data_type = std::remove_cv_t<typename ProblemContext::a_matrix_type::value_type>;
	using b_data_type = std::remove_cv_t<typename ProblemContext::b_matrix_type::value_type>;
	using c_data_type = std::remove_cv_t<typename ProblemContext::c_matrix_type::value_type>;

	if constexpr(std::is_same_v<scalar_type, bf16>
	          && std::is_same_v<a_data_type, bf16>
	          && std::is_same_v<b_data_type, bf16>
	          && std::is_same_v<c_data_type, bf16>) {

		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			const auto features = perflibs::machine::cpu_info::get_cpu_features();
			if (is_cntg_contig(pctx.a)) {

				if(features.neon_bf16) {
					return bgemv_t_bfdot_spec;
				}
				else {
					return bgemv_t_fp32_fmla_spec;
				}
			}
			else {
				if(features.neon_bf16) {
					return bgemv_n_bfmlal_spec;
				}
				return bgemv_n_fp32_fmla_spec;
			}
		}
	}
	else if constexpr(std::is_same_v<scalar_type, r32>
	               && std::is_same_v<a_data_type, bf16>
	               && std::is_same_v<b_data_type, bf16>
	               && std::is_same_v<c_data_type, r32>) {

		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			const auto features = perflibs::machine::cpu_info::get_cpu_features();
			if (is_cntg_contig(pctx.a)) {
				if(features.neon_bf16) {
					return sbgemv_t_bfdot_spec;
				}
				else {
					return sbgemv_t_fp32_fmla_spec;
				}
			}
			else {
				if(features.neon_bf16) {
					return sbgemv_n_bfmlal_spec;
				}
				else {
					return sbgemv_n_fp32_fmla_spec;
				}
			}
		}
	}
	else if constexpr(std::is_same_v<a_data_type, r32>
	               && std::is_same_v<b_data_type, r32>
	               && std::is_same_v<c_data_type, r32>) {

		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				return sgemv_t_neon_spec;
			}
			else {
				return sgemv_n_neon_spec;
			}
		}
	}
	else if constexpr(std::is_same_v<a_data_type, r64>
                   && std::is_same_v<b_data_type, r64>
	               && std::is_same_v<c_data_type, r64>) {
		if (is_cntg_contig(pctx.a)) {
			if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
				if (max_threads == 1) {
					return dgemv_t_neon_serial_spec;
				}
				else {
					return dgemv_t_neon_parallel_spec;
				}
			}
		}
		else if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			return dgemv_n_neon_spec;
		}
	}
	else if constexpr(std::is_same_v<a_data_type, c32>
	               && std::is_same_v<b_data_type, c32>
	               && std::is_same_v<c_data_type, c32>) {

		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				if (pctx.a.is_conj()) {
					return cgemv_c_neon_spec;
				}
				else {
					return cgemv_t_neon_spec;
				}
			}
			else if (is_strd_contig(pctx.a)) {
				if (!pctx.a.is_conj()) {
					return cgemv_n_neon_spec;
				}
			}
		}
	}
	else if constexpr(std::is_same_v<a_data_type, c64>
	               && std::is_same_v<b_data_type, c64>
	               && std::is_same_v<c_data_type, c64>) {

		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				if (pctx.a.is_conj()) {
					return zgemv_c_neon_spec;
				}
				else {
					return zgemv_t_neon_spec;
				}
			}
			else if (is_strd_contig(pctx.a)) {
				if (!pctx.a.is_conj()) {
					return zgemv_n_neon_spec;
				}
			}
		}
	}

	return get_gemv_fallback_spec(pctx);
}

template <typename ProblemContext>
PERFLIBS_LINALG_INLINE
gemv_problem_kernel_spec_t<typename ProblemContext::a_matrix_type, typename ProblemContext::b_matrix_type,
                           typename ProblemContext::c_matrix_type, typename ProblemContext::scalar_type>
get_gemv_sve_kernel(const ProblemContext& pctx, kernel_inttype max_threads) {
	using scalar_type = typename ProblemContext::scalar_type;
	using a_data_type = std::remove_cv_t<typename ProblemContext::a_matrix_type::value_type>;
	using b_data_type = std::remove_cv_t<typename ProblemContext::b_matrix_type::value_type>;
	using c_data_type = std::remove_cv_t<typename ProblemContext::c_matrix_type::value_type>;

	if constexpr(std::is_same_v<scalar_type, r32>
	          && std::is_same_v<a_data_type, r32>
	          && std::is_same_v<b_data_type, r32>
	          && std::is_same_v<c_data_type, r32>) {
		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				return sgemv_t_sve_spec;
			}
			else {
				return sgemv_n_sve_spec;
			}
		}
	}
	else if constexpr(std::is_same_v<scalar_type, r64>
	               && std::is_same_v<a_data_type, r64>
	               && std::is_same_v<b_data_type, r64>
	               && std::is_same_v<c_data_type, r64>) {
		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				return dgemv_t_sve_spec;
			}
			else {
				return dgemv_n_sve_spec;
			}
		}
	}
	else if constexpr(std::is_same_v<scalar_type, c32>
	               && std::is_same_v<a_data_type, c32>
	               && std::is_same_v<b_data_type, c32>
	               && std::is_same_v<c_data_type, c32>) {
		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				if (pctx.a.is_conj()) {
					return cgemv_c_sve_spec;
				}
				else {
					return cgemv_t_sve_spec;
				}
			}
			else {
				return cgemv_n_sve_spec;
			}
		}
	}
	else if constexpr(std::is_same_v<scalar_type, c64>
	               && std::is_same_v<a_data_type, c64>
	               && std::is_same_v<b_data_type, c64>
	               && std::is_same_v<c_data_type, c64>) {
		if (gemv_incx(pctx) == 1 && gemv_incy(pctx) == 1) {
			if (is_cntg_contig(pctx.a)) {
				if (pctx.a.is_conj()) {
					return zgemv_c_sve_spec;
				}
				else {
					return zgemv_t_sve_spec;
				}
			}
			else {
				return zgemv_n_sve_spec;
			}
		}
	}

	return get_gemv_neon_kernel(pctx, max_threads);
}

}
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_GEMV_KERNELS_HPP
