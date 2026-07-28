/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GEMM_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GEMM_HPP

#include "packages/matmul/strategies.hpp"
#include "packages/matmul/strategies/backstop.hpp"

#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include "detect/cpu_info.hpp"
#include "detect/os.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view gemm_name = "?GEMM ";

template<> inline constexpr std::string_view gemm_name<bf16, bf16, r32>  = "SBGEMM";
template<> inline constexpr std::string_view gemm_name<bf16, bf16, bf16> = "BGEMM";
template<> inline constexpr std::string_view gemm_name<r16,  r16,  r16>  = "HGEMM ";
template<> inline constexpr std::string_view gemm_name<r32,  r32,  r32>  = "SGEMM ";
template<> inline constexpr std::string_view gemm_name<r64,  r64,  r64>  = "DGEMM ";
template<> inline constexpr std::string_view gemm_name<c32,  c32,  c32>  = "CGEMM ";
template<> inline constexpr std::string_view gemm_name<c64,  c64,  c64>  = "ZGEMM ";

template<typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
__attribute__((noinline))
void matmul::compute(const spec::problem_context<
	matmul::matmul3<
		general_matrix<matrix_base<const AType>>,
		general_matrix<matrix_base<const BType>>,
		general_matrix<matrix_base<      CType>>,
		ScalarType
	>,
	ArchitectureSpec>& pctx) {

	matmul::compute_impl(pctx);
}

template<typename IntType, typename AType, typename BType, typename CType>
PERFLIBS_LINALG_INLINE
bool gemm_param_check(
	const char *transa, const char *transb, const IntType *m,
	const IntType *n, const IntType *k,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc,
	std::string_view name) {

	IntType nota = option_matches(*transa, 'N');
	IntType notb = option_matches(*transb, 'N');

	IntType nrowa = nota ? *m : *k;
	IntType nrowb = notb ? *k : *n;

	//regardless of IntType, info needs to match xerbla's as it is passed by pointer
	pl_linalg_int_t info = 0;
	if (! nota && ! option_matches(*transa, 'C', 'T')) {
		info = 1;
	} else if (! notb && ! option_matches(*transb, 'C', 'T')) {
		info = 2;
	} else if (*m < 0) {
		info = 3;
	} else if (*n < 0) {
		info = 4;
	} else if (*k < 0) {
		info = 5;
	} else if (*lda < max(1,nrowa)) {
		info = 8;
	} else if (*ldb < max(1,nrowb)) {
		info = 10;
	} else if (*ldc < max(1,*m)) {
		info = 13;
	}
	if (info != 0) {
		call_xerbla(name, info);
	return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gemm(
	const char *transa, const char *transb,
	const IntType *m, const IntType *n, const IntType *k,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc,
	std::string_view routine_name) {

	if constexpr(ParamCheck) {
		const bool res = gemm_param_check(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, routine_name);
		if(! res) return;
	}

	if(*m == 0 || *n == 0)
		return;

	const auto a_trans                 = c_to_trans(*transa);
	const bool is_a_trans              = is_trans(a_trans);
	const auto b_trans                 = c_to_trans(*transb);
	const bool is_b_trans              = is_trans(b_trans);

	const kernel_inttype a_strd        = *m;
	const kernel_inttype b_strd        = *n;
	const kernel_inttype cntg          = *k;

	const kernel_inttype a_cntg_stride = is_a_trans ? 1 : *lda;
	const kernel_inttype a_strd_stride = is_a_trans ? *lda : 1;

	const kernel_inttype b_cntg_stride = is_b_trans ? *ldb : 1;
	const kernel_inttype b_strd_stride = is_b_trans ? 1 : *ldb;

	const kernel_inttype c_cntg_stride = 1;
	const kernel_inttype c_strd_stride = *ldc;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix { matrix_base { a, cntg,   a_strd, a_cntg_stride, a_strd_stride }, is_conj(a_trans) },
			general_matrix { matrix_base { b, cntg,   b_strd, b_cntg_stride, b_strd_stride }, is_conj(b_trans) },
			general_matrix { matrix_base { c, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	if constexpr(std::is_same_v<AType, half>) {
		const auto features = perflibs::machine::cpu_info::get_cpu_features();
		// We don't have a valid optimized HGEMM kernel for Arm64EC
		// so use the backstop that promotes to FP32
		if (os::windows_arm64ec || !features.fphp) {
			matmul::backstop{}(pctx);
			return;
		}
	}

	matmul::compute(pctx);
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gemm(
	const char *transa, const char *transb,
	const IntType *m, const IntType *n, const IntType *k,
	const promote_t<AType, BType, CType> *alpha,
    const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc) {

	gemm<ParamCheck, IntType, AType, BType, CType, ArchitectureSpec>(
		transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
		gemm_name<AType, BType, CType>);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GEMM_HPP
