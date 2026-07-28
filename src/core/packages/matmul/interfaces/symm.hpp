/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SYMM_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SYMM_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include "detect/system.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view symm_name = "?SYMM ";

template<> inline constexpr std::string_view symm_name<r32, r32, r32> = "SSYMM ";
template<> inline constexpr std::string_view symm_name<r64, r64, r64> = "DSYMM ";
template<> inline constexpr std::string_view symm_name<c32, c32, c32> = "CSYMM ";
template<> inline constexpr std::string_view symm_name<c64, c64, c64> = "ZSYMM ";

template<typename IntType, typename AType, typename BType, typename CType>
static bool symm_hemm_param_check(
	const char *side, const char *uplo,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	const CType *c, const IntType *ldc,
	std::string_view name) {

	bool left = option_matches(*side, 'L');
	bool right = option_matches(*side, 'R');
	bool upper = option_matches(*uplo, 'U');
	bool lower = option_matches(*uplo, 'L');

	pl_linalg_int_t info = 0;
	if (!left && !right) {
		info = 1;
	} else if (!lower && !upper) {
		info = 2;
	} else if (*m < 0) {
		info = 3;
	} else if (*n < 0) {
		info = 4;
	} else if (*lda < max(1, right ? *n : *m)) {
		info = 7;
	} else if (*ldb < max(1, *m)) {
		info = 9;
	} else if (*ldc < max(1, *m)) {
		info = 12;
	}
	if (info) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void symm(const char *side, const char *uplo,
	const IntType *m, const IntType *n,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	      CType *c, const IntType *ldc) {

	using scalar_type = promote_t<AType, BType, CType>;

	if constexpr(ParamCheck) {
		const bool res = symm_hemm_param_check(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
			symm_name<AType, BType, CType>);
		if(! res) return;
	}

	if(*m == 0 || *n == 0 || ( *alpha == zero<scalar_type> && *beta == one<scalar_type> ))
		return;

	if (option_matches(*side, 'L')) {
		const    kernel_inttype a_strd         = *m;
		const    kernel_inttype b_strd         = *n;
		const    kernel_inttype cntg           = *m;

		const     auto           a_uplo        = lower_flip(c_to_uplo(*uplo));
		const     auto           a_ptr         = a;
		const     kernel_inttype a_cntg_stride = *lda;
		constexpr kernel_inttype a_strd_stride = 1;

		const     auto           b_ptr         = b;
		constexpr kernel_inttype b_cntg_stride = 1;
		const     kernel_inttype b_strd_stride = *ldb;

		const     auto           c_ptr         = c;
		constexpr kernel_inttype c_cntg_stride = 1;
		const     kernel_inttype c_strd_stride = *ldc;

		spec::problem_context pctx {
			matmul::matmul3 {
				symmetric_matrix { a_uplo, matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
				general_matrix   {         matrix_base { b_ptr, cntg,   b_strd, b_cntg_stride, b_strd_stride } },
				general_matrix   {         matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
				*alpha, *beta
			},
			ArchitectureSpec { machine::get_system_unsafe() }
		};

		matmul::compute(pctx);
	}
	else {
		const     kernel_inttype a_strd        = *m;
		const     kernel_inttype b_strd        = *n;
		const     kernel_inttype cntg          = *n;

		const     auto           a_ptr         = b;
		const     kernel_inttype a_cntg_stride = *ldb;
		constexpr kernel_inttype a_strd_stride = 1;

		const     auto           b_uplo        = c_to_uplo(*uplo);
		const     auto           b_ptr         = a;
		constexpr kernel_inttype b_cntg_stride = 1;
		const     kernel_inttype b_strd_stride = *lda;

		const     auto           c_ptr         = c;
		constexpr kernel_inttype c_cntg_stride = 1;
		const     kernel_inttype c_strd_stride = *ldc;

		spec::problem_context pctx {
			matmul::matmul3 {
				general_matrix   {         matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
				symmetric_matrix { b_uplo, matrix_base { b_ptr, cntg,   b_strd, b_cntg_stride, b_strd_stride } },
				general_matrix   {         matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
				*alpha, *beta
			},
			ArchitectureSpec { machine::get_system_unsafe() }
		};

		matmul::compute(pctx);
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_SYMM_HPP
