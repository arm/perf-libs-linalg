/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_SYR2K_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_SYR2K_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view syr2k_name = "?SYR2K ";

template<> inline constexpr std::string_view syr2k_name<r32, r32, r32> = "SSYR2K ";
template<> inline constexpr std::string_view syr2k_name<r64, r64, r64> = "DSYR2K ";
template<> inline constexpr std::string_view syr2k_name<c32, c32, c32> = "CSYR2K ";
template<> inline constexpr std::string_view syr2k_name<c64, c64, c64> = "ZSYR2K ";

template<typename IntType, typename AType, typename BType, typename CType>
PERFLIBS_LINALG_INLINE
bool syr2k_param_check(
	const char *uplo, const char *trans,
	const IntType *n, const IntType *k,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	const CType *c, const IntType *ldc,
	std::string_view name) {

	constexpr bool not_cmplx = ! is_complex_v<promote_t<AType, CType>>;

	IntType nrowa, info, upper;

	if (option_matches(*trans, 'N')) {
		nrowa = *n;
	}
	else {
		nrowa = *k;
	}
	upper = option_matches(*uplo, 'U');

	info = 0;
	if (! upper && ! option_matches(*uplo, 'L')) {
		info = 1;
	}
	else if (!(option_matches(*trans, 'N', 'T') ||
		(not_cmplx && option_matches(*trans, 'C')))) {
		info = 2;
	}
	else if (*n < 0) {
		info = 3;
	}
	else if (*k < 0) {
		info = 4;
	}
	else if (*lda < max(1,nrowa)) {
		info = 7;
	}
	else if (*ldb < max(1,nrowa)) {
		info = 9;
	}
	else if (*ldc < max(1,*n)) {
		info = 12;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

/**
 * linalg interface to SYR2K, converts Fortran types into linalg objects
 */
template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void syr2k(const char *uplo, const char *trans,
           const IntType *n, const IntType *k,
           const promote_t<AType, BType, CType> *alpha,
           const AType *a, const IntType *lda,
           const BType *b, const IntType *ldb,
           const promote_t<AType, BType, CType> *beta,
                 CType *c, const IntType *ldc) {

	if constexpr(ParamCheck) {
		const bool res = syr2k_param_check(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
			syr2k_name<AType, BType, CType>);
		if(! res) return;
	}

	if(*n == 0 || ((*k ==0 || *alpha == zero<>) && *beta == one<>))
		return;

	const     kernel_inttype a_strd        = *n;
	const     kernel_inttype b_strd        = *n;
	const     kernel_inttype cntg          = *k;

	const     auto           ab_trans       = c_to_trans(*trans);
	const     auto           ab_is_trans    = is_trans( ab_trans );

	const     auto           a_ptr         = a;
	const     kernel_inttype a_cntg_stride = ab_is_trans ? 1 : *lda;
	const     kernel_inttype a_strd_stride = ab_is_trans ? *lda : 1;

	const     auto           b_ptr         = b;
	const     kernel_inttype b_cntg_stride = ab_is_trans ? 1 : *ldb;
	const     kernel_inttype b_strd_stride = ab_is_trans ? *ldb : 1;

	const     auto           c_uplo        = c_to_uplo(*uplo);
	const     auto           c_ptr         = c;
	constexpr kernel_inttype c_cntg_stride = 1;
	const     kernel_inttype c_strd_stride = *ldc;

	spec::problem_context pctx {
		matmul::rank_update_2k {
			general_matrix   {         matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix   {         matrix_base { b_ptr, cntg,   a_strd, b_cntg_stride, b_strd_stride } },
			symmetric_matrix { c_uplo, matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_SYR2K_HPP
