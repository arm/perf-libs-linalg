/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_GEMMT_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_GEMMT_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view gemmt_name;

template<typename AType, typename BType, typename CType>
inline constexpr std::string_view gemmtr_name;

template<> inline constexpr std::string_view gemmt_name<r32, r32, r32> = "SGEMMT ";
template<> inline constexpr std::string_view gemmt_name<r64, r64, r64> = "DGEMMT ";
template<> inline constexpr std::string_view gemmt_name<c32, c32, c32> = "CGEMMT ";
template<> inline constexpr std::string_view gemmt_name<c64, c64, c64> = "ZGEMMT ";

template<> inline constexpr std::string_view gemmtr_name<r32, r32, r32> = "SGEMMTR";
template<> inline constexpr std::string_view gemmtr_name<r64, r64, r64> = "DGEMMTR";
template<> inline constexpr std::string_view gemmtr_name<c32, c32, c32> = "CGEMMTR";
template<> inline constexpr std::string_view gemmtr_name<c64, c64, c64> = "ZGEMMTR";

template<typename IntType, typename AType, typename BType, typename CType>
PERFLIBS_LINALG_INLINE
bool gemmt_param_check(
	const char *uplo, const char *transa, const char *transb,
	const IntType *n, const IntType *k,
	const promote_t<AType, BType, CType> *alpha,
	const AType *a, const IntType *lda,
	const BType *b, const IntType *ldb,
	const promote_t<AType, BType, CType> *beta,
	const CType *c, const IntType *ldc,
	std::string_view name) {

	IntType nota = option_matches(*transa, 'N');
	IntType notb = option_matches(*transb, 'N');

	IntType nrowa = nota ? *n : *k;
	IntType nrowb = notb ? *k : *n;

	IntType upper = option_matches(*uplo, 'U');

	//regardless of IntType, info needs to match xerbla's as it is passed by pointer
	pl_linalg_int_t info = 0;
	if (! upper && ! option_matches(*uplo, 'L')) {
		info = 1;
	}
	else if (! nota && ! option_matches(*transa, 'C', 'T')) {
		info = 2;
	}
	else if (! notb && ! option_matches(*transb, 'C', 'T')) {
		info = 3;
	}
	else if (*n < 0) {
		info = 4;
	}
	else if (*k < 0) {
		info = 5;
	}
	else if (*lda < max(1,nrowa)) {
		info = 8;
	}
	else if (*ldb < max(1,nrowb)) {
		info = 10;
	}
	else if (*ldc < max(1,*n)) {
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
void gemmt(const char *uplo, const char *transa, const char *transb,
           const IntType *n, const IntType *k,
           const promote_t<AType, BType, CType> *alpha,
           const AType *a, const IntType *lda,
           const BType *b, const IntType *ldb,
           const promote_t<AType, BType, CType> *beta,
                 CType *c, const IntType *ldc,
           std::string_view routine_name) {

	if constexpr(ParamCheck) {
		const bool res = gemmt_param_check(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
			routine_name);

		if(! res) return;
	}

	const     kernel_inttype a_strd        = *n;
	const     kernel_inttype b_strd        = *n;
	const     kernel_inttype cntg          = *k;

	const     auto           a_ptr         = a;
	const     auto           a_trans       = c_to_trans(*transa);
	const     auto           a_is_trans    = is_trans( a_trans );

	const     kernel_inttype a_cntg_stride = a_is_trans ? 1 : *lda;
	const     kernel_inttype a_strd_stride = a_is_trans ? *lda : 1;

	const     auto           b_trans       = c_to_trans(*transb);
	const     auto           b_is_trans    = is_trans( b_trans );

	const     auto           b_ptr         = b;
	const     kernel_inttype b_cntg_stride = b_is_trans ? *ldb : 1;
	const     kernel_inttype b_strd_stride = b_is_trans ? 1 : *ldb;

	const     auto           c_ptr         = c;
	const     auto           c_uplo        = c_to_uplo(*uplo);
	constexpr kernel_inttype c_cntg_stride = 1;
	const     kernel_inttype c_strd_stride = *ldc;

	spec::problem_context pctx {
		matmul::matmul3 {
			general_matrix    {                       matrix_base { a_ptr, cntg,   a_strd, a_cntg_stride, a_strd_stride }, is_conj(a_trans) },
			general_matrix    {                       matrix_base { b_ptr, cntg,   b_strd, b_cntg_stride, b_strd_stride }, is_conj(b_trans) },
			triangular_matrix { c_uplo, PERFLIBS_NOUNIT, matrix_base { c_ptr, a_strd, b_strd, c_cntg_stride, c_strd_stride },                  },
			*alpha, *beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gemmt(const char *uplo, const char *transa, const char *transb,
           const IntType *n, const IntType *k,
           const promote_t<AType, BType, CType> *alpha,
           const AType *a, const IntType *lda,
           const BType *b, const IntType *ldb,
           const promote_t<AType, BType, CType> *beta,
                 CType *c, const IntType *ldc) {

	gemmt<ParamCheck, IntType, AType, BType, CType, ArchitectureSpec>(
		uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
		gemmt_name<AType, BType, CType>);
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename CType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void gemmtr(const char *uplo, const char *transa, const char *transb,
            const IntType *n, const IntType *k,
            const promote_t<AType, BType, CType> *alpha,
            const AType *a, const IntType *lda,
            const BType *b, const IntType *ldb,
            const promote_t<AType, BType, CType> *beta,
                  CType *c, const IntType *ldc) {

	gemmt<ParamCheck, IntType, AType, BType, CType, ArchitectureSpec>(
		uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
		gemmtr_name<AType, BType, CType>);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_GEMMT_HPP
