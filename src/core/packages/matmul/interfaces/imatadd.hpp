/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_IMATADD_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_IMATADD_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType>
inline constexpr std::string_view imatadd_name = "?IMATADD ";

template<> inline constexpr std::string_view imatadd_name<r32, r32> = "SIMATADD ";
template<> inline constexpr std::string_view imatadd_name<r64, r64> = "DIMATADD ";
template<> inline constexpr std::string_view imatadd_name<c32, c32> = "CIMATADD ";
template<> inline constexpr std::string_view imatadd_name<c64, c64> = "ZIMATADD ";

template<typename IntType, typename AMatrixType, typename BMatrixType>
PERFLIBS_LINALG_INLINE
bool imatadd_param_check(
	char order, char transa, char transb,
	IntType m, IntType n,
	promote_t<AMatrixType, BMatrixType> alpha,
	      AMatrixType *a, IntType lda,
	promote_t<AMatrixType, BMatrixType> beta,
	const BMatrixType *b, IntType ldb,
	std::string_view name) {

	const bool corder     = option_matches(order, 'C');
	const bool rorder     = option_matches(order, 'R');
	const bool read_a     = alpha != zero<>;
	const bool validate_b = beta  != zero<>;

	const bool ntransa    = option_matches(transa, 'N');
	const bool ttransa    = option_matches(transa, 'T');
	const bool ctransa    = option_matches(transa, 'C');
	const bool rtransa    = option_matches(transa, 'R');

	const bool ntransb    = option_matches(transb, 'N');
	const bool ttransb    = option_matches(transb, 'T');
	const bool ctransb    = option_matches(transb, 'C');
	const bool rtransb    = option_matches(transb, 'R');

	const bool is_trans_a = ttransa || ctransa;
	const bool is_trans_b = ttransb || ctransb;

	IntType info = 0;

	if (!corder && !rorder) {
		info = 1;
	}
	else if (!ntransa && !ttransa && !ctransa && !rtransa) {
		info = 2;
	}
	else if (!ntransb && !ttransb && !ctransb && !rtransb) {
		info = 3;
	}
	else if (m < 0) {
		info = 4;
	}
	else if (n < 0) {
		info = 5;
	}
	else if ((corder && lda < m) || (rorder && lda < n)) {
		info = 8;
	}
	else if (read_a && ((corder && lda < (is_trans_a ? n : m)) || (rorder && lda < (is_trans_a ? m : n)))) {
		info = 8;
	}
	else if (validate_b && ((corder && ldb < (is_trans_b ? n : m)) || (rorder && ldb < (is_trans_b ? m : n)))) {
		info = 11;
	}

	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}

	return true;
}

template<bool ParamCheck, typename IntType, typename AMatrixType, typename BMatrixType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void imatadd(
	char order, char transa, char transb,
	IntType m, IntType n,
	promote_t<AMatrixType, BMatrixType> alpha,
	      AMatrixType *a, IntType lda,
	promote_t<AMatrixType, BMatrixType> beta,
	const BMatrixType *b, IntType ldb) {

	if constexpr (ParamCheck) {
		const bool res = imatadd_param_check(
			order, transa, transb, m, n, alpha, a, lda, beta, b, ldb,
			imatadd_name<AMatrixType, BMatrixType>);

		if (!res) return;
	}

	if (m == 0 || n == 0) {
		return;
	}

	const auto           atrans        = c_to_trans         (transa);
	const auto           btrans        = c_to_trans         (transb);
	const bool           is_colmajor   = option_matches (order, 'C');
	const bool           is_trans_a    = is_trans           (atrans);
	const bool           is_trans_b    = is_trans           (btrans);
	const bool           a_is_conj     = is_conj            (atrans);
	const bool           b_is_conj     = is_conj            (btrans);
	const kernel_inttype a_strd        = m;
	const kernel_inttype b_strd        = n;

	const kernel_inttype a_cntg_stride = (is_colmajor ^ is_trans_a) ? 1 : lda;
	const kernel_inttype a_strd_stride = (is_colmajor ^ is_trans_a) ? lda : 1;
	const kernel_inttype b_cntg_stride = (is_colmajor ^ is_trans_b) ? 1 : ldb;
	const kernel_inttype b_strd_stride = (is_colmajor ^ is_trans_b) ? ldb : 1;
	const kernel_inttype c_cntg_stride =  is_colmajor               ? 1 : lda;
	const kernel_inttype c_strd_stride =  is_colmajor               ? lda : 1;

	spec::problem_context pctx {
		matmul::matadd2 {
			general_matrix { matrix_base { static_cast<const AMatrixType*>(a), a_strd, b_strd, a_cntg_stride, a_strd_stride }, a_is_conj },
			general_matrix { matrix_base { b, a_strd, b_strd, b_cntg_stride, b_strd_stride }, b_is_conj },
			general_matrix { matrix_base { a, a_strd, b_strd, c_cntg_stride, c_strd_stride } },
			alpha,
			beta
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_BLAS_INTERFACES_IMATADD_HPP
