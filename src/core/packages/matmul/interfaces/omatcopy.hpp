/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_OMATCOPY_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_OMATCOPY_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType, typename BType>
inline constexpr std::string_view omatcopy_name = "?OMATCOPY ";

template<> inline constexpr std::string_view omatcopy_name<r32, r32> = "SOMATCOPY ";
template<> inline constexpr std::string_view omatcopy_name<r64, r64> = "DOMATCOPY ";
template<> inline constexpr std::string_view omatcopy_name<c32, c32> = "COMATCOPY ";
template<> inline constexpr std::string_view omatcopy_name<c64, c64> = "ZOMATCOPY ";

template<typename IntType, typename AType, typename BType>
PERFLIBS_LINALG_INLINE
bool omatcopy_param_check(
	char order, char trans,
	IntType m, IntType n,
	promote_t<AType, BType> alpha,
	const AType *a, IntType lda,
	      BType *b, IntType ldb,
	std::string_view name) {

	const bool corder = option_matches(order, 'C');
	const bool rorder = option_matches(order, 'R');
	const bool ttrans = option_matches(trans, 'T');
	const bool ntrans = option_matches(trans, 'N');
	const bool ctrans = option_matches(trans, 'C');
	const bool rtrans = option_matches(trans, 'R');

	const bool is_trans = ttrans || ctrans;

	IntType info = 0;

	if (!corder && !rorder) {
		info = 1;
	}
	else if (!ntrans && !ttrans && !ctrans && !rtrans) {
		info = 2;
	}
	else if (m < 0) {
		info = 3;
	}
	else if (n < 0) {
		info = 4;
	}
	// To align with MKL, we will no longer return
	// error if the leading dimension is zero while
	// m = n = 0
	if ((rorder && lda < n) || (corder && lda < m)) {
		info = 7;
	}
	else if ((rorder && ldb < (is_trans ? m : n)) || (corder && ldb < (is_trans ? n : m))) {
		info = 9;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}

	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename BType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void omatcopy(
	char order, char transa,
	IntType m, IntType n,
	promote_t<AType, BType> alpha,
	const AType *a, IntType lda,
	      BType *b, IntType ldb) {

	using scalar_type = promote_t<AType, BType>;

	if constexpr (ParamCheck) {
		const bool res = omatcopy_param_check(
			order, transa, m, n, alpha, a, lda, b, ldb,
			omatcopy_name<AType, BType>);

		if (!res) return;
	}

	const     auto           atrans        = c_to_trans(transa);
	const     bool           is_colmajor   = option_matches(order, 'C');
	const     bool           is_trans_a    = is_trans(atrans);
	const     bool           c_is_conj     = is_conj(atrans);

	const     kernel_inttype a_strd        = m;
	const     kernel_inttype b_strd        = n;
	constexpr kernel_inttype cntg          = 0;

	constexpr kernel_inttype a_cntg_stride = 0;
	constexpr kernel_inttype a_strd_stride = 0;
	constexpr kernel_inttype b_cntg_stride = 0;
	constexpr kernel_inttype b_strd_stride = 0;

	const     kernel_inttype c_cntg_stride = is_colmajor ? 1 : lda;
	const     kernel_inttype c_strd_stride = is_colmajor ? lda : 1;
	const     kernel_inttype d_cntg_stride = (is_colmajor ^ is_trans_a) ? 1 : ldb;
	const     kernel_inttype d_strd_stride = (is_colmajor ^ is_trans_a) ? ldb : 1;

	spec::problem_context pctx{
		matmul::matmul4{
			general_matrix { matrix_base { &zero<scalar_type>, cntg,   a_strd, a_cntg_stride, a_strd_stride }            },
			general_matrix { matrix_base { &zero<scalar_type>, cntg,   b_strd, b_cntg_stride, b_strd_stride }            },
			general_matrix { matrix_base { a,                  a_strd, b_strd, c_cntg_stride, c_strd_stride }, c_is_conj },
			general_matrix { matrix_base { b,                  a_strd, b_strd, d_cntg_stride, d_strd_stride }            },
			zero<scalar_type>, alpha
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_OMATCOPY_HPP
