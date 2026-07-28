/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_IMATCOPY_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_IMATCOPY_HPP

#include "packages/matmul/strategies.hpp"
#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <string_view>

namespace perflibs::linalg {

template<typename AType>
inline constexpr std::string_view imatcopy_name = "?IMATCOPY ";

template<> inline constexpr std::string_view imatcopy_name<r32> = "SIMATCOPY ";
template<> inline constexpr std::string_view imatcopy_name<r64> = "DIMATCOPY ";
template<> inline constexpr std::string_view imatcopy_name<c32> = "CIMATCOPY ";
template<> inline constexpr std::string_view imatcopy_name<c64> = "ZIMATCOPY ";

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool imatcopy_param_check(
	char order, char trans,
	IntType m, IntType n,
	AType alpha,
	AType *ab, IntType lda, IntType ldb,
	std::string_view name) {

	const bool corder   = option_matches(order, 'C');
	const bool rorder   = option_matches(order, 'R');
	const bool ttrans   = option_matches(trans, 'T');
	const bool ntrans   = option_matches(trans, 'N');
	const bool ctrans   = option_matches(trans, 'C');
	const bool rtrans   = option_matches(trans, 'R');
	const bool is_trans = ttrans || ctrans;

	IntType info = 0;

	if (! corder && ! rorder) {
		info = 1;
	}
	else if (! ntrans && ! ttrans  && ! ctrans && ! rtrans) {
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
	if ((corder && lda < m) || (rorder && lda < n)) {
		info = 7;
	}
	else if ((corder && ldb < (is_trans ? n : m)) || (rorder && ldb < (is_trans ? m : n))) {
		info = 8;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}

	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void imatcopy(
	char order, char transa,
	IntType m,  IntType n,
	AType alpha,
	AType *ab, IntType lda, IntType ldb) {

	if constexpr(ParamCheck) {
		const bool res = imatcopy_param_check(
			order, transa, m, n, alpha, ab, lda, ldb,
				imatcopy_name<AType>);

		if(! res) return;
	}
	// TODO: Add column alignment for buffer
	const bool is_colmajor = option_matches(order, 'C');

	const     auto           atrans        = c_to_trans(transa);
	const     bool           is_trans_a    = is_trans(atrans);
	          bool           c_is_conj     = is_conj(atrans);

	const     kernel_inttype a_strd        = m;
	const     kernel_inttype b_strd        = n;
	constexpr kernel_inttype cntg          = 0;

	          auto           buffer        = get_memory<AType>(a_strd * b_strd);

	constexpr kernel_inttype a_cntg_stride = 0;
	constexpr kernel_inttype a_strd_stride = 0;
	constexpr kernel_inttype b_cntg_stride = 0;
	constexpr kernel_inttype b_strd_stride = 0;

	const     kernel_inttype c_cntg_stride = is_colmajor ? 1 : lda;
	const     kernel_inttype c_strd_stride = is_colmajor ? lda : 1;

	const     kernel_inttype d_cntg_stride = (is_colmajor ^ is_trans_a) ? 1 : ldb;
	const     kernel_inttype d_strd_stride = (is_colmajor ^ is_trans_a) ? ldb : 1;

	spec::problem_context pctx_in {
		matmul::matmul4 {
			general_matrix { matrix_base { &zero<AType>,                  cntg,   a_strd, a_cntg_stride, a_strd_stride }            },
			general_matrix { matrix_base { &zero<AType>,                  cntg,   b_strd, b_cntg_stride, b_strd_stride }            },
			general_matrix { matrix_base { static_cast<const AType*>(ab), a_strd, b_strd, c_cntg_stride, c_strd_stride }, c_is_conj },
			general_matrix { matrix_base { buffer,                        a_strd, b_strd, 1,             a_strd        }            },
			zero<AType>, alpha
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx_in);

	spec::problem_context pctx_out {
		matmul::matmul4 {
			general_matrix { matrix_base { &zero<AType>,                      cntg,   a_strd, a_cntg_stride, a_strd_stride } },
			general_matrix { matrix_base { &zero<AType>,                      cntg,   b_strd, b_cntg_stride, b_strd_stride } },
			general_matrix { matrix_base { static_cast<const AType*>(buffer), a_strd, b_strd, 1,             a_strd        } },
			general_matrix { matrix_base { ab,                                a_strd, b_strd, d_cntg_stride, d_strd_stride } },
			zero<AType>, one<AType>
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};

	matmul::compute(pctx_out);

	return_memory<AType>(buffer);
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_IMATCOPY_HPP
