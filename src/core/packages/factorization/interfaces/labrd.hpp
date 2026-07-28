/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LABRD_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LABRD_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "spec/strategy_tag.hpp"
#include "matrix/matrix.hpp"
#include "framework/compute.hpp"
#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool labrd_param_check(const IntType *m, const IntType *n, const IntType *nb, AType *a, const IntType *lda,
                       remove_complex_t<promote_t<AType>> *d, remove_complex_t<promote_t<AType>> *e,
                       AType *tauq, AType *taup, AType *x, const IntType *ldx, AType *y, const IntType *ldy,
                       IntType *info) {
	*info = 0;

	if (*m < 0) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*nb < 0 || *nb > min(*m, *n)) {
		*info = -3;
	}
	else if (*lda < max(1, *m)) {
		*info = -5;
	}
	else if (*ldx < max(1, *m)) {
		*info = -11;
	}
	else if (*ldy < max(1, *n)) {
		*info = -13;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;
		if constexpr (std::is_same_v<AType, double>) {
			fname = "DLABRD ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SLABRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CLABRD ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZLABRD ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void labrd(const IntType *m, const IntType *n, const IntType *nb, AType *a, const IntType *lda,
           remove_complex_t<promote_t<AType>> *d, remove_complex_t<promote_t<AType>> *e, AType *tauq,
           AType *taup, AType *x, const IntType *ldx, AType *y, const IntType *ldy) {

	if constexpr (Paramcheck) {
		IntType info = 0;
		if (!labrd_param_check(m, n, nb, a, lda, d, e, tauq, taup, x, ldx, y, ldy, &info)) {
			return;
		}
	}

	// Quick return if possible
	if (*m == 0 || *n == 0 || *nb == 0) return;

	// Matrix A dimensions
	const     kernel_inttype a_cntg      = *m;
	const     kernel_inttype a_strd      = *n;
	constexpr kernel_inttype a_cntg_step = 1;
	const     kernel_inttype a_strd_step = *lda;

	// Matrix X dimensions
	const     kernel_inttype x_cntg      = *m;
	const     kernel_inttype x_strd      = *nb;
	constexpr kernel_inttype x_cntg_step = 1;
	const     kernel_inttype x_strd_step = *ldx;

	// Matrix Y dimensions
	const     kernel_inttype y_cntg      = *n;
	const     kernel_inttype y_strd      = *nb;
	constexpr kernel_inttype y_cntg_step = 1;
	const     kernel_inttype y_strd_step = *ldy;

	// Shared vector dimensions
	constexpr kernel_inttype vec_strd    = 1;
	constexpr kernel_inttype vec_step    = 1;

	auto pctx = spec::problem_context{
		factorization::bidiagonalization_block{
			general_matrix { matrix_base { a,    a_cntg, a_strd,   a_cntg_step, a_strd_step } },
			general_matrix { matrix_base { d,    *nb,    vec_strd, vec_step,    vec_step    } },
			general_matrix { matrix_base { e,    *nb,    vec_strd, vec_step,    vec_step    } },
			general_matrix { matrix_base { tauq, *nb,    vec_strd, vec_step,    vec_step    } },
			general_matrix { matrix_base { taup, *nb,    vec_strd, vec_step,    vec_step    } },
			general_matrix { matrix_base { x,    x_cntg, x_strd,   x_cntg_step, x_strd_step } },
			general_matrix { matrix_base { y,    y_cntg, y_strd,   y_cntg_step, y_strd_step } }
		},
		ArchitectureSpec { machine::get_system_unsafe() }
	};
	factorization::compute(pctx);
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LABRD_HPP
