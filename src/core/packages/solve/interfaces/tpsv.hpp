/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BLAS_INTERFACES_TPSV_HPP
#define PERFLIBS_LINALG_BLAS_INTERFACES_TPSV_HPP

#include "packages/solve/strategies.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include "matrix/matrix.hpp"
#include "framework/xerbla.hpp"

namespace perflibs::linalg {

template<typename AType, typename XType>
inline constexpr std::string_view tpsv_name = "?TPSV ";

template<> inline constexpr std::string_view tpsv_name<r32, r32> = "STPSV ";
template<> inline constexpr std::string_view tpsv_name<r64, r64> = "DTPSV ";
template<> inline constexpr std::string_view tpsv_name<c32, c32> = "CTPSV ";
template<> inline constexpr std::string_view tpsv_name<c64, c64> = "ZTPSV ";

template<typename IntType, typename AType, typename XType>
PERFLIBS_LINALG_INLINE
bool tpsv_param_check(const char *uplo, const char *trans, const char *diag,
	const IntType *n,
	const AType *a,
	      XType *x, const IntType *incx,
	std::string_view name) {

	const bool is_upper     = option_matches(*uplo, 'U');
	const bool is_lower     = option_matches(*uplo, 'L');
	const bool is_notrans   = option_matches(*trans, 'N');
	const bool is_trans     = option_matches(*trans, 'T');
	const bool is_conjtrans = option_matches(*trans, 'C');
	const bool is_unit      = option_matches(*diag, 'U');
	const bool is_nounit    = option_matches(*diag, 'N');

	pl_linalg_int_t info = 0;
	if (! is_upper && ! is_lower) {
		info = 1;
	}
	else if (! is_notrans  && ! is_trans && ! is_conjtrans) {
		info = 2;
	}
	else if (! is_unit  && ! is_nounit) {
		info = 3;
	}
	else if (*n < 0) {
		info = 4;
	}
	else if (*incx == 0) {
		info = 7;
	}
	if (info != 0) {
		call_xerbla(name, info);
		return false;
	}
	return true;
}

template<bool ParamCheck, typename IntType, typename AType, typename XType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void tpsv(
	const char *uplo, const char *trans, const char *diag,
	const IntType *n,
	const AType *a,
	      XType *x, const IntType *incx) {

	using scalar_type = promote_t<XType, AType>;

	if constexpr(ParamCheck) {
		const auto res = perflibs::linalg::tpsv_param_check(
			uplo, trans, diag, n, a, x, incx,
				tpsv_name<AType, XType>);

		if(! res) return;
	}

	if (*n == 0)
		return;

	const     auto           aside       = PERFLIBS_LEFT;
	const     auto           atransa     = c_to_trans(*trans);
	const     bool           is_trans_a  = is_trans(atransa);
	const     auto           auplo       = c_to_uplo(*uplo);
	const     auto           auplo_mod   = is_trans_a ? lower_flip(auplo) : auplo;
	const     auto           adiag       = c_to_diag(*diag);

	const     kernel_inttype a_cntg      = *n;
	const     kernel_inttype c_cntg      = *n;
	constexpr kernel_inttype c_strd      = 1;

	const     kernel_inttype b_cntg_step = *incx;
	constexpr kernel_inttype b_strd_step = 1;

	if (*incx < 0) x += (*incx * -1) * (*n - 1);

	spec::problem_context pctx {
		solve::solve {
			aside, atransa,
			triangular_matrix { auplo_mod, adiag, packed_matrix_base { a, c_cntg, a_cntg, auplo_mod, false, is_trans_a }, is_conj(atransa) },
			general_matrix    {                   matrix_base        { x, a_cntg, c_strd, b_cntg_step, b_strd_step     }                   },
			one<scalar_type>
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	solve::compute(pctx);
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_BLAS_INTERFACES_TPSV_HPP
