/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARF_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARF_HPP

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
bool larf_param_check(
	const char* side,
	const IntType* m, const IntType* n,
	const AType* v, const IntType* incv,
	const AType* tau,
	const AType* c, const IntType* ldc,
	const AType* work,
	IntType* info) {

	*info = 0;
	const bool lside = option_matches(*side, 'L');
	const bool rside = option_matches(*side, 'R');

	if (!lside && !rside) {
		*info = -1;
	}
	else if (*m < 0) {
		*info = -2;
	}
	else if (*n < 0) {
		*info = -3;
	}
	else if (*incv == 0) {
		*info = -5;
	}
	else if (*ldc < max(1, *m)) {
		*info = -8;
	}

	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;

		if constexpr (std::is_same_v<AType, double>) {
			fname = "DLARF  ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SLARF  ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CLARF  ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZLARF  ";
		}

		call_xerbla(fname, ninfo);
		return false;
	}

	return true;
}

template<bool Paramcheck, typename IntType, typename Atype, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void larf(const char* side, const IntType* m, const IntType* n,
          const Atype* v, const IntType* incv,
          const Atype* tau, Atype* c, const IntType* ldc, Atype* work) {

	if constexpr (Paramcheck) {
		IntType info = 0;
		const bool res = larf_param_check(side, m, n, v, incv, tau, c, ldc, work, &info);
		if (!res) return;
	}

	// Quick return if possible
	if (*m <= 0 || *n <= 0 || *tau == zero<Atype>) {
		return;
	}

	const auto aside    = c_to_side(*side);
	const bool is_lside = is_left(aside);

	const kernel_inttype c_strd        = *n;
	const kernel_inttype c_cntg        = *m;
	const kernel_inttype c_cntg_stride = 1;
	const kernel_inttype c_strd_stride = *ldc;

	const kernel_inttype v_cntg        = is_lside ? *m : *n;
	const kernel_inttype v_strd        = 1;
	const kernel_inttype v_cntg_stride = *incv;
	const kernel_inttype v_strd_stride = 1;

	const kernel_inttype work_cntg        = is_lside ? *n : *m;
	const kernel_inttype work_strd        = 1;
	const kernel_inttype work_cntg_stride = 1;
	const kernel_inttype work_strd_stride = 1;

	auto pctx = spec::problem_context{
		factorization::apply_elementary_reflector{
			aside,
			general_matrix{ matrix_base{ v, v_cntg, v_strd, v_cntg_stride, v_strd_stride } },
			general_matrix{ matrix_base{ c, c_cntg, c_strd, c_cntg_stride, c_strd_stride } },
			general_matrix{ matrix_base{ work, work_cntg, work_strd, work_cntg_stride, work_strd_stride } },
			*tau
		},
		ArchitectureSpec{ machine::get_system_unsafe() }
	};

	factorization::compute(pctx);
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARF_HPP
