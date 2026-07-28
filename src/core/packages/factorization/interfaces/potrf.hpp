/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_POTRF_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_POTRF_HPP

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
bool potrf_param_check(const char *uplo, const IntType *n, AType *a, const IntType *lda, IntType *info) {

	*info = 0;

	bool upper = option_matches(*uplo, 'U');
	bool lower = option_matches(*uplo, 'L');

	if (!upper && !lower) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*lda < max(1, *n)) {
		*info = -4;
	}
	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		std::string_view fname;

		if constexpr (std::is_same_v<AType, double>) {
			fname = "DPOTRF ";
		}
		else if constexpr (std::is_same_v<AType, float>) {
			fname = "SPOTRF ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<float>>) {
			fname = "CPOTRF ";
		}
		else if constexpr (std::is_same_v<AType, std::complex<double>>) {
			fname = "ZPOTRF ";
		}
		call_xerbla(fname, ninfo);
		return false;
	}
	return true;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void potrf(const char *uplo, const IntType *n, AType *a, const IntType *lda, IntType *info) {
	if constexpr (Paramcheck) {
		const bool res = potrf_param_check(uplo, n, a, lda, info);

		if (!res) return;
	}

	if (*n == 0) return;

	const kernel_inttype a_cntg      = *n;
	const kernel_inttype a_strd      = *n;
	const kernel_inttype a_cntg_step = 1;
	const kernel_inttype a_strd_step = *lda;
	const auto           a_uplo      = c_to_uplo(*uplo);

	if constexpr (perflibs::is_complex_v<AType>) {
		auto pctx = spec::problem_context{
			factorization::cholesky_factorization{
				hermitian_matrix {a_uplo, matrix_base{ a, a_cntg, a_strd, a_cntg_step, a_strd_step } },
				*info
			},
			ArchitectureSpec{ machine::get_system_unsafe() }
		};
		compute(pctx);
	}
	else {
		auto pctx = spec::problem_context{
			factorization::cholesky_factorization{
				symmetric_matrix {a_uplo, matrix_base{ a, a_cntg, a_strd, a_cntg_step, a_strd_step } },
				*info
			},
			ArchitectureSpec{ machine::get_system_unsafe() }
		};
		factorization::compute(pctx);
	}
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_POTRF_HPP
