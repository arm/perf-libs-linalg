/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GETRFNP_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GETRFNP_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "spec/strategy_tag.hpp"
#include "matrix/matrix.hpp"
#include "matrix/adaptors.hpp"
#include "framework/compute.hpp"

#include "framework/linalg_util.hpp"
#include "framework/xerbla.hpp"

#include <span>

namespace perflibs::linalg {

// LU without pivoting (?getrfnp) and of the first nfact columns only, leaving the Schur
// complement in the trailing block (?getrfnpi); also implemented in oneMKL and AOCL.
// At the first exactly zero pivot k the factorization stops with info = k.

template<typename AType>
PERFLIBS_LINALG_INLINE
constexpr std::string_view getrfnp_name(bool incomplete) {
	if constexpr (std::is_same_v<AType, double>) {
		return incomplete ? "DGETRFNPI " : "DGETRFNP ";
	}
	else if constexpr (std::is_same_v<AType, float>) {
		return incomplete ? "SGETRFNPI " : "SGETRFNP ";
	}
	else if constexpr (std::is_same_v<AType, std::complex<float>>) {
		return incomplete ? "CGETRFNPI " : "CGETRFNP ";
	}
	else {
		return incomplete ? "ZGETRFNPI " : "ZGETRFNP ";
	}
}

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool getrfnp_param_check(const IntType *m, const IntType *n, AType *a,
                         const IntType *lda, IntType *info) {

	*info = 0;

	if (*m < 0) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*lda < max(1, *m)) {
		*info = -4;
	}
	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		call_xerbla(getrfnp_name<AType>(false), ninfo);
		return false;
	}
	return true;
}

template<typename IntType, typename AType>
PERFLIBS_LINALG_INLINE
bool getrfnpi_param_check(const IntType *m, const IntType *n, const IntType *nfact,
                          AType *a, const IntType *lda, IntType *info) {

	*info = 0;

	if (*m < 0) {
		*info = -1;
	}
	else if (*n < 0) {
		*info = -2;
	}
	else if (*nfact < 0 || *nfact > min(*m, *n)) {
		*info = -3;
	}
	else if (*lda < max(1, *m)) {
		*info = -5;
	}
	if (*info != 0) {
		pl_linalg_int_t ninfo = -(*info);
		call_xerbla(getrfnp_name<AType>(true), ninfo);
		return false;
	}
	return true;
}

template<typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void getrfnpi_compute(const IntType m, const IntType n, const IntType nfact,
                      AType *a, const IntType lda, IntType *info) {
	*info = 0;
	if (m == 0 || n == 0 || nfact == 0) return;

	const kernel_inttype a_cntg      = m;
	const kernel_inttype a_strd      = n;
	const kernel_inttype a_cntg_step = 1;
	const kernel_inttype a_strd_step = lda;
	const kernel_inttype n_fact      = nfact;

	auto architecture_spec = ArchitectureSpec{ machine::get_system_unsafe() };
	auto a_full = general_matrix{ matrix_base{ a, a_cntg, a_strd, a_cntg_step, a_strd_step } };

	// 1. factor the first nfact columns
	auto a_panel = get_strd_panel(a_full, 0, n_fact);
	IntType iinfo = 0;
	auto factorization_pctx = spec::problem_context{
		factorization::lu_factorization_no_pivot{ a_panel, iinfo },
		architecture_spec
	};
	factorization::compute(factorization_pctx);

	// 2. apply the factored columns to the trailing columns (Schur complement)
	const kernel_inttype factored = iinfo > 0 ? kernel_inttype(iinfo - 1) : n_fact;
	if (a_strd > n_fact && factored > 0) {
		auto a_factored = get_strd_panel(a_full, 0, factored);
		auto a_trailing = get_strd_panel(a_full, n_fact, a_strd - n_fact);
		auto update_pctx = spec::problem_context{
			factorization::lu_panel_update{ a_factored, a_trailing, std::span<IntType>{} },
			architecture_spec
		};
		factorization::compute(update_pctx);
	}
	*info = iinfo;
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void getrfnpi(const IntType *m, const IntType *n, const IntType *nfact,
              AType *a, const IntType *lda, IntType *info) {
	if constexpr (Paramcheck) {
		const bool res = getrfnpi_param_check(m, n, nfact, a, lda, info);

		if (!res) return;
	}
	getrfnpi_compute<IntType, AType, ArchitectureSpec>(*m, *n, *nfact, a, *lda, info);
}

template<bool Paramcheck, typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void getrfnp(const IntType *m, const IntType *n,
             AType *a, const IntType *lda, IntType *info) {
	if constexpr (Paramcheck) {
		const bool res = getrfnp_param_check(m, n, a, lda, info);

		if (!res) return;
	}
	getrfnpi_compute<IntType, AType, ArchitectureSpec>(*m, *n, min(*m, *n), a, *lda, info);
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_GETRFNP_HPP
