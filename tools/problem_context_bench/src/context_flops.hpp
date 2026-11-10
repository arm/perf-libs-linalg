/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_CONTEXT_FLOPS_HPP
#define PERFLIBS_PCTX_CONTEXT_FLOPS_HPP

/*
 * Operation counts used by bench() to report throughput.
 * To support a new context, add a get_dnflops() overload here before the
 * benchmark template is defined. Read dimensions from the context's matrix
 * views; their cntg/strd axes can differ from the public routine's M/N/K axes.
 */

#include "blas_nflop.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_util.hpp"
#include "pl_linalg_int.h"
#include "framework/linalg_util.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include <cmath>

namespace perflibs::linalg {

template<typename MatrixType>
auto get_dnflops(const factorization::lu_factorization<MatrixType, pl_linalg_int_t>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.a.cntg();
	const kernel_inttype n = pctx.a.strd();

	const kernel_inttype op_tol_mod = is_complex_v<compute_precision> ? 4_ki : 1_ki;
	double lu_flops;
	if (n < m)
		lu_flops = (m * n * n) - ((n * n * n) / 3.0);
	else if (n == m)
		lu_flops = ((2.0 / 3.0) * n * n * n);
	else
		lu_flops = (m * m) * (n - m / 3.0);

	return (kernel_inttype) lu_flops * op_tol_mod;
}

template<typename MatrixType>
auto get_dnflops(const factorization::cholesky_factorization<MatrixType, pl_linalg_int_t>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype n = pctx.a.strd();

	const kernel_inttype op_tol_mod = is_complex_v<compute_precision> ? 4_ki : 1_ki;
	return (kernel_inttype) n * n * n * op_tol_mod / 3.0;
}

template<typename MatrixType>
auto get_dnflops(const factorization::qr_factorization<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.a.cntg();
	const kernel_inttype n = pctx.a.strd();

	const kernel_inttype op_tol_mod = is_complex_v<compute_precision> ? 4_ki : 1_ki;
	const kernel_inttype min_mn = perflibs::min(m, n);

	return (min_mn / 6.0) * (8.0 * std::pow(min_mn, 2) - 12.0 * m * min_mn - 12.0 * n * min_mn -
	                         9.0 * min_mn + 24.0 * m * n + 6.0 * m + 12 * n + 25);
}

template<typename MatrixType>
auto get_dnflops(const factorization::bidiagonalization<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.a.cntg();
	const kernel_inttype n = pctx.a.strd();

	const kernel_inttype op_tol_mod = is_complex_v<compute_precision> ? 4_ki : 1_ki;

	double fmuls, fadds;

	if (m >= n) {
		// For m >= n case (tall matrix)
		fmuls = 2.0 * m * n * n - 2.0 / 3.0 * n * n * n + 2.0 * n * n + 20.0 / 3.0 * n;
		fadds = 2.0 * m * n * n - 2.0 / 3.0 * n * n * n + n * n - m * n + 5.0 / 3.0 * n;
	}
	else {
		// For m < n case (wide matrix)
		fmuls = 2.0 * n * m * m - 2.0 / 3.0 * m * m * m + 2.0 * m * m + 20.0 / 3.0 * m;
		fadds = 2.0 * n * m * m - 2.0 / 3.0 * m * m * m + m * m - n * m + 5.0 / 3.0 * m;
	}

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	}
	else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops * op_tol_mod;
}

template<typename MatrixType>
auto get_dnflops(const factorization::bidiagonalization_block<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m  = pctx.a.cntg();
	const kernel_inttype n  = pctx.a.strd();
	const kernel_inttype nb = pctx.x.strd();

	const kernel_inttype op_tol_mod = is_complex_v<compute_precision> ? 4_ki : 1_ki;

	// For block bidiagonalization, we only factor the first nb columns/rows
	// We need to calculate the flops for a partial bidiagonalization

	double fmuls, fadds;

	if (m >= n) {
		// For m >= n case (tall matrix), but only processing nb columns
		fmuls = 2.0 * m * nb * nb - 2.0 / 3.0 * nb * nb * nb + 2.0 * nb * nb + 20.0 / 3.0 * nb;
		fadds = 2.0 * m * nb * nb - 2.0 / 3.0 * nb * nb * nb + nb * nb - m * nb + 5.0 / 3.0 * nb;
	}
	else {
		// For m < n case (wide matrix), but only processing nb rows
		fmuls = 2.0 * n * nb * nb - 2.0 / 3.0 * nb * nb * nb + 2.0 * nb * nb + 20.0 / 3.0 * nb;
		fadds = 2.0 * n * nb * nb - 2.0 / 3.0 * nb * nb * nb + nb * nb - n * nb + 5.0 / 3.0 * nb;
	}

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	}
	else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops * op_tol_mod;
}

template<typename MatrixType>
auto get_dnflops(const factorization::tridiagonalization<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype n = pctx.a.strd();

	const double fmuls = (2.0 / 3.0) * n * n * n + 2.5 * n * n - (1.0 / 6.0) * n;
	const double fadds = (2.0 / 3.0) * n * n * n + n * n - (8.0 / 3.0) * n;

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	} else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops;
}

template<typename MatrixType>
auto get_dnflops(const factorization::apply_q_from_qr<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.c.cntg();
	const kernel_inttype n = pctx.c.strd();
	const kernel_inttype k = pctx.tau.cntg();

	double fmuls, fadds;

	if (pctx.side == PERFLIBS_LEFT) {
		fmuls = 2. * n * m * k - n * k * k + 2. * n * k;
		fadds = 2. * n * m * k - n * k * k + n * k;
	}
	else {
		fmuls = 2. * n * m * k - m * k * k + m * k + n * k - 0.5 * k * k + 0.5 * k;
		fadds = 2. * n * m * k - m * k * k + m * k;
	}

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	}
	else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops;
}

template<typename MatrixType>
auto get_dnflops(const factorization::generate_q_from_qr<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.a.cntg();
	const kernel_inttype n = pctx.a.strd();
	const kernel_inttype k = pctx.tau.cntg();

	const double fmuls = 2. * m * n * k - (m + n) * k * k + 2. / 3. * k * k * k + 2. * n * k - k * k - 5. / 3. * k;
	const double fadds = 2. * m * n * k - (m + n) * k * k + 2. / 3. * k * k * k + n * k - m * k + 1. / 3. * k;

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	}
	else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops;
}

template<typename MatrixType>
auto get_dnflops(const factorization::apply_q_from_lq<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.c.cntg();
	const kernel_inttype n = pctx.c.strd();
	const kernel_inttype k = pctx.tau.cntg();

	double fmuls, fadds;

	if (pctx.side == PERFLIBS_LEFT) {
		fmuls = 2. * n * m * k - n * k * k + 2. * n * k;
		fadds = 2. * n * m * k - n * k * k + n * k;
	}
	else {
		fmuls = 2. * n * m * k - m * k * k + m * k + n * k - 0.5 * k * k + 0.5 * k;
		fadds = 2. * n * m * k - m * k * k + m * k;
	}

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	}
	else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops;
}

template<typename MatrixType>
auto get_dnflops(const factorization::generate_q_from_lq<MatrixType>& pctx) {
	using compute_precision = typename MatrixType::value_type;

	const kernel_inttype m = pctx.a.cntg();
	const kernel_inttype n = pctx.a.strd();
	const kernel_inttype k = pctx.tau.cntg();

	const double fmuls = 2. * m * n * k - (m + n) * k * k + 2. / 3. * k * k * k + m * k + n * k - k * k - 2. / 3. * k;
	const double fadds = 2. * m * n * k - (m + n) * k * k + 2. / 3. * k * k * k + m * k - n * k + 1. / 3. * k;

	double total_flops;
	if constexpr (is_complex_v<compute_precision>) {
		total_flops = 6.0 * fmuls + 2.0 * fadds;
	}
	else {
		total_flops = fmuls + fadds;
	}

	return (kernel_inttype) total_flops;
}


template<typename AMatrixType, typename BMatrixType, typename ScalarType>
auto get_dnflops(const matmul::matmul2<AMatrixType, BMatrixType, ScalarType>& pctx) {
	using compute_precision = promote_t<
		typename AMatrixType::value_type,
		typename BMatrixType::value_type>;

	return get_nflops<compute_precision>(
		pctx.a.strd(), pctx.b.strd(), pctx.a.cntg(),
		pctx.alpha != one<ScalarType>,
		pctx.beta  != one<ScalarType>);
}

template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType>
auto get_dnflops(const matmul::matmul3<AMatrixType, BMatrixType, CMatrixType, ScalarType>& pctx) {
	using compute_precision = promote_t<
		typename AMatrixType::value_type,
		typename BMatrixType::value_type,
		typename CMatrixType::value_type>;

	return get_nflops<compute_precision>(
		pctx.a.strd(), pctx.b.strd(), pctx.a.cntg(),
		pctx.alpha != one<ScalarType>,
		pctx.beta  != one<ScalarType>);
}

template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType>
auto get_dnflops(const matmul::matadd3<AMatrixType, BMatrixType, CMatrixType, ScalarType>& pctx) {
	using compute_precision = promote_t<
		typename AMatrixType::value_type,
		typename BMatrixType::value_type,
		typename CMatrixType::value_type>;

	constexpr kernel_inttype multiply_flops = is_complex_v<compute_precision> ? 6_ki : 1_ki;
	constexpr kernel_inttype add_flops = is_complex_v<compute_precision> ? 2_ki : 1_ki;
	return pctx.c.cntg() * pctx.c.strd() * (2_ki * multiply_flops + add_flops);
}

template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType>
auto get_dnflops(const matmul::rank_update_2k<AMatrixType, BMatrixType, CMatrixType, ScalarType>& pctx) {
	using compute_precision = promote_t<
		typename AMatrixType::value_type,
		typename BMatrixType::value_type,
		typename CMatrixType::value_type>;

	return get_nflops<compute_precision>(pctx.a.strd(), pctx.b.strd(), pctx.a.cntg(), true,  true)
	     + get_nflops<compute_precision>(pctx.a.strd(), pctx.b.strd(), pctx.a.cntg(), false, false);
}

template<typename AMatrixType, typename BMatrixType, typename ScalarType>
auto get_dnflops(const solve::solve<AMatrixType, BMatrixType, ScalarType>& pctx) {
	using compute_precision = promote_t<
		typename AMatrixType::value_type,
		typename BMatrixType::value_type>;

	const kernel_inttype op_mod_mls = perflibs::is_complex_v<compute_precision> ?  8_ki : 2_ki;
	const kernel_inttype op_mod_mul = perflibs::is_complex_v<compute_precision> ?  6_ki : 1_ki;
	const kernel_inttype op_mod_div = perflibs::is_complex_v<compute_precision> ? 11_ki : 1_ki;

	const bool is_unit      = pctx.a.is_unit();
	const bool alpha_is_one = pctx.alpha == one<>;

	const kernel_inttype k  = pctx.b.cntg();
	const kernel_inttype nn = pctx.b.strd();

	kernel_inttype nflops = (kernel_inttype)(k*(k-1)/2)*op_mod_mls;
	nflops += is_unit      ? 0_ki : (kernel_inttype)k*op_mod_div;
	nflops += alpha_is_one ? 0_ki : (kernel_inttype)k*op_mod_mul;
	nflops *= nn;

	return nflops;
}

} // namespace perflibs::linalg

#endif
