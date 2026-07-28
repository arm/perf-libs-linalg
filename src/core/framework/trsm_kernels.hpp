/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TRSM_KERNELS_HPP
#define PERFLIBS_LINALG_TRSM_KERNELS_HPP

#include "perflibs_complex.hpp"
#include "perflibs_unused.hpp"
#include "perflibs_util.hpp"
#include "framework/trsm_kernel_helpers.hpp"
#include "packages/solve/strategies/solve_references.hpp"
#include <type_traits>

namespace perflibs::linalg {
namespace {

template<typename ScalarType, bool Forward, bool TransA, bool TransB, bool NoUnit, bool ConjA>
PERFLIBS_LINALG_INLINE
void trsm_reference_kernel(const int_type m, const int_type n,
                           const ScalarType alpha,
                           const ScalarType * restrict a, int_type lda,
                                 ScalarType * restrict b, int_type ldb) {
	constexpr char C = 'C';
	constexpr char L = 'L';
	constexpr char N = 'N';
	constexpr char R = 'R';
	constexpr char U = 'U';
	constexpr char T = 'T';

	// recover BLAS parameters from template parameters and call reference kernel
	if constexpr (TransB) {
		if constexpr (TransA) {
			if constexpr (Forward) {  // left, notrans, lower
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&L, &L, &N, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&L, &L, &N, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
			else {  // left, notrans, upper
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&L, &U, &N, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&L, &U, &N, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
		}
		else if constexpr (ConjA) {
			if constexpr (Forward) {  // left, conjtrans, upper
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&L, &U, &C, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&L, &U, &C, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
			else {  // left, conjtrans, lower
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&L, &L, &C, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&L, &L, &C, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
		}
		else {
			if constexpr (Forward) {  // left, trans, upper
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&L, &U, &T, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&L, &U, &T, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
			else {  // left, trans, lower
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&L, &L, &T, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&L, &L, &T, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
		}
	}
	else {
		if constexpr (ConjA) {
			if constexpr (Forward) {  // right, conjtrans, lower
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&R, &L, &C, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&R, &L, &C, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
			else {  // right, conjtrans, upper
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&R, &U, &C, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&R, &U, &C, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
		}
		else if constexpr (TransA) {
			if constexpr (Forward) {  // right, trans, lower
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&R, &L, &T, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&R, &L, &T, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
			else {  // right, trans, upper
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&R, &U, &T, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&R, &U, &T, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
		}
		else {
			if constexpr (Forward) {  // right, notrans, upper
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&R, &U, &N, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&R, &U, &N, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
			else {  // right, notrans, lower
				if constexpr (NoUnit) solve::reference::trsm<ScalarType>(&R, &L, &N, &N, &m, &n, &alpha, a, &lda, b, &ldb);
				else                  solve::reference::trsm<ScalarType>(&R, &L, &N, &U, &m, &n, &alpha, a, &lda, b, &ldb);
			}
		}
	}
}

// Performs a mx4 TRSM for right-sided or 4xn TRSM for left-sided problems,
// where m / n are divisible by 4.
template<typename ScalarType, typename VecType, bool Forward, bool TransA, bool TransB, bool NoUnit, bool ConjA>
PERFLIBS_LINALG_INLINE
void trsm_kernel_impl(const kernel_inttype m,
                      const ScalarType * restrict a, kernel_inttype lda,
                            ScalarType * restrict b, kernel_inttype ldb) {

	// Special case for float: we need to divide rather than multiply by
	// reciprocal otherwise too much accuracy is lost in certain cases
	// (e.g. certain LAPACK test cases).
	constexpr auto recip = !std::is_same_v<ScalarType, float>;

	// load a
	auto a0 = load<VecType>(a + 0*lda);
	auto a1 = load<VecType>(a + 1*lda);
	auto a2 = load<VecType>(a + 2*lda);
	auto a3 = load<VecType>(a + 3*lda);

	if constexpr (TransA) transpose(a0, a1, a2, a3);

	VecType a00, a11, a22, a33;
	if constexpr (NoUnit) {
		if constexpr (recip)
			recip_diagonal(a0, a1, a2, a3);
		else {
			// duplicate diagonal elements x4 to prepare for divides below
			a00 = dup_lane<0>(a0);
			a11 = dup_lane<1>(a1);
			a22 = dup_lane<2>(a2);
			a33 = dup_lane<3>(a3);
		}
	}

	for (unsigned k = 0; k < (m >> 2); ++k) {
		// load b
		auto b0 = load<VecType>(b + 0*ldb);
		auto b1 = load<VecType>(b + 1*ldb);
		auto b2 = load<VecType>(b + 2*ldb);
		auto b3 = load<VecType>(b + 3*ldb);

		if constexpr (TransB) transpose(b0, b1, b2, b3);

		// 4x4 trsm
		if constexpr (Forward) {
			// 4x2 trsm
			if constexpr (NoUnit) {
				if constexpr (recip) b0 = mul_lane<0, ConjA>(b0,     a0 );
				else                 b0 = div               (b0,     a00);
			}
			b1 = fms_lane<0, ConjA>(b1, b0, a1);
			if constexpr (NoUnit) {
				if constexpr (recip) b1 = mul_lane<1, ConjA>(b1,     a1 );
				else                 b1 = div               (b1,     a11);
			}

			// 4x2x2 gemm
			b2 = fms_lane<0, ConjA>(b2, b0, a2);
			b2 = fms_lane<1, ConjA>(b2, b1, a2);
			b3 = fms_lane<0, ConjA>(b3, b0, a3);
			b3 = fms_lane<1, ConjA>(b3, b1, a3);

			// 4x2 trsm
			if constexpr (NoUnit) {
				if constexpr (recip) b2 = mul_lane<2, ConjA>(b2,     a2 );
				else                 b2 = div               (b2,     a22);
			}
			b3 = fms_lane<2, ConjA>(b3, b2, a3);
			if constexpr (NoUnit) {
				if constexpr (recip) b3 = mul_lane<3, ConjA>(b3,     a3 );
				else                 b3 = div               (b3,     a33);
			}
		}
		else {
			// 4x2 trsm
			if constexpr (NoUnit) {
				if constexpr (recip) b3 = mul_lane<3, ConjA>(b3,     a3 );
				else                 b3 = div               (b3,     a33);
			}
			b2 = fms_lane<3, ConjA>(b2, b3, a2);
			if constexpr (NoUnit) {
				if constexpr (recip) b2 = mul_lane<2, ConjA>(b2,     a2 );
				else                 b2 = div               (b2,     a22);
			}

			// 4x2x2 gemm
			b1 = fms_lane<3, ConjA>(b1, b3, a1);
			b1 = fms_lane<2, ConjA>(b1, b2, a1);
			b0 = fms_lane<3, ConjA>(b0, b3, a0);
			b0 = fms_lane<2, ConjA>(b0, b2, a0);

			// 4x2 trsm
			if constexpr (NoUnit) {
				if constexpr (recip) b1 = mul_lane<1, ConjA>(b1,     a1 );
				else                 b1 = div               (b1,     a11);
			}
			b0 = fms_lane<1, ConjA>(b0, b1, a0);
			if constexpr (NoUnit) {
				if constexpr (recip) b0 = mul_lane<0, ConjA>(b0,     a0 );
				else                 b0 = div               (b0,     a00);
			}
		}

		// restore b
		if constexpr (TransB) transpose(b0, b1, b2, b3);

		// store b
		store(b + 0*ldb, b0);
		store(b + 1*ldb, b1);
		store(b + 2*ldb, b2);
		store(b + 3*ldb, b3);

		// line up next 4x4 trsm
		if constexpr (TransB) b += 4*ldb;
		else                  b += 4;
	}

	// Ignore set-but-unused warnings given by GCC < 10
	PERFLIBS_UNUSED(recip);
	PERFLIBS_UNUSED(a0); PERFLIBS_UNUSED(a1); PERFLIBS_UNUSED(a2); PERFLIBS_UNUSED(a3);
	PERFLIBS_UNUSED(a00); PERFLIBS_UNUSED(a11); PERFLIBS_UNUSED(a22); PERFLIBS_UNUSED(a33);
}

template<typename AType, typename BType = AType, typename CType = AType,
         typename = std::enable_if_t<std::is_same_v<BType, CType>>>
using trsm_kernel_t = void (const AType * restrict a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
                                  BType * restrict b, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
                            kernel_inttype b_cntg, kernel_inttype b_strd);

/*
 * Performs a mx4 TRSM for right-sided or 4xn TRSM for left-sided problems.
 *
 * For right-sided problems, a m' x 4 TRSM is performed using kernel_impl, where
 * m' is the largest multiple of 4 lte to m. A TRSM is then performed on the
 * remaining 1, 2, or 3 rows using the reference kernel. Similarly for left-sided
 * problems.
 *
 * @tparam ScalarType  corresponds to the data type in the x part of xTRSM
 * @tparam Forward     Do forwards or backwards substitution
 * @tparam TransA      Transpose the 4x4 A matrix
 * @tparam TransB      Work on b matrix transposed. problem is left sided iff transb
 * @tparam NoUnit      Assume diagonal elements of A are all 1
 * @tparam ConjA       Conjugate elements of A
 */
template<typename ScalarType, bool Forward, bool TransA, bool TransB, bool NoUnit, bool ConjA>
PERFLIBS_LINALG_INLINE
void trsm_kernel(const ScalarType * restrict a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
                       ScalarType * restrict b, kernel_inttype b_cntg_step, kernel_inttype b_strd_step,
                kernel_inttype b_cntg, kernel_inttype b_strd) {
	int_type m, n, lda, ldb;
	if constexpr (TransA) lda = a_cntg_step;
	else                  lda = a_strd_step;
	if constexpr (TransB) {
		m = b_cntg;
		n = b_strd;
		ldb = b_strd_step;
	}
	else {
		m = b_strd;
		n = b_cntg;
		ldb = b_cntg_step;
	}

	if (b_cntg != 4) {
		trsm_reference_kernel<ScalarType, Forward, TransA, TransB, NoUnit, ConjA>(m, n, 1, a, lda, b, ldb);
		return;
	}

	// map scalar type to neon type. We operate on 4 values at a time,
	// with groups of 4 complex values arranged as RRRR IIII. The mapping is
	// as follows:
	//
	//   float          -> float32x4_t
	//   double         -> float64x2x2_t
	//   complex_float  -> float32x4x2_t
	//   complex_double -> float64x2x4_t
	using VecType = std::conditional_t<
		std::is_same_v<ScalarType, float>,
		float32x4_t,
		std::conditional_t<
			std::is_same_v<ScalarType, double>,
			float64x2x2_t,
			std::conditional_t<
				std::is_same_v<ScalarType, complex_float>,
				float32x4x2_t,
				float64x2x4_t  // assume std::is_same_v<ScalarType, complex_double>
			>
		>
	>;

	trsm_kernel_impl<ScalarType, VecType, Forward, TransA, TransB, NoUnit, ConjA>(b_strd, a, lda, b, ldb);

	if ((b_strd & 3) > 0) {
		b += (b_strd & ~3)*b_strd_step;
		if constexpr (TransB) n &= 3;
		else                  m &= 3;
		trsm_reference_kernel<ScalarType, Forward, TransA, TransB, NoUnit, ConjA>(m, n, 1, a, lda, b, ldb);
	}
}

} // anonymous namespace
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_TRSM_KERNELS_HPP
