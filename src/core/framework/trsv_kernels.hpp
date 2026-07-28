/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TRSV_KERNELS_HPP
#define PERFLIBS_LINALG_TRSV_KERNELS_HPP

#include "framework/axpby_kernels.hpp"
#include "framework/dot_kernels.hpp"
#include "framework/linalg_util.hpp"
#include "perflibs_util.hpp"

namespace perflibs::linalg {
namespace {

template<typename AType, typename BType=AType, typename CType=AType,
         typename = std::enable_if_t<std::is_same_v<BType, CType>>>
using trsv_kernel_t = void(const AType *restrict a, kernel_inttype a_cntg_step, kernel_inttype a_strd_step,
                           CType *restrict x, kernel_inttype x_cntg,
                           axpby_kernel_t<AType, BType, CType> axpby,
                           dot_kernel_t<AType, BType, CType> dot);

template<typename ScalarType, bool NoUnit>
PERFLIBS_LINALG_INLINE
void trsv_notrans_upper(const ScalarType * restrict a, kernel_inttype lda, kernel_inttype,
                              ScalarType * restrict x, kernel_inttype n,
                        axpby_kernel_t<ScalarType> axpby, dot_kernel_t<ScalarType>) {
	for (kernel_inttype j = n - 1; j >= 0; --j) {
		// x(j) <- x(j) / a(j, j)
		if constexpr (NoUnit) x[j] /= a[j * (1 + lda)];

		// x(:j-1) <- x(0:j-1) - x(j) * a(0:j-1, j)
		const kernel_inttype i = 0;
		const ScalarType * a_ij = a + i + j*lda;
		axpby(j, -x[j], a_ij, 1, x + i, 1, 1);
	}
}

template<typename ScalarType, bool NoUnit>
PERFLIBS_LINALG_INLINE
void trsv_notrans_lower(const ScalarType * restrict a, kernel_inttype lda, kernel_inttype,
                              ScalarType * restrict x, kernel_inttype n,
                        axpby_kernel_t<ScalarType> axpby, dot_kernel_t<ScalarType>) {
	for (kernel_inttype j = 0; j < n; ++j) {
		// x(j) <- x(j) / a(j, j)
		if constexpr (NoUnit) x[j] /= a[j * (1 + lda)];

		// x(j+1:n-1) <- x(j+1:n-1) - x(j) * a(j+1:n-1, j)
		const kernel_inttype i = j + 1;
		const ScalarType * a_ij = a + i + j*lda;
		axpby(n - i, -x[j], a_ij, 1, x + i, 1, 1);
	}
}

template<typename ScalarType, bool NoUnit, bool Conj>
PERFLIBS_LINALG_INLINE
void trsv_trans_upper(const ScalarType * restrict a, kernel_inttype, kernel_inttype lda,
                            ScalarType * restrict x, kernel_inttype n,
                      axpby_kernel_t<ScalarType>, dot_kernel_t<ScalarType> dot) {
	for (kernel_inttype j = 0; j < n; ++j) {
		// x(j) <- x(j) - dot( a(0:j-1, j), x(0:j-1) )
		const kernel_inttype i = 0;
		const ScalarType * a_ij = a + i + j*lda;
		x[j] -= dot(j, a_ij, x + i, 1, 1);

		// x(j) <- x(j) / a(j, j)
		if constexpr (NoUnit) {
			if constexpr (Conj) x[j] /= conj(a[j * (1 + lda)]);
			else                x[j] /=      a[j * (1 + lda)] ;
		}
	}
}

template<typename ScalarType, bool NoUnit, bool Conj>
PERFLIBS_LINALG_INLINE
void trsv_trans_lower(const ScalarType * restrict a, kernel_inttype, kernel_inttype lda,
                            ScalarType * restrict x, kernel_inttype n,
                      axpby_kernel_t<ScalarType>, dot_kernel_t<ScalarType> dot) {
	for (kernel_inttype j = n - 1; j >= 0; --j) {
		// x(j) <- x(j) - dot( a(j+1:n-1, j), x(j+1:n-1) )
		const kernel_inttype i = j + 1;
		const ScalarType * a_ij = a + i + j*lda;
		x[j] -= dot(n - i, a_ij, x + i, 1, 1);

		// x(j) <- x(j) / a(j, j)
		if constexpr (NoUnit) {
			if constexpr (Conj) x[j] /= conj(a[j * (1 + lda)]);
			else                x[j] /=      a[j * (1 + lda)] ;
		}
	}
}

} // namespace anonymous
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_TRSV_KERNELS_HPP
