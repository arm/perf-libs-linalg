/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_LDL_PACKED_NO_PIVOT_KERNEL_HPP
#define PERFLIBS_LINALG_LDL_PACKED_NO_PIVOT_KERNEL_HPP

#include "framework/linalg_util.hpp"
#include "packages/factorization/problem_context_bases.hpp"

namespace perflibs::linalg {

/**
 *  A basic implementation of partial LDL^T factorization without pivoting
 *  in lower packed storage (right-looking)
 */
template<typename ArchitectureSpec, typename T>
void ldl_packed_no_pivot_kernel(const kernel_inttype n, const kernel_inttype ncolm, T *ap,
                                const factorization::ldl_packed_output output) {

	PERFLIBS_ASSERT(n >= 0, " The order of the matrix A must be greater or equal to zero");
	PERFLIBS_ASSERT(ncolm >= 0 && ncolm <= n, " The number of columns to eliminate must be in [0, n]");

	kernel_inttype pk = 0;  // packed offset of A(k,k)
	for (kernel_inttype k = 0; k < ncolm; k++) {
		const kernel_inttype len  = n - k;   // elements in column k, diagonal included
		// an exactly zero pivot has reciprocal zero: it updates nothing
		const T              rcpd = (ap[pk] == T(0)) ? T(0) : T(1) / ap[pk];

		// update the columns to the right with column k
		kernel_inttype pj = pk + len;        // packed offset of A(j,j)
		for (kernel_inttype j = k + 1; j < n; j++) {
			const T  ljk = ap[pk + (j - k)] * rcpd;
			const T *x   = ap + pk + (j - k);
			T       *c   = ap + pj;
			const kernel_inttype m = n - j;
			for (kernel_inttype i = 0; i < m; i++) {
				c[i] = c[i] - x[i] * ljk;
			}
			pj += m;
		}

		// store column k
		if (output == factorization::ldl_packed_output::inverse_diagonal) {
			ap[pk] = rcpd;
		}
		else {
			for (kernel_inttype i = 1; i < len; i++) {
				ap[pk + i] = ap[pk + i] * rcpd;
			}
		}
		pk += len;
	}
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_LDL_PACKED_NO_PIVOT_KERNEL_HPP
