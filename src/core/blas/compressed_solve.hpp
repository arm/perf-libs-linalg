/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_COMPRESSED_SOLVE_HPP
#define PERFLIBS_LINALG_COMPRESSED_SOLVE_HPP

#include "framework/linalg_blas_types.hpp"

namespace perflibs::linalg {

/*
 * ---TBSV---
 *
 * TBSV takes Matrix A and Vector X, and solves the following system of equations for Vector B
 * where A is triangular and banded (only k super or sub diagonals). E.g. Upper, k = 1
 *
 * | A00 A01  X  |   | B0 |   | X0 |
 * |             |   |    |   |    |
 * |  X  A11 A12 | . | B1 | = | X1 |
 * |             |   |    |   |    |
 * |  X   X  A22 |   | B2 |   | X2 |
 *
 * Expanding this multiplication:
 *
 * X0 = A00.B0 + A01.B1 + 0
 * X1 =      0 + A11.B1 + A12.B2
 * X2 =      0 +      0 + A22.B2
 *
 * Re-arrange to find B we find that for an upper matrix values depend on those below them
 *
 * B2 = (X2/A22)
 * B1 = (X1 - A12.B2)/A11
 * B0 = (X0 - A01.B1)/A00
 *
 * With higher values of k, we find there are more terms similar to (- A12.B2) for each non-diagonal element.
 *
 * If the rows of A are contiguous (i.e. transposed) the final value of B[i] is calculated at each step i,
 * by subtracting the dot product of the non-diagonal elements of row i of A and X from X[i], and then
 * dividing by the diagonal A[i,i]. The division can be skipped if the matrix A is unit.
 *
 * If the columns of A are contiguous (i.e. not transposed) then dot would be operating on non-contiguous
 * data, so instead the division is performed first to find the final value of B[i]. If A is unit the value
 * B[i] is already final. Once B[i] is final, all depending elements B[x] that include a term of the form
 * (- Axi.Bi) have this value subtracted from them using AXPY.
 * The value of B[x] will then be equal to the LHS of the division once i=x.
 *
 * This calculation is currently done using axpby with b set as 1 since there are no conjugate axpy kernels.
*/

namespace {
template <typename KernelAxpby, typename KernelDot>
class tri_band_sv {
	KernelAxpby kernel_axpby_;
	KernelDot  kernel_dot_;
private:
	template<enum perflibs_trans a_trans,
	         enum perflibs_uplo  a_uplo,
	         enum perflibs_diag  a_diag,
	         typename MatrixTypeA, typename MatrixTypeB>
	void runner(MatrixTypeA &A, MatrixTypeB &B) {
		const auto n = B.cntg();

		if constexpr (a_uplo == PERFLIBS_UPPER) {
			// Iterate over B bottom to top
			for (kernel_inttype i = (n-1); i >= 0; --i) {
				if constexpr (a_trans != PERFLIBS_NOTRANS) {
					const auto [band_start_cntg_pos, band_end_cntg_pos] = A.get_cntg_band_pos(i);
					const kernel_inttype length = band_end_cntg_pos - band_start_cntg_pos;

					// Skip if there are no subdiagonals
					if(length != 0) {
						// Calculate final value of B[i] based on contiguous row of A
						const kernel_inttype top = i+1;
						auto A_ptr = A.get_absolute_physical_elem(i, top);

						B(i,0,write) -= kernel_dot_(length, A_ptr, B.data() + (top * B.cntg_step()), 1, B.cntg_step());
					}
				}
				if constexpr (a_diag != PERFLIBS_UNIT) {
					B(i,0,write) /= A(i,i);
				}
				if constexpr (a_trans == PERFLIBS_NOTRANS) {
					const auto [band_start_strd_pos, band_end_strd_pos] = A.get_strd_band_pos(i);

					// Calculate portion of all B[i] affected by values in the column band contiguous with A[i, i]
					const kernel_inttype length = band_end_strd_pos - band_start_strd_pos;

					// Skip if there are no subdiagonals
					if(length != 0) {
						auto A_ptr = A.get_absolute_physical_elem(band_start_strd_pos, i);

						kernel_axpby_(length, -B(i,0), A_ptr, 1, B.data() + (band_start_strd_pos * B.cntg_step()), 1, B.cntg_step());
					}
				}
			}
		}
		else {
			// Iterate over B top to bottom
			for (kernel_inttype i = 0; i < n; ++i) {

				if constexpr (a_trans != PERFLIBS_NOTRANS) {
					// Calculate final value of B[i] based on contiguous row of A
					const auto [band_start_cntg_pos, band_end_cntg_pos] = A.get_cntg_band_pos(i);
					const kernel_inttype length = band_end_cntg_pos - band_start_cntg_pos;

					if(length != 0) {
						auto A_ptr = A.get_absolute_physical_elem(i, band_start_cntg_pos);

						B(i,0,write) -= kernel_dot_(length, A_ptr, B.data() + (band_start_cntg_pos * B.cntg_step()), 1, B.cntg_step());
					}

				}
				if constexpr (a_diag != PERFLIBS_UNIT) {
					B(i,0,write) /= A(i,i);
				}
				if constexpr (a_trans == PERFLIBS_NOTRANS) {
					// Calculate portion of all B[i] affected by values in the column band contiguous with A[i, i]
					const auto [band_start_strd_pos, band_end_strd_pos] = A.get_strd_band_pos(i);
					const kernel_inttype length = band_end_strd_pos - band_start_strd_pos;

					// Skip if there are no subdiagonals
					if(length != 0) {
						const kernel_inttype top = band_start_strd_pos + 1;
						auto A_ptr = A.get_absolute_physical_elem(top, i);

						kernel_axpby_( length, -B(i,0), A_ptr, 1, B.data() + (top * B.cntg_step()), 1, B.cntg_step());
					}
				}
			}
		}
	}
public:
	tri_band_sv(KernelAxpby kernel_axpby, KernelDot kernel_dot)
	: kernel_axpby_ { std::move(kernel_axpby) }
	, kernel_dot_   { std::move(kernel_dot)   }
	{ }

	template<typename MatrixTypeA, typename MatrixTypeB, typename... Args>
	inline
	void operator()(MatrixTypeA &A, MatrixTypeB &B, Args &&... args) {
		// Convert parameters to compile time constants
		if (A.is_trans())
			if (A.is_lower())
				if (A.is_unit())
					runner<PERFLIBS_TRANS,   PERFLIBS_LOWER, PERFLIBS_UNIT  >(A, B);
				else
					runner<PERFLIBS_TRANS,   PERFLIBS_LOWER, PERFLIBS_NOUNIT>(A, B);
			else
				if (A.is_unit())
					runner<PERFLIBS_TRANS,   PERFLIBS_UPPER, PERFLIBS_UNIT  >(A, B);
				else
					runner<PERFLIBS_TRANS,   PERFLIBS_UPPER, PERFLIBS_NOUNIT>(A, B);
		else
			if (A.is_lower())
				if (A.is_unit())
					runner<PERFLIBS_NOTRANS, PERFLIBS_LOWER, PERFLIBS_UNIT  >(A, B);
				else
					runner<PERFLIBS_NOTRANS, PERFLIBS_LOWER, PERFLIBS_NOUNIT>(A, B);
			else
				if (A.is_unit())
					runner<PERFLIBS_NOTRANS, PERFLIBS_UPPER, PERFLIBS_UNIT  >(A, B);
				else
					runner<PERFLIBS_NOTRANS, PERFLIBS_UPPER, PERFLIBS_NOUNIT>(A, B);
	}
}; // tri_band_sv
} // namespace anon
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_COMPRESSED_SOLVE_HPP
