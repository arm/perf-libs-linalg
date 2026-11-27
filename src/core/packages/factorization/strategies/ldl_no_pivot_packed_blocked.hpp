/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LDL_NO_PIVOT_PACKED_BLOCKED_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LDL_NO_PIVOT_PACKED_BLOCKED_HPP

#include "framework/compute.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/fwd.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/matmul/fwd.hpp"

#include "matrix/matrix.hpp"
#include "matrix/adaptors.hpp"

#include "pod_vector.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

/*
 * Blocked partial LDL^T factorization without pivoting, lower packed storage.
 *
 * For each panel of panel_width columns:
 *   1. copy the panel (all rows below its first diagonal element) into a dense buffer X
 *   2. factor the panel in X recursively, keeping X = L*D (unscaled) and V = L
 *   3. update the trailing columns block by block: copy block_width packed columns into a
 *      dense buffer T, T = T - X * V^T on its lower part, copy that part back
 *   4. store the panel columns in the requested output form
 */
class ldl_no_pivot_packed_blocked {

	template<typename DataType, typename ArchitectureSpec>
	using ldl_packed_pctx_t = spec::problem_context<ldl_factorization_no_pivot_packed<packed_matrix_base<DataType>>, ArchitectureSpec>;

	// packed offset of A(j,j) in a lower packed matrix of order n
	PERFLIBS_LINALG_INLINE
	static constexpr kernel_inttype diag_offset(kernel_inttype j, kernel_inttype n) { return j * n - j * (j - 1) / 2; }

	// C = C - A * B^T on the lower trapezoid of the mr x w block C
	template<typename T, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	static void lower_update(T *c, kernel_inttype ldc, kernel_inttype mr, kernel_inttype w,
	                         const T *a, kernel_inttype lda, const T *b, kernel_inttype ldb,
	                         kernel_inttype kk, const ArchitectureSpec& architecture_spec) {
		const auto b_op = to_const(general_matrix{ matrix_base{ const_cast<T *>(b), kk, w, ldb, kernel_inttype{ 1 } } });

		auto diagonal_pctx = spec::problem_context{
			matmul::matmul3{
				to_const(general_matrix{ matrix_base{ const_cast<T *>(a), kk, w, lda, kernel_inttype{ 1 } } }),
				b_op,
				triangular_matrix{ PERFLIBS_LOWER, PERFLIBS_NOUNIT, matrix_base{ c, w, w, kernel_inttype{ 1 }, ldc } },
				T(-1), one<T> },
			architecture_spec
		};
		matmul::compute(diagonal_pctx);

		if (mr > w) {
			auto below_pctx = spec::problem_context{
				matmul::matmul3{
					to_const(general_matrix{ matrix_base{ const_cast<T *>(a) + w, kk, mr - w, lda, kernel_inttype{ 1 } } }),
					b_op,
					general_matrix{ matrix_base{ c + w, mr - w, w, kernel_inttype{ 1 }, ldc } },
					T(-1), one<T> },
				architecture_spec
			};
			matmul::compute(below_pctx);
		}
	}

	// factor columns [c0, c1) of the dense panel: X keeps L*D, V gets L
	template<typename T, typename ArchitectureSpec>
	static void factor_panel(T *X, T *V, kernel_inttype m0, kernel_inttype c0, kernel_inttype c1,
	                         kernel_inttype base_width, const ArchitectureSpec& architecture_spec) {
		if (c1 - c0 <= base_width) {
			for (kernel_inttype l = c0; l < c1; l++) {
				const T d    = X[l + l * m0];
				const T rcpd = (d == T(0)) ? T(0) : T(1) / d;   // a zero pivot updates nothing
				for (kernel_inttype r = l + 1; r < m0; r++) {
					V[r + l * m0] = X[r + l * m0] * rcpd;
				}
				for (kernel_inttype j = l + 1; j < c1; j++) {
					const T ljl = V[j + l * m0];
					for (kernel_inttype r = j; r < m0; r++) {
						X[r + j * m0] = X[r + j * m0] - X[r + l * m0] * ljl;
					}
				}
			}
			return;
		}
		const kernel_inttype mid = c0 + (c1 - c0) / 2;
		factor_panel(X, V, m0, c0, mid, base_width, architecture_spec);
		lower_update(X + mid + mid * m0, m0, m0 - mid, c1 - mid,
		             X + mid + c0 * m0, m0, V + mid + c0 * m0, m0, mid - c0, architecture_spec);
		factor_panel(X, V, m0, mid, c1, base_width, architecture_spec);
	}

public:
	static constexpr std::string_view name() { return "ldl_no_pivot_packed_blocked"; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ldl_packed_pctx_t<DataType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using T = DataType;
		const kernel_inttype n     = pctx.a.cntg();
		const kernel_inttype ncolm = pctx.ncolm;
		T *const             ap    = pctx.a.data();

		const auto spec = get_spec(spec::strategy_tag<ldl_factorization_no_pivot_packed_generic>{}, pctx);
		const kernel_inttype block_width = spec.block_width;

		const kernel_inttype nb = min(spec.panel_width, ncolm);
		pod_vector<T> xbuf(static_cast<std::size_t>(n) * nb);
		pod_vector<T> vbuf(static_cast<std::size_t>(n) * nb);
		pod_vector<T> tbuf(static_cast<std::size_t>(n) * min(block_width, n));

		for (kernel_inttype k0 = 0; k0 < ncolm; k0 += nb) {
			const kernel_inttype b  = min(nb, ncolm - k0);
			const kernel_inttype m0 = n - k0;   // rows of the panel, leading dimension of X and V
			T *const X = xbuf.data();
			T *const V = vbuf.data();

			// 1. X(r,l) = A(k0+r, k0+l) for r >= l
			for (kernel_inttype l = 0; l < b; l++) {
				const T *col = ap + diag_offset(k0 + l, n);
				for (kernel_inttype r = l; r < m0; r++) {
					X[r + l * m0] = col[r - l];
				}
			}

			// 2. factor the panel
			factor_panel(X, V, m0, kernel_inttype{ 0 }, b, spec.base_width, pctx.architecture_spec);

			// 3. A(j0:, j0:j0+w-1) -= X(j0-k0:, :) * V(j0-k0 : j0-k0+w-1, :)^T on the lower part
			for (kernel_inttype j0 = k0 + b; j0 < n; j0 += block_width) {
				const kernel_inttype w  = min(block_width, n - j0);
				const kernel_inttype mt = n - j0;   // rows of the block, leading dimension of T
				const kernel_inttype r0 = j0 - k0;  // first row of the block in X and V
				T *const Tb = tbuf.data();

				for (kernel_inttype c = 0; c < w; c++) {
					const T *col = ap + diag_offset(j0 + c, n);
					for (kernel_inttype r = c; r < mt; r++) {
						Tb[r + c * mt] = col[r - c];
					}
				}

				lower_update(Tb, mt, mt, w, X + r0, m0, V + r0, m0, b, pctx.architecture_spec);

				for (kernel_inttype c = 0; c < w; c++) {
					T *col = ap + diag_offset(j0 + c, n);
					for (kernel_inttype r = c; r < mt; r++) {
						col[r - c] = Tb[r + c * mt];
					}
				}
			}

			// 4. store the panel columns
			for (kernel_inttype l = 0; l < b; l++) {
				T *col = ap + diag_offset(k0 + l, n);
				const T d = X[l + l * m0];
				if (pctx.output == ldl_packed_output::inverse_diagonal) {
					col[0] = (d == T(0)) ? T(0) : T(1) / d;
					for (kernel_inttype r = l + 1; r < m0; r++) col[r - l] = X[r + l * m0];
				}
				else {
					col[0] = d;
					for (kernel_inttype r = l + 1; r < m0; r++) col[r - l] = V[r + l * m0];
				}
			}
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const ldl_packed_pctx_t<DataType, ArchitectureSpec>& pctx) const {
		// smaller problems use ldl_no_pivot_packed_unblocked
		const auto spec = get_spec(spec::strategy_tag<ldl_factorization_no_pivot_packed_generic>{}, pctx);
		return pctx.a.cntg() >= spec.min_order && pctx.ncolm >= spec.min_columns;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class ldl_no_pivot_packed_blocked
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_LDL_NO_PIVOT_PACKED_BLOCKED_HPP
