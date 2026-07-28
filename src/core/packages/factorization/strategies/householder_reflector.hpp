/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_REFLECTOR_STRATEGIES_HPP
#define PERFLIBS_LINALG_FACTORIZATION_REFLECTOR_STRATEGIES_HPP

#include "matrix/matrix.hpp"
#include "framework/linalg_util.hpp"
#include "spec/problem_context.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/helpers/householder_numeric.hpp"
#include "packages/factorization/kernels/householder_reflector_kernels.hpp"

#include "packages/matmul/interfaces/scal.hpp"
#include "packages/matmul/fwd.hpp"

#include "packages/misc/interfaces/nrm2.hpp"

#include "perflibs_util.hpp"

namespace perflibs::linalg::factorization {

class generate_reflector_basic {
	template<typename MatrixType, typename ArchitectureSpec>
	using reflector_pctx_t = spec::problem_context<generate_reflector<MatrixType>, ArchitectureSpec>;

public:
	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const reflector_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename MatrixType::value_type;
		using real_type  = remove_complex_t<value_type>;

		// Constants
		constexpr real_type   rone                = real_type(1.0);
		constexpr value_type  one                 = value_type(1.0);
		constexpr pl_linalg_int_t scaling_bound       = 20;
		const     real_type   safe_min            = safe_scaling_threshold<value_type>();
		const     auto        reciprocal_safe_min = rone / safe_min;
		const     pl_linalg_int_t length_x            = pctx.x.cntg();
		const     pl_linalg_int_t incx                = pctx.x.cntg_step();

		// Compute XNORM = ||x||
		auto xnorm = perflibs::linalg::nrm2<false, pl_linalg_int_t, value_type, real_type, ArchitectureSpec>(&length_x, pctx.x.data(), &incx);

		if (xnorm == 0.0 && imag(pctx.alpha) == 0.0) {
			// H = I
			pctx.tau = 0.0;
		}
		else {
			// Generate elementary reflector H such that:
			//   H**H * (alpha) = (beta)  where alpha = pctx.alpha
			//          (  x  ) = (0  )          x    = pctx.x

			auto beta = compute_householder_beta(pctx.alpha, xnorm);
			auto scaling_iterations = 0_ki;
			if (std::abs(beta) < safe_min) {
				do {
					scaling_iterations++;
					perflibs::linalg::scal<false, pl_linalg_int_t, real_type, value_type, ArchitectureSpec>(&length_x, &reciprocal_safe_min, pctx.x.data(), &incx);
					beta       *= reciprocal_safe_min;
					pctx.alpha *= reciprocal_safe_min;
				} while (std::abs(beta) < safe_min && scaling_iterations < scaling_bound);

				xnorm = perflibs::linalg::nrm2<false, pl_linalg_int_t, value_type, real_type, ArchitectureSpec>(&length_x, pctx.x.data(), &incx);
				beta = compute_householder_beta(pctx.alpha, xnorm);
			}

			pctx.tau = (beta - pctx.alpha) / beta;
			value_type scaling_factor = one / (pctx.alpha - beta);
			perflibs::linalg::scal<false, pl_linalg_int_t, value_type, value_type, ArchitectureSpec>(&length_x, &scaling_factor, pctx.x.data(), &incx);

			for (auto j = 0_ki; j < scaling_iterations; j++) {
				beta *= safe_min;
			}
			pctx.alpha = beta;
		}
		return true;
	}

	template<typename ProblemContext>
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const reflector_pctx_t<MatrixType, ArchitectureSpec>& pctx) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; // class generate_reflector


class form_block_reflector_factor_sequential {
	template<typename MatrixType, typename TriangularMatrixType, typename ArchitectureSpec>
	using larft_pctx_t = spec::problem_context<
		form_block_reflector_factor<MatrixType, TriangularMatrixType>,
		ArchitectureSpec
	>;

public:
	template<typename MatrixType, typename TriangularMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const larft_pctx_t<MatrixType, TriangularMatrixType, ArchitectureSpec>& pctx) const {

		if (!can_compute(pctx)) return false;

		const auto direct = direct_to_c(pctx.direct);
		const auto storev = storev_to_c(pctx.storev);

		kernel_inttype v_strd_step;
		if (pctx.storev  == PERFLIBS_COLUMNWISE) {
			v_strd_step = pctx.v.strd_step();
		}
		else {
			v_strd_step = pctx.v.cntg_step();
		}
		larft<ArchitectureSpec>(direct, storev, pctx.v.cntg(), pctx.v.strd(), pctx.v.data(), v_strd_step,
		                        pctx.tau.data(), pctx.t.data(), pctx.t.strd_step());
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename MatrixType, typename TriangularMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE constexpr bool
	can_compute(const larft_pctx_t<MatrixType, TriangularMatrixType, ArchitectureSpec>&) const {
		// This strategy can compute any LARFT
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
};

class apply_elementary_reflector_basic {
	template<typename VMatrixType, typename CMatrixType, typename WMatrixType, typename ArchitectureSpec>
	using larf_pctx_t = spec::problem_context<apply_elementary_reflector<VMatrixType, CMatrixType, WMatrixType>, ArchitectureSpec>;

public:
	template<typename VMatrixType, typename CMatrixType, typename WMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const larf_pctx_t<VMatrixType, CMatrixType, WMatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename CMatrixType::value_type;

		if (pctx.tau == zero<>) return true;

		const bool is_lside  = is_left(pctx.side);
		kernel_inttype lastv = is_lside ? pctx.c.cntg() : pctx.c.strd();

		// Set up variables for scanning V
		// lastv begins pointing to the end of V.
		kernel_inttype i;
		if (pctx.v.cntg_step() > 0) {
			i = lastv - 1;
		}
		else {
			i = 0;
		}

		// Look for the last non-zero element in v
		while (lastv > 0 && pctx.v(i, 0) == zero<>) {
			lastv--;
			i--;
		}

		// Find extent of non-zero data from the end
		kernel_inttype lastc;
		if (is_lside) {
			lastc = count_cols_to_last_nonzero(pctx.c, lastv);
		}
		else {
			lastc = count_rows_to_last_nonzero(pctx.c, lastv);
		}

		if (lastc == 0) return true;

		// Apply H = I - tau * v * v**H from the left
		if (is_lside) {
			// w := C**H * v
			spec::problem_context gemv_pctx{
				matmul::matmul3{
					to_const(to_conj(pctx.c.sub_matrix(0, lastv, 0, lastc))),
					to_const(pctx.v.sub_matrix(0, lastv, 0, 1)),
					pctx.work.sub_matrix(0, lastc, 0, 1),
					one<value_type>, zero<value_type>
				},
				pctx.architecture_spec
			};
			compute(gemv_pctx);

			// C := C - v * w**H
			spec::problem_context ger_pctx{
				matmul::matmul3{
					to_const(pctx.v.sub_matrix(0, lastv, 0, 1).transpose()),
					to_const(to_conj(pctx.work.sub_matrix(0, lastc, 0, 1).transpose())),
					pctx.c.sub_matrix(0, lastv, 0, lastc),
					-pctx.tau, one<value_type>
				},
				pctx.architecture_spec
			};
			compute(ger_pctx);
		}
		// Apply H = I - tau * v * v**H from the right
		else {
			// w := C * v
			spec::problem_context gemv_pctx{
				matmul::matmul3{
					to_const(pctx.c.sub_matrix(0, lastc, 0, lastv).transpose()),
					to_const(pctx.v.sub_matrix(0, lastv, 0, 1)),
					pctx.work.sub_matrix(0, lastc, 0, 1),
					one<value_type>, zero<value_type>
				},
				pctx.architecture_spec
			};
			compute(gemv_pctx);

			// C := C - w * v**H
			spec::problem_context ger_pctx{
				matmul::matmul3{
					to_const(general_matrix { matrix_base { pctx.work.data(), 1, lastc, 0, 1 },              false }),
					to_const(toggle_conj(pctx.v.sub_matrix(0, lastv, 0, 1).transpose())),
					pctx.c.sub_matrix(0, lastc, 0, lastv),
					-pctx.tau, one<value_type>
				},
				pctx.architecture_spec
			};
			compute(ger_pctx);
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename VMatrixType, typename CMatrixType, typename WMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const larf_pctx_t<VMatrixType, CMatrixType, WMatrixType, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

private:
	template<typename MatrixType>
	PERFLIBS_LINALG_INLINE
	kernel_inttype count_cols_to_last_nonzero(const MatrixType& c, kernel_inttype m) const {
		for (kernel_inttype j = c.strd() - 1; j >= 0; --j) {
			for (kernel_inttype i = 0; i < m; ++i) {
				if (c(i, j) != zero<>) {
					return j + 1;
				}
			}
		}
		return 0;
	}

	template<typename MatrixType>
	PERFLIBS_LINALG_INLINE
	kernel_inttype count_rows_to_last_nonzero(const MatrixType& c, kernel_inttype n) const {
		for (kernel_inttype i = c.cntg() - 1; i >= 0; --i) {
			for (kernel_inttype j = 0; j < n; ++j) {
				if (c(i, j) != zero<>) {
					return i + 1;
				}
			}
		}
		return 0;
	}
}; // class apply_elementary_reflector_basic
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_REFLECTOR_STRATEGIES_HPP
