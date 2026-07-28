/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_BLOCK_REFLECTOR_HPP
#define PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_BLOCK_REFLECTOR_HPP

#include "spec/problem_context.hpp"
#include "framework/compute.hpp"

#include "matrix/matrix.hpp"
#include "perflibs_util.hpp"

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/matmul/fwd.hpp"

#include <string_view>

namespace perflibs::linalg::factorization {

class apply_block_reflector_sequential {
	template<typename VMatrixType, typename TriangularMatrixType, typename CMatrixType, typename ArchitectureSpec>
	using left_pctx_t = spec::problem_context<
		apply_block_reflector<block_reflector<VMatrixType, TriangularMatrixType>, CMatrixType>,
		ArchitectureSpec
	>;

	template<typename CMatrixType, typename VMatrixType, typename TriangularMatrixType, typename ArchitectureSpec>
	using right_pctx_t = spec::problem_context<
		apply_block_reflector<CMatrixType, block_reflector<VMatrixType, TriangularMatrixType>>,
		ArchitectureSpec
	>;

public:
	static constexpr std::string_view name() { return "apply_block_reflector_sequential"; }

	template<typename VMatrixType, typename TriangularMatrixType, typename CMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const left_pctx_t<VMatrixType, TriangularMatrixType, CMatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename CMatrixType::value_type;
		const auto& reflector = pctx.reflector;
		auto matrix = pctx.matrix;

		const kernel_inttype m = matrix.cntg();
		const kernel_inttype n = matrix.strd();
		const kernel_inttype k = reflector.v.strd();

		if (m <= 0 || n <= 0 || k <= 0) return true;

		// work stores the intermediate n-by-k product used for the block update.
		auto work = pctx.work.sub_matrix(0, n, 0, k);

		// The block reflector layout selects whether the active unit-triangular
		// block lives at the front or back of V and C; the operator carried by T
		// is independent.
		const auto block_offset = reflector.active_offset(m);
		auto c_block = matrix.sub_matrix(block_offset, k, 0, n);
		auto c_rest  = block_offset == 0 ? matrix.sub_matrix(k, m - k, 0, n)
		                                 : matrix.sub_matrix(0, m - k, 0, n);
		auto v_rest  = reflector.remainder();
		auto v_diag  = reflector.unit_block();

		// Form W from the active block and the remaining reflector basis rows.
		accumulate_product(
			toggle_conj(c_block),
			v_diag,
			work,
			one<value_type>,
			zero<value_type>,
			pctx.architecture_spec
		);
		if (m > k) {
			accumulate_product(
				toggle_conj(c_rest),
				v_rest,
				work,
				one<value_type>,
				one<value_type>,
				pctx.architecture_spec
			);
		}
		// Apply the triangular factor through the stored work representation.
		apply_triangular_factor(adjoint(reflector.t), work, one<value_type>, pctx.architecture_spec);
		// Update the remaining rows of C.
		if (m > k) {
			accumulate_product(
				v_rest.transpose(),
				adjoint(work),
				c_rest,
				-one<value_type>,
				one<value_type>,
				pctx.architecture_spec
			);
		}
		// Fold the active rows of W back into the leading/trailing block of C.
		apply_triangular_factor(adjoint(v_diag), work, one<value_type>, pctx.architecture_spec);
		for (kernel_inttype j = 0; j < c_block.cntg(); ++j) {
			for (kernel_inttype i = 0; i < c_block.strd(); ++i) {
				c_block(j, i, write) = c_block(j, i) - perflibs::conj(work(i, j));
			}
		}

		return true;
	}

	template<typename CMatrixType, typename VMatrixType, typename TriangularMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator()(const right_pctx_t<CMatrixType, VMatrixType, TriangularMatrixType, ArchitectureSpec>& pctx) const {
		if (!can_compute(pctx)) return false;

		using value_type = typename CMatrixType::value_type;
		auto matrix = pctx.matrix;
		const auto& reflector = pctx.reflector;

		const kernel_inttype m = matrix.cntg();
		const kernel_inttype n = matrix.strd();
		const kernel_inttype k = reflector.v.strd();

		if (m <= 0 || n <= 0 || k <= 0) return true;

		// work stores the intermediate m-by-k product used for the block update.
		auto work = pctx.work.sub_matrix(0, m, 0, k);

		const auto block_offset = reflector.active_offset(n);
		auto c_block = matrix.sub_matrix(0, m, block_offset, k);
		auto c_rest  = block_offset == 0 ? matrix.sub_matrix(0, m, k, n - k)
		                                 : matrix.sub_matrix(0, m, 0, n - k);
		auto v_rest  = reflector.remainder();
		auto v_diag  = reflector.unit_block();

		// Form W from the active block and the remaining reflector basis columns.
		accumulate_product(
			c_block.transpose(),
			v_diag,
			work,
			one<value_type>,
			zero<value_type>,
			pctx.architecture_spec
		);
		if (n > k) {
			accumulate_product(
				c_rest.transpose(),
				v_rest,
				work,
				one<value_type>,
				one<value_type>,
				pctx.architecture_spec
			);
		}
		// Apply the triangular factor through the stored work representation.
		apply_triangular_factor(reflector.t, work, one<value_type>, pctx.architecture_spec);
		// Update the remaining columns of C.
		if (n > k) {
			accumulate_product(
				work.transpose(),
				adjoint(v_rest),
				c_rest,
				-one<value_type>,
				one<value_type>,
				pctx.architecture_spec
			);
		}
		// Fold the active columns of W back into the leading/trailing block of C.
		accumulate_product(
			work.transpose(),
			adjoint(v_diag),
			c_block,
			-one<value_type>,
			one<value_type>,
			pctx.architecture_spec
		);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const {
		return false;
	}

	template<typename VMatrixType, typename TriangularMatrixType, typename CMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const left_pctx_t<VMatrixType, TriangularMatrixType, CMatrixType, ArchitectureSpec>&) const {
		return true;
	}

	template<typename CMatrixType, typename VMatrixType, typename TriangularMatrixType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const right_pctx_t<CMatrixType, VMatrixType, TriangularMatrixType, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return false;
	}

private:
	template<typename TriangularMatrixType, typename WorkMatrixType, typename ValueType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	static void apply_triangular_factor(const TriangularMatrixType& matrix, WorkMatrixType work,
	                                    const ValueType& alpha, const ArchitectureSpec& arch) {
		spec::problem_context pctx {
			matmul::matmul2 {
				to_const(matrix),
				work.transpose(),
				alpha,
				zero<ValueType>
			},
			arch
		};
		compute(pctx);
	}

	template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ValueType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	static void accumulate_product(const AMatrixType& a, const BMatrixType& b, CMatrixType c,
	                               const ValueType& alpha, const ValueType& beta, const ArchitectureSpec& arch) {
		spec::problem_context pctx {
			matmul::matmul3 {
				to_const(a),
				to_const(b),
				c,
				alpha,
				beta
			},
			arch
		};
		compute(pctx);
	}

};
} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_STRATEGIES_APPLY_BLOCK_REFLECTOR_HPP
