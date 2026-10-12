/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_PROBLEM_CONTEXT_BASES_HPP

#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/batch_problem.hpp"

#include <utility>

namespace perflibs::linalg::batch_interleave::solve {

template<typename AMatrixType, typename BMatrixType, typename ScalarType>
struct triangular_solve {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	scalar_type alpha;
}; // struct triangular_solve

} // namespace perflibs::linalg::batch_interleave::solve

namespace perflibs::linalg::batch_interleave {

template<typename AMatrixType, typename BMatrixType, typename ScalarType>
struct batch_problem<solve::triangular_solve<AMatrixType, BMatrixType, ScalarType>> {
	using problem_type  = solve::triangular_solve<AMatrixType, BMatrixType, ScalarType>;
	using a_matrix_type = strided_batch_view<AMatrixType>;
	using b_matrix_type = strided_batch_view<BMatrixType>;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	scalar_type alpha;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_, b_matrix_type b_, scalar_type alpha_)
	: a      { std::move(a_) }
	, b      { std::move(b_) }
	, alpha  { alpha_        }
	{
		const auto& a_matrix = a.matrix();
		const auto& b_matrix = b.matrix();

		PERFLIBS_ASSERT(a_matrix.cntg() == a_matrix.strd() && a_matrix.cntg() == b_matrix.cntg(),
		             "batch solve matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == b.batch_count(),
		             "batch solve matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == b_matrix.interleave_count(),
		             "batch solve matrices have incompatible interleave counts");
	}

	PERFLIBS_LINALG_INLINE
	kernel_inttype batch_count() const {
		return a.batch_count();
	}

	PERFLIBS_LINALG_INLINE
	kernel_inttype interleave_count() const {
		return a.matrix().interleave_count();
	}

	PERFLIBS_LINALG_INLINE
	problem_type problem(kernel_inttype index) const {
		return {
			a[index],
			b[index],
			alpha
		};
	}
}; // struct batch_problem

namespace solve {

template<typename ScalarType>
using solve_problem = batch_problem<
	triangular_solve<
		triangular_interleave_batch_matrix<const ScalarType>,
		general_interleave_batch_matrix<ScalarType>,
		ScalarType
	>
>;

} // namespace solve

} // namespace perflibs::linalg::batch_interleave

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_SOLVE_PROBLEM_CONTEXT_BASES_HPP
