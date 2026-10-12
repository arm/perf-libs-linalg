/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_PROBLEM_CONTEXT_BASES_HPP

#include "packages/batch_interleave/batch_problem.hpp"
#include "packages/matmul/problem_context_bases.hpp"

#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"

#include <utility>

namespace perflibs::linalg::batch_interleave {

template<typename AMatrixType, typename BMatrixType, typename ScalarType>
struct batch_problem<::perflibs::linalg::matmul::matmul2<AMatrixType, BMatrixType, ScalarType>> {
	using problem_type  = ::perflibs::linalg::matmul::matmul2<AMatrixType, BMatrixType, ScalarType>;
	using a_matrix_type = strided_batch_view<AMatrixType>;
	using b_matrix_type = strided_batch_view<BMatrixType>;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;

	scalar_type alpha;
	scalar_type beta;

	zero_mode beta_zero_mode;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_, b_matrix_type b_, scalar_type alpha_, scalar_type beta_,
	              zero_mode beta_zero_mode_)
	: a              { std::move(a_)   }
	, b              { std::move(b_)   }
	, alpha          { alpha_          }
	, beta           { beta_           }
	, beta_zero_mode { beta_zero_mode_ }
	{
		const auto& a_matrix = a.matrix();
		const auto& b_matrix = b.matrix();

		PERFLIBS_ASSERT(a_matrix.cntg() == a_matrix.strd() && a_matrix.cntg() == b_matrix.cntg(),
		             "batch problem matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == b.batch_count(),
		             "batch problem matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == b_matrix.interleave_count(),
		             "batch problem matrices have incompatible interleave counts");
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
			alpha,
			beta,
			beta_zero_mode
		};
	}
}; // struct batch_problem

template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType>
struct batch_problem<::perflibs::linalg::matmul::matmul3<AMatrixType, BMatrixType, CMatrixType, ScalarType>> {
	using problem_type  = ::perflibs::linalg::matmul::matmul3<AMatrixType, BMatrixType, CMatrixType, ScalarType>;
	using a_matrix_type = strided_batch_view<AMatrixType>;
	using b_matrix_type = strided_batch_view<BMatrixType>;
	using c_matrix_type = strided_batch_view<CMatrixType>;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	c_matrix_type c;

	scalar_type alpha;
	scalar_type beta;

	zero_mode beta_zero_mode;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_, b_matrix_type b_, c_matrix_type c_, scalar_type alpha_,
	              scalar_type beta_, zero_mode beta_zero_mode_)
	: a              { std::move(a_)              }
	, b              { std::move(b_)              }
	, c              { std::move(c_)              }
	, alpha          { alpha_                     }
	, beta           { beta_                      }
	, beta_zero_mode { beta_zero_mode_            }
	{
		const auto& a_matrix = a.matrix();
		const auto& b_matrix = b.matrix();
		const auto& c_matrix = c.matrix();

		PERFLIBS_ASSERT(a_matrix.cntg() == b_matrix.cntg() &&
		             a_matrix.strd() == c_matrix.cntg() &&
		             b_matrix.strd() == c_matrix.strd(),
		             "batch problem matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == b.batch_count() && a.batch_count() == c.batch_count(),
		             "batch problem matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == b_matrix.interleave_count() &&
		             a_matrix.interleave_count() == c_matrix.interleave_count(),
		             "batch problem matrices have incompatible interleave counts");
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
			c[index],
			alpha,
			beta,
			beta_zero_mode
		};
	}
}; // struct batch_problem

namespace matmul {

template<typename ScalarType>
using matmul2_problem = batch_problem<
	::perflibs::linalg::matmul::matmul2<
		triangular_interleave_batch_matrix<const ScalarType>,
		general_interleave_batch_matrix<ScalarType>,
		ScalarType
	>
>;

template<typename ScalarType>
using matmul3_problem = batch_problem<
	::perflibs::linalg::matmul::matmul3<
		general_interleave_batch_matrix<const ScalarType>,
		general_interleave_batch_matrix<const ScalarType>,
		general_interleave_batch_matrix<ScalarType>,
		ScalarType
	>
>;

} // namespace matmul

} // namespace perflibs::linalg::batch_interleave

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_MATMUL_PROBLEM_CONTEXT_BASES_HPP
