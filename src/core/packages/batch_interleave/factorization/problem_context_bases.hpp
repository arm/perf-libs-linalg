/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_PROBLEM_CONTEXT_BASES_HPP

#include "framework/linalg_util.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/strided_batch_view.hpp"
#include "packages/batch_interleave/batch_problem.hpp"

#include <algorithm>
#include <utility>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * One interleaved Cholesky factorization problem in an outer batch.
 */
template<typename AMatrixType>
struct cholesky_factorization {
	AMatrixType a;
}; // struct cholesky_factorization
/**
 * Form Q matrices from batches of interleaved QR factorizations.
 */
template<typename AMatrixType, typename ReflectorCountMatrixType, typename TauMatrixType>
struct generate_q_from_qr {
	AMatrixType              a;
	ReflectorCountMatrixType reflector_count;
	TauMatrixType            tau;
}; // struct generate_q_from_qr

/**
 * Apply Q or Q^T from interleaved QR factorizations to a matrix batch from the left.
 * transpose_q selects Q^T when true and Q when false.
 */
template<typename AMatrixType, typename ReflectorCountMatrixType,
         typename TauMatrixType, typename CMatrixType>
struct apply_q_from_qr {
	bool                     transpose_q;
	AMatrixType              a;
	ReflectorCountMatrixType reflector_count;
	TauMatrixType            tau;
	CMatrixType              c;
}; // struct apply_q_from_qr

/**
 * Apply one elementary reflector from the left to a matrix of interleaved problems.
 *
 * Each interleaved lane has its own reflector and scalar factor. The leading
 * element of every reflector is implicitly one and is not read from v.
 */
template<typename VMatrixType, typename TauMatrixType, typename CMatrixType>
struct apply_elementary_reflector {
	using c_matrix_type = CMatrixType;

	VMatrixType   v;
	TauMatrixType tau;
	CMatrixType   c;

	PERFLIBS_LINALG_INLINE
	kernel_inttype interleave_count() const {
		return c.interleave_count();
	}
}; // struct apply_elementary_reflector

/**
 * One interleaved LU factorization problem with threshold partial pivoting.
 */
template<typename AMatrixType, typename PivotMatrixType, typename T>
struct threshold_pivoted_lu_factorization {
	AMatrixType     a;
	PivotMatrixType pivot;
	T               threshold;
}; // struct threshold_pivoted_lu_factorization

/**
 * One interleaved problem in a pivoted QR factorization outer batch.
 */
template<typename AMatrixType, typename PivotMatrixType, typename TauMatrixType, typename RankMatrixType>
struct pivoted_qr_factorization {
	AMatrixType     a;
	PivotMatrixType pivot;
	TauMatrixType   tau;
	RankMatrixType  rank;
}; // struct pivoted_qr_factorization

} // namespace perflibs::linalg::batch_interleave::factorization

namespace perflibs::linalg::batch_interleave {

template<typename AMatrixType>
struct batch_problem<factorization::cholesky_factorization<AMatrixType>> {
	using problem_type  = factorization::cholesky_factorization<AMatrixType>;
	using a_matrix_type = strided_batch_view<AMatrixType>;

	a_matrix_type a;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_)
	: a { std::move(a_) }
	{
		const auto& a_matrix = a.matrix();
		PERFLIBS_ASSERT(a_matrix.cntg() == a_matrix.strd(),
		             "Cholesky factorization requires square matrices");
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
		return { a[index] };
	}
}; // struct batch_problem

template<typename AMatrixType, typename ReflectorCountMatrixType, typename TauMatrixType>
struct batch_problem<factorization::generate_q_from_qr<
	AMatrixType, ReflectorCountMatrixType, TauMatrixType>> {
	using problem_type = factorization::generate_q_from_qr<
		AMatrixType, ReflectorCountMatrixType, TauMatrixType
	>;
	using a_matrix_type               = strided_batch_view<AMatrixType>;
	using reflector_count_matrix_type = strided_batch_view<ReflectorCountMatrixType>;
	using tau_matrix_type             = strided_batch_view<TauMatrixType>;

	a_matrix_type               a;
	reflector_count_matrix_type reflector_count;
	tau_matrix_type             tau;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_, reflector_count_matrix_type reflector_count_,
	              tau_matrix_type tau_)
	: a               { std::move(a_)               }
	, reflector_count { std::move(reflector_count_) }
	, tau             { std::move(tau_)             }
	{
		const auto& a_matrix               = a.matrix();
		const auto& reflector_count_matrix = reflector_count.matrix();
		const auto& tau_matrix             = tau.matrix();

		PERFLIBS_ASSERT(a_matrix.strd() <= a_matrix.cntg()
		          && reflector_count_matrix.cntg() == 1
		          && reflector_count_matrix.strd() == 1
		          && tau_matrix.cntg() == a_matrix.strd()
		          && tau_matrix.strd() == 1,
		             "batch problem matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == reflector_count.batch_count()
		          && a.batch_count() == tau.batch_count(),
		             "batch problem matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == reflector_count_matrix.interleave_count()
		          && a_matrix.interleave_count() == tau_matrix.interleave_count(),
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
			reflector_count[index],
			tau[index]
		};
	}
}; // struct batch_problem

template<typename AMatrixType, typename ReflectorCountMatrixType,
         typename TauMatrixType, typename CMatrixType>
struct batch_problem<factorization::apply_q_from_qr<
	AMatrixType, ReflectorCountMatrixType, TauMatrixType, CMatrixType>> {
	using problem_type = factorization::apply_q_from_qr<
		AMatrixType, ReflectorCountMatrixType, TauMatrixType, CMatrixType
	>;
	using a_matrix_type               = strided_batch_view<AMatrixType>;
	using reflector_count_matrix_type = strided_batch_view<ReflectorCountMatrixType>;
	using tau_matrix_type             = strided_batch_view<TauMatrixType>;
	using c_matrix_type               = strided_batch_view<CMatrixType>;

	bool                        transpose_q;
	a_matrix_type               a;
	reflector_count_matrix_type reflector_count;
	tau_matrix_type             tau;
	c_matrix_type               c;

	PERFLIBS_LINALG_INLINE
	batch_problem(bool transpose_q_, a_matrix_type a_,
	              reflector_count_matrix_type reflector_count_, tau_matrix_type tau_,
	              c_matrix_type c_)
	: transpose_q     { transpose_q_                }
	, a               { std::move(a_)               }
	, reflector_count { std::move(reflector_count_) }
	, tau             { std::move(tau_)             }
	, c               { std::move(c_)               }
	{
		const auto& a_matrix               = a.matrix();
		const auto& reflector_count_matrix = reflector_count.matrix();
		const auto& tau_matrix             = tau.matrix();
		const auto& c_matrix               = c.matrix();
		const auto q_order = c_matrix.cntg();

		PERFLIBS_ASSERT(a_matrix.cntg() == q_order
		          && a_matrix.strd() == q_order
		          && reflector_count_matrix.cntg() == 1
		          && reflector_count_matrix.strd() == 1
		          && tau_matrix.cntg() == q_order
		          && tau_matrix.strd() == 1,
		             "batch problem matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == reflector_count.batch_count()
		          && a.batch_count() == tau.batch_count()
		          && a.batch_count() == c.batch_count(),
		             "batch problem matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == reflector_count_matrix.interleave_count()
		          && a_matrix.interleave_count() == tau_matrix.interleave_count()
		          && a_matrix.interleave_count() == c_matrix.interleave_count(),
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
			transpose_q,
			a[index],
			reflector_count[index],
			tau[index],
			c[index]
		};
	}
}; // struct batch_problem

template<typename AMatrixType, typename PivotMatrixType, typename T>
struct batch_problem<factorization::threshold_pivoted_lu_factorization<
	AMatrixType, PivotMatrixType, T>> {
	using problem_type = factorization::threshold_pivoted_lu_factorization<
		AMatrixType, PivotMatrixType, T
	>;
	using a_matrix_type     = strided_batch_view<AMatrixType>;
	using pivot_matrix_type = strided_batch_view<PivotMatrixType>;

	a_matrix_type     a;
	pivot_matrix_type pivot;
	T                 threshold;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_, pivot_matrix_type pivot_, T threshold_)
	: a         { std::move(a_)     }
	, pivot     { std::move(pivot_) }
	, threshold { threshold_        }
	{
		const auto& a_matrix     = a.matrix();
		const auto& pivot_matrix = pivot.matrix();

		PERFLIBS_ASSERT(pivot_matrix.cntg() == std::min(a_matrix.cntg(), a_matrix.strd())
		          && pivot_matrix.strd() == 1,
		             "batch problem matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == pivot.batch_count(),
		             "batch problem matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == pivot_matrix.interleave_count(),
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
			pivot[index],
			threshold
		};
	}
}; // struct batch_problem

template<typename AMatrixType, typename PivotMatrixType, typename TauMatrixType, typename RankMatrixType>
struct batch_problem<factorization::pivoted_qr_factorization<
	AMatrixType, PivotMatrixType, TauMatrixType, RankMatrixType>> {
	using problem_type = factorization::pivoted_qr_factorization<
		AMatrixType, PivotMatrixType, TauMatrixType, RankMatrixType
	>;
	using a_matrix_type     = strided_batch_view<AMatrixType>;
	using pivot_matrix_type = strided_batch_view<PivotMatrixType>;
	using tau_matrix_type   = strided_batch_view<TauMatrixType>;
	using rank_matrix_type  = strided_batch_view<RankMatrixType>;

	a_matrix_type     a;
	pivot_matrix_type pivot;
	tau_matrix_type   tau;
	rank_matrix_type  rank;

	PERFLIBS_LINALG_INLINE
	batch_problem(a_matrix_type a_, pivot_matrix_type pivot_, tau_matrix_type tau_,
	              rank_matrix_type rank_)
	: a     { std::move(a_)     }
	, pivot { std::move(pivot_) }
	, tau   { std::move(tau_)   }
	, rank  { std::move(rank_)  }
	{
		const auto& a_matrix     = a.matrix();
		const auto& pivot_matrix = pivot.matrix();
		const auto& tau_matrix   = tau.matrix();
		const auto& rank_matrix  = rank.matrix();

		PERFLIBS_ASSERT(a_matrix.strd() == pivot_matrix.cntg()
		          && pivot_matrix.strd() == 1
		          && tau_matrix.cntg() == std::min(a_matrix.cntg(), a_matrix.strd())
		          && tau_matrix.strd() == 1
		          && rank_matrix.cntg() == 1
		          && rank_matrix.strd() == 1,
		             "batch problem matrices have incompatible dimensions");
		PERFLIBS_ASSERT(a.batch_count() == pivot.batch_count()
		          && a.batch_count() == tau.batch_count()
		          && a.batch_count() == rank.batch_count(),
		             "batch problem matrices have different batch counts");
		PERFLIBS_ASSERT(a_matrix.interleave_count() == pivot_matrix.interleave_count()
		          && a_matrix.interleave_count() == tau_matrix.interleave_count()
		          && a_matrix.interleave_count() == rank_matrix.interleave_count(),
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
			pivot[index],
			tau[index],
			rank[index]
		};
	}
}; // struct batch_problem

namespace factorization {

template<typename T>
using cholesky_problem = batch_problem<
	cholesky_factorization<symmetric_interleave_batch_matrix<T>>
>;

template<typename T, typename IntType>
using apply_q_from_qr_problem =
	batch_problem<
		apply_q_from_qr<
			general_interleave_batch_matrix<const T>,
			general_interleave_batch_matrix<const IntType>,
			general_interleave_batch_matrix<const T>,
			general_interleave_batch_matrix<T>
		>
	>;

template<typename T, typename IntType>
using generate_q_from_qr_problem =
	batch_problem<
		generate_q_from_qr<
			general_interleave_batch_matrix<T>,
			general_interleave_batch_matrix<const IntType>,
			general_interleave_batch_matrix<const T>
		>
	>;

template<typename T>
using apply_elementary_reflector_problem =
	apply_elementary_reflector<
		general_interleave_batch_matrix<const T>,
		general_interleave_batch_matrix<const T>,
		general_interleave_batch_matrix<T>
	>;

template<typename T, typename IntType>
using threshold_pivoted_lu_problem =
	batch_problem<
		threshold_pivoted_lu_factorization<
			general_interleave_batch_matrix<T>,
			general_interleave_batch_matrix<IntType>,
			T
		>
	>;

template<typename T, typename IntType>
using pivoted_qr_problem =
	batch_problem<
		pivoted_qr_factorization<
			general_interleave_batch_matrix<T>,
			general_interleave_batch_matrix<IntType>,
			general_interleave_batch_matrix<T>,
			general_interleave_batch_matrix<IntType>
		>
	>;
} // namespace factorization

} // namespace perflibs::linalg::batch_interleave

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_PROBLEM_CONTEXT_BASES_HPP
