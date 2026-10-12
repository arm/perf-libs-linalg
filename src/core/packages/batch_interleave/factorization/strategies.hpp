/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_HPP

#include "framework/compute.hpp"

#include "packages/batch_interleave/factorization/problem_context_bases.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_cholesky_factorization_generic.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_cholesky_left_looking.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_generate_q_from_qr_generic.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_apply_q_from_qr_generic.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_apply_elementary_reflector_generic.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_pivoted_qr_factorization_generic.hpp"
#include "packages/batch_interleave/factorization/strategies/batch_interleave_threshold_pivoted_lu_factorization_generic.hpp"

namespace perflibs::linalg {

template<typename T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<batch_interleave::factorization::cholesky_problem<T>,ArchitectureSpec>> = std::tuple {
	batch_interleave::factorization::batch_interleave_cholesky_factorization_generic{},
	batch_interleave::factorization::batch_interleave_cholesky_left_looking{}
};

template<typename T, typename IntType, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<batch_interleave::factorization::generate_q_from_qr_problem<T, IntType>, ArchitectureSpec>> = std::tuple {
	batch_interleave::factorization::batch_interleave_generate_q_from_qr_generic{}
};

template<typename T, typename IntType, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<batch_interleave::factorization::apply_q_from_qr_problem<T, IntType>, ArchitectureSpec>> = std::tuple {
	batch_interleave::factorization::batch_interleave_apply_q_from_qr_generic{}
};

template<typename T, typename IntType, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<batch_interleave::factorization::threshold_pivoted_lu_problem<T, IntType>, ArchitectureSpec>> = std::tuple {
	batch_interleave::factorization::batch_interleave_threshold_pivoted_lu_factorization_generic{}
};
template<typename T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<batch_interleave::factorization::apply_elementary_reflector_problem<T>, ArchitectureSpec>> = std::tuple {
	batch_interleave::factorization::batch_interleave_apply_elementary_reflector_generic{}
};

template<typename T, typename IntType, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<batch_interleave::factorization::pivoted_qr_problem<T, IntType>, ArchitectureSpec>> = std::tuple {
	batch_interleave::factorization::batch_interleave_pivoted_qr_factorization_generic{}
};

namespace batch_interleave::factorization {

template<typename T, typename ArchitectureSpec>
void compute(const spec::problem_context<cholesky_problem<T>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<generate_q_from_qr_problem<T, IntType>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_q_from_qr_problem<T, IntType>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<threshold_pivoted_lu_problem<T, IntType>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

template<typename T, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_elementary_reflector_problem<T>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<pivoted_qr_problem<T, IntType>, ArchitectureSpec>& pctx) {
	return compute_impl(pctx);
}

} // namespace batch_interleave::factorization

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_PACKAGE_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_HPP
