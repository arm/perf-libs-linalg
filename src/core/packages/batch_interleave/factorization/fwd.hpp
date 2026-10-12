/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_FWD_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_FWD_HPP

#include "packages/batch_interleave/factorization/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/compute.hpp"

// DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg::batch_interleave::factorization {

class batch_interleave_cholesky_factorization_generic;
class batch_interleave_cholesky_left_looking;
class batch_interleave_generate_q_from_qr_generic;
class batch_interleave_apply_q_from_qr_generic;
class batch_interleave_apply_elementary_reflector_generic;
class batch_interleave_pivoted_qr_factorization_generic;
class batch_interleave_threshold_pivoted_lu_factorization_generic;

template<typename T, typename ArchitectureSpec>
void compute(const spec::problem_context<cholesky_problem<T>, ArchitectureSpec>& pctx);

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<generate_q_from_qr_problem<T, IntType>, ArchitectureSpec>& pctx);

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_q_from_qr_problem<T, IntType>, ArchitectureSpec>& pctx);

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<threshold_pivoted_lu_problem<T, IntType>, ArchitectureSpec>& pctx);

template<typename T, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_elementary_reflector_problem<T>, ArchitectureSpec>& pctx);

template<typename T, typename IntType, typename ArchitectureSpec>
void compute(const spec::problem_context<pivoted_qr_problem<T, IntType>, ArchitectureSpec>& pctx);

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_FWD_HPP
