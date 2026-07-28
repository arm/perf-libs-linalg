/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_FWD_HPP
#define PERFLIBS_LINALG_FACTORIZATION_FWD_HPP

#include "packages/factorization/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/compute.hpp"

// DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg::factorization  {

class lu_factorization_generic;
class qr_factorization_generic;
class cholesky_factorization_generic;
class bidiagonalization_generic;
class bidiagonalization_block_generic;
class tridiagonalization_generic;
class tridiagonalization_block_generic;
class apply_q_from_qr_generic;
class generate_q_from_qr_generic;
class apply_q_from_lq_generic;
class generate_q_from_lq_generic;

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<lu_factorization<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<lu_apply_pivot<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<lu_panel_update<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<cholesky_factorization<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<qr_factorization<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<qr_panel_update<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<form_block_reflector_factor<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_block_reflector<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<generate_reflector<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<tridiagonalization<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<tridiagonalization_block<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_elementary_reflector<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<bidiagonalization<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<bidiagonalization_block<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_q_from_qr<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<generate_q_from_qr<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<apply_q_from_lq<T...>, ArchitectureSpec>&);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<generate_q_from_lq<T...>, ArchitectureSpec>&);

} // namespace perflibs::linalg::factorization

#endif // PERFLIBS_LINALG_FACTORIZATION_FWD_HPP
