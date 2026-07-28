/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PACKAGE_FACTORIZATION_STRATEGIES_HPP
#define PERFLIBS_LINALG_PACKAGE_FACTORIZATION_STRATEGIES_HPP

#include "framework/compute.hpp"

#include "spec/problem_context.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "packages/factorization/strategies/lu_non_recursive.hpp"
#include "packages/factorization/strategies/lu_sequential.hpp"
#include "packages/factorization/strategies/lu_sequential_acc.hpp"
#include "packages/factorization/strategies/lu_parallel.hpp"
#include "packages/factorization/strategies/apply_pivot.hpp"
#include "packages/factorization/strategies/lu_panel_update.hpp"
#include "packages/factorization/strategies/cholesky_sequential.hpp"
#include "packages/factorization/strategies/cholesky_non_recursive.hpp"
#include "packages/factorization/strategies/cholesky_parallel.hpp"
#include "packages/factorization/strategies/qr_non_recursive.hpp"
#include "packages/factorization/strategies/qr_sequential.hpp"
#include "packages/factorization/strategies/householder_reflector.hpp"
#include "packages/factorization/strategies/apply_block_reflector.hpp"
#include "packages/factorization/strategies/qr_panel_update.hpp"
#include "packages/factorization/strategies/qr_parallel.hpp"
#include "packages/factorization/strategies/tridiagonalize_unblocked.hpp"
#include "packages/factorization/strategies/tridiagonalize_block.hpp"
#include "packages/factorization/strategies/tridiagonalize_recursive.hpp"
#include "packages/factorization/strategies/bidiagonalize_unblocked.hpp"
#include "packages/factorization/strategies/bidiagonalize_block.hpp"
#include "packages/factorization/strategies/bidiagonalize_block_parallel.hpp"
#include "packages/factorization/strategies/bidiagonalize_recursive.hpp"
#include "packages/factorization/strategies/apply_q_from_qr_unblocked.hpp"
#include "packages/factorization/strategies/apply_q_from_qr_blocked.hpp"
#include "packages/factorization/strategies/apply_q_from_qr_parallel.hpp"
#include "packages/factorization/strategies/generate_q_from_qr_unblocked.hpp"
#include "packages/factorization/strategies/generate_q_from_qr_blocked.hpp"
#include "packages/factorization/strategies/generate_q_from_qr_parallel.hpp"
#include "packages/factorization/strategies/apply_q_from_lq_unblocked.hpp"
#include "packages/factorization/strategies/apply_q_from_lq_blocked.hpp"
#include "packages/factorization/strategies/apply_q_from_lq_parallel.hpp"
#include "packages/factorization/strategies/generate_q_from_lq_unblocked.hpp"
#include "packages/factorization/strategies/generate_q_from_lq_blocked.hpp"
#include "packages/factorization/strategies/generate_q_from_lq_parallel.hpp"

#include "matrix/matrix.hpp"

namespace perflibs::linalg {

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::lu_factorization<T...>, ArchitectureSpec>> = std::tuple{
    factorization::lu_non_recursive{},
    factorization::lu_parallel{},
    factorization::lu_sequential{},
    factorization::lu_sequential_acc{},
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::lu_apply_pivot<T...>, ArchitectureSpec>> = std::tuple{
    factorization::apply_pivot{}
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::lu_panel_update<T...>, ArchitectureSpec>> = std::tuple{
    factorization::panel_update{}
};

// Cholesky
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::cholesky_factorization<T...>, ArchitectureSpec>> = std::tuple{
    factorization::cholesky_non_recursive{},
    factorization::cholesky_parallel{},
    factorization::cholesky_sequential{},
};

// QR
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::qr_factorization<T...>, ArchitectureSpec>> = std::tuple{
    factorization::qr_non_recursive{},
    factorization::qr_parallel{},
    factorization::qr_sequential{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::form_block_reflector_factor<T...>, ArchitectureSpec>> = std::tuple{
    factorization::form_block_reflector_factor_sequential{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::apply_block_reflector<T...>, ArchitectureSpec>> = std::tuple{
    factorization::apply_block_reflector_sequential{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::qr_panel_update<T...>, ArchitectureSpec>> = std::tuple{
    factorization::qr_panel_update_sequential{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::generate_reflector<T...>, ArchitectureSpec>> = std::tuple{
    factorization::generate_reflector_basic{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::tridiagonalization<T...>, ArchitectureSpec>> = std::tuple{
    factorization::tridiagonalize_unblocked{},
    factorization::tridiagonalize_recursive{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::tridiagonalization_block<T...>, ArchitectureSpec>> = std::tuple{
    factorization::tridiagonalize_block{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::apply_elementary_reflector<T...>, ArchitectureSpec>> = std::tuple{
    factorization::apply_elementary_reflector_basic{},
};
template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::bidiagonalization<T...>, ArchitectureSpec>> = std::tuple{
    factorization::bidiagonalize_unblocked{},
    factorization::bidiagonalize_recursive{},
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::bidiagonalization_block<T...>, ArchitectureSpec>> = std::tuple{
    factorization::bidiagonalize_block_parallel{},
    factorization::bidiagonalize_block{},
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::apply_q_from_qr<T...>, ArchitectureSpec>> = std::tuple{
    factorization::apply_q_from_qr_unblocked{},
    factorization::apply_q_from_qr_parallel{},
    factorization::apply_q_from_qr_blocked{},
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::generate_q_from_qr<T...>, ArchitectureSpec>> = std::tuple{
    factorization::generate_q_from_qr_unblocked{},
    factorization::generate_q_from_qr_parallel{},
    factorization::generate_q_from_qr_blocked{},
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::apply_q_from_lq<T...>, ArchitectureSpec>> = std::tuple{
    factorization::apply_q_from_lq_unblocked{},
    factorization::apply_q_from_lq_parallel{},
    factorization::apply_q_from_lq_blocked{},
};

template<typename... T, typename ArchitectureSpec>
constexpr auto strategies<spec::problem_context<factorization::generate_q_from_lq<T...>, ArchitectureSpec>> = std::tuple{
    factorization::generate_q_from_lq_unblocked{},
    factorization::generate_q_from_lq_parallel{},
    factorization::generate_q_from_lq_blocked{},
};

namespace factorization {

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<lu_factorization<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<lu_apply_pivot<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<lu_panel_update<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<cholesky_factorization<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<qr_factorization<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<qr_panel_update<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<form_block_reflector_factor<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<apply_block_reflector<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<generate_reflector<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<tridiagonalization<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<tridiagonalization_block<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<apply_elementary_reflector<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<bidiagonalization<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<bidiagonalization_block<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<apply_q_from_qr<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<generate_q_from_qr<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<apply_q_from_lq<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

template<typename... T, typename ArchitectureSpec>
__attribute__((noinline))
void compute(const spec::problem_context<generate_q_from_lq<T...>, ArchitectureSpec>& pctx) { return compute_impl(pctx); }

} //namespace perflibs::factorization

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_PACKAGE_FACTORIZATION_STRATEGIES_HPP¯
