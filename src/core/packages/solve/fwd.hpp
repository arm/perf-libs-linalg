/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_FWD_HPP
#define PERFLIBS_LINALG_SOLVE_FWD_HPP

#include "packages/solve/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/compute.hpp"
#include "framework/axpby_kernels.hpp"

//DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg {

namespace solve {

class triangular_solve_vector_reference;
class triangular_solve_vector;
class triangular_solve_matrix_reference;
class triangular_solve_matrix_small;
class triangular_solve_matrix_large;

class compressed_triangular_solve_vector;

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<solve<T...>, ArchitectureSpec>&);

} //namespace solve

template<>
struct axpby_kernel_tag<spec::strategy_tag<solve::triangular_solve_vector>> {

	axpby_kernel_tag(const spec::strategy_tag<solve::triangular_solve_vector>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.b.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return typename ProblemContext::scalar_type { 1.1 };
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::b_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<solve::triangular_solve_vector>>

template<>
struct axpby_kernel_tag<spec::strategy_tag<solve::compressed_triangular_solve_vector>> {
	axpby_kernel_tag(const spec::strategy_tag<solve::compressed_triangular_solve_vector>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return 1_ki;
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.b.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return typename ProblemContext::scalar_type { 1.1 };
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::b_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<solve::compressed_triangular_solve_vector>

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_SOLVE_FWD_HPP
