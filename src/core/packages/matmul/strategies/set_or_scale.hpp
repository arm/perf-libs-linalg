/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_SET_SCALE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_SET_SCALE_HPP

#include "spec/problem_context_helpers.hpp"
#include "spec/problem_context.hpp"

#include "matrix/operations.hpp"
#include "blas/rank_update_strategies/helpers.hpp"
#include "framework/linalg_util.hpp"

#include "packages/matmul/problem_context_bases.hpp"

#include <string_view>

/*
 * When the scalars and dimensions of the matrix
 * mean there is actually no need to compute dot products
 * out of A & B, then we can get away with just scaling or setting
 * the values of the output matrix C
 */
namespace perflibs::linalg::matmul {

class set_or_scale {
public:
	static constexpr std::string_view name() { return "set_or_scale"; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		if(pctx.a.strd() == 0) {
			return true;
		}

		if(pctx.beta == zero<>) {
			using c_data_type = typename ProblemContext::c_matrix_type::value_type;
			set(zero<c_data_type>, pctx.c);
		}
		else if(pctx.beta != one<>) {
			scale(pctx.beta, pctx.c);
			if constexpr(is_hermitian_matrix_v<decltype(pctx.c)>) {
				zero_diag_imag(pctx.c);
			}
		}

		return true;
	}

	// Operator for matmul2 routines
	template<
		typename ADataType,
		typename BDataType,
		typename ScalarType,
	    typename ArchitectureSpec
	>
	PERFLIBS_LINALG_INLINE
	bool operator() (
		const spec::problem_context<matmul2<
	    	triangular_matrix<    matrix_base<const ADataType>>,
			general_matrix<       matrix_base<      BDataType>>,
			ScalarType>,
			ArchitectureSpec>& pctx) const {

		if( !this->can_compute(pctx) ) return false;

		if(pctx.a.strd() == 0) {
			return true;
		}

		if(pctx.beta == zero<decltype(pctx.beta)>) {
			set(0.0, pctx.b);
		}
		else if(pctx.beta != one<decltype(pctx.beta)>) {
			scale(pctx.beta, pctx.b);
		}

		return true;
	}


	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const spec::problem_context<matmul4<Ts...>, ArchitectureSpec>& pctx) const {
		return false;
	}

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const spec::problem_context<matadd3<Ts...>, ArchitectureSpec>&) const {
		return false;
	}

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const spec::problem_context<matadd2<Ts...>, ArchitectureSpec>&) const {
		return false;
	}

	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (
		const spec::problem_context<matmul2<
			triangular_matrix<packed_matrix_base<const DataType>>,
			general_matrix   <matrix_base       <      DataType>>,
			DataType>,
			ArchitectureSpec>& pctx) const {

		return false;
	}


	template<typename DataType, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (
		const spec::problem_context<matmul2<
			triangular_matrix<banded_matrix_base<const DataType>>,
			general_matrix   <matrix_base       <      DataType>>,
			DataType>,
			ArchitectureSpec>& pctx) const {

		return false;
	}




	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<matmul4<Ts...>, ArchitectureSpec>& pctx) const {
		return false;
	}

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<matadd3<Ts...>, ArchitectureSpec>&) const {
		return false;
	}

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<matadd2<Ts...>, ArchitectureSpec>&) const {
		return false;
	}


	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return ( pctx.a.cntg() == 0 || pctx.alpha == zero<> ) && pctx.beta_zero_mode == zero_mode::set;
	}
}; //class set_or_scale

} // perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_SET_SCALE_HPP
