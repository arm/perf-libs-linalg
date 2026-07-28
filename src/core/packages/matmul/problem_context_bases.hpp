/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_MATMUL_PROBLEM_CONTEXT_BASES_HPP

#include "framework/linalg_util.hpp"
#include "spec/problem_context.hpp"

#include "matrix/matrix_base.hpp"

#include <utility>

namespace perflibs::linalg::matmul {


template<
	typename AMatrixType,
	typename BMatrixType,
	typename ScalarType
>
struct matmul2 {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using scalar_type   = ScalarType;

	struct c_matrix_type { using value_type = typename b_matrix_type::value_type; };

	a_matrix_type a;
	b_matrix_type b;

	scalar_type   alpha;
	scalar_type   beta;

	zero_mode beta_zero_mode { zero_mode::set };
}; //struct matmul2

template<
	typename AMatrixType,
	typename BMatrixType,
	typename CMatrixType,
	typename ScalarType
>
struct matmul3 {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using c_matrix_type = CMatrixType;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	c_matrix_type c;

	scalar_type   alpha;
	scalar_type   beta;

	zero_mode beta_zero_mode;
}; //struct matmul3

template<
	typename AMatrixType,
	typename BMatrixType,
	typename CMatrixType,
	typename ScalarType
>
struct rank_update_2k {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using c_matrix_type = CMatrixType;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	c_matrix_type c;

	scalar_type   alpha;
	scalar_type   beta;

	zero_mode beta_zero_mode { zero_mode::set };
}; //struct rank_update_2k

template<
	typename AMatrixType,
	typename BMatrixType,
	typename CMatrixType,
	typename DMatrixType,
	typename ScalarType
>
struct matmul4 {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using c_matrix_type = CMatrixType;
	using d_matrix_type = DMatrixType;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	c_matrix_type c;
	d_matrix_type d;

	scalar_type   alpha;
	scalar_type   beta;
}; //struct matmul4

// C aliases the destination storage for A
template<
	typename AMatrixType,
	typename BMatrixType,
	typename CMatrixType,
	typename ScalarType
>
struct matadd2 {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using c_matrix_type = CMatrixType;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	c_matrix_type c;

	scalar_type   alpha;
	scalar_type   beta;
}; //struct matadd2

// C = alpha * A + beta * B
template<
	typename AMatrixType,
	typename BMatrixType,
	typename CMatrixType,
	typename ScalarType
>
struct matadd3 {
	using a_matrix_type = AMatrixType;
	using b_matrix_type = BMatrixType;
	using c_matrix_type = CMatrixType;
	using scalar_type   = ScalarType;

	a_matrix_type a;
	b_matrix_type b;
	c_matrix_type c;

	scalar_type   alpha;
	scalar_type   beta;
}; //struct matadd3

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const matmul4<Args...>& pctx) {
	return pctx.c.cntg() * pctx.c.strd() * pctx.a.cntg();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const matmul3<Args...>& pctx) {
	return pctx.c.cntg() * pctx.c.strd() * pctx.a.cntg();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const rank_update_2k<Args...>& pctx) {
	return pctx.c.cntg() * pctx.c.strd() * pctx.a.cntg();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const matadd3<Args...>& pctx) {
	return pctx.c.cntg() * pctx.c.strd();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const matadd2<Args...>& pctx) {
	return pctx.c.cntg() * pctx.c.strd();
}

template<typename... Args>
PERFLIBS_LINALG_INLINE
kernel_inttype pctx_scale(const matmul2<Args...>& pctx) {
	return pctx.b.cntg() * pctx.b.strd() * pctx.a.cntg();
}

template<typename ArchitectureSpec, typename... Args>
PERFLIBS_LINALG_INLINE
bool is_left(const spec::problem_context<matmul2<Args...>, ArchitectureSpec>& pctx) {
	return is_cntg_contig(pctx.b) && (pctx.b.cntg() <= 1 || !is_strd_contig(pctx.b));
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
constexpr bool is_left(const ProblemContext&) {
	return true;
}

} // namespace perflibs::linalg::matmul

namespace perflibs::linalg {

template<
	template<typename> class AMatrixType, typename ADataType,
	template<typename> class BMatrixType, typename BDataType,
	template<typename> class CMatrixType, typename CDataType,
	typename ScalarType,
	typename ArchitectureSpec
>
PERFLIBS_LINALG_INLINE
auto estrange(
	spec::problem_context<
		matmul::matmul3<
			AMatrixType<matrix_base<const ADataType>>,
			BMatrixType<matrix_base<const BDataType>>,
			CMatrixType<matrix_base<      CDataType>>,
			ScalarType
		>,
		ArchitectureSpec
	> pctx) {

	pctx.a.estrange_parent();
	pctx.b.estrange_parent();
	pctx.c.estrange_parent();
	return pctx;
}

}


namespace perflibs::linalg::spec {

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<matmul::matmul4<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<matmul::matmul4<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::d_matrix_type::value_type;
};

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<matmul::matmul3<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<matmul::matmul3<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::c_matrix_type::value_type;
};

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<matmul::matadd3<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<matmul::matadd3<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::c_matrix_type::value_type;
};

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<matmul::matadd2<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<matmul::matadd2<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::c_matrix_type::value_type;
};

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<matmul::rank_update_2k<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<matmul::rank_update_2k<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::c_matrix_type::value_type;
};

template<typename... T, typename ArchitectureSpec>
struct compute_precision<problem_context<matmul::matmul2<T...>, ArchitectureSpec>> {
	using problem_context_t = problem_context<matmul::matmul2<T...>, ArchitectureSpec>;
	using type              = typename problem_context_t::b_matrix_type::value_type;
};

} // /namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_MATMUL_PROBLEM_CONTEXT_BASES_HPP
