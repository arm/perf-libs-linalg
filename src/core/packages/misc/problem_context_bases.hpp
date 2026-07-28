/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_PROBLEM_CONTEXT_BASES_HPP
#define PERFLIBS_LINALG_MISC_PROBLEM_CONTEXT_BASES_HPP

#include <utility>

namespace perflibs::linalg::misc {

template<
    typename VectorType,
	typename ScalarType
>
struct rot {
    using vector_type = VectorType;
	using scalar_type = ScalarType;

    kernel_inttype n;
	vector_type *x;
	kernel_inttype incx;
	vector_type *y;
	kernel_inttype incy;
	const remove_complex_t<promote_t<vector_type, scalar_type>> c;
    scalar_type s;

    rot(kernel_inttype n, vector_type *x, kernel_inttype incx,
		vector_type *y, kernel_inttype incy,
		const remove_complex_t<promote_t<vector_type, scalar_type>> c,
		scalar_type s)
	:   n        { std::move(n) }
	,   x        { std::move(x) }
	,   incx     { std::move(incx) }
	,   y        { std::move(y) }
	,   incy     { std::move(incy) }
	,   c        { std::move(c) }
	,   s        { std::move(s) }
	{	}
}; // struct rot


template<
	typename ScalarType
>
struct rotg {
	using scalar_type   = ScalarType;

    scalar_type *a;
	scalar_type *b;
	perflibs::remove_complex_t<scalar_type> *c;
	scalar_type *s;

    rotg(scalar_type *a,
		scalar_type *b,
		perflibs::remove_complex_t<scalar_type> *c,
		scalar_type *s)
	:   a        { std::move(a) }
	,   b        { std::move(b) }
	,   c        { std::move(c) }
	,   s        { std::move(s) }
	{	}
}; // struct rotg


template<
	typename ScalarType
>
struct rotm {
	using scalar_type = ScalarType;

	kernel_inttype n;
	scalar_type *x;
	kernel_inttype incx;
	scalar_type *y;
	kernel_inttype incy;
	const scalar_type *param;

    rotm(kernel_inttype n,
		scalar_type *x,
		kernel_inttype incx,
		scalar_type *y,
		kernel_inttype incy,
		const scalar_type *param)
	:   n        { std::move(n) }
	,   x        { std::move(x) }
	,   incx     { std::move(incx) }
	,   y        { std::move(y) }
	,   incy     { std::move(incy) }
	,   param    { std::move(param) }
	{	}
}; // struct rot


template<
	typename ScalarType
>
struct rotmg {
	using scalar_type = ScalarType;

	scalar_type *d1;
	scalar_type *d2;
	scalar_type *x;
	const scalar_type *y;
	scalar_type *param;

    rotmg(scalar_type *d1,
		scalar_type *d2,
		scalar_type *x,
		const scalar_type *y,
		scalar_type *param)
	:   d1       { std::move(d1) }
	,   d2       { std::move(d2) }
	,   x        { std::move(x) }
	,   y        { std::move(y) }
	,   param    { std::move(param) }
	{	}
}; // struct rotmg


enum class find_operation{ absolute_max, absolute_min };

template<typename ScalarType>
struct find_index {
	using scalar_type   = ScalarType;
	using a_matrix_type = general_matrix<matrix_base<const ScalarType>>;

	a_matrix_type   x;
	kernel_inttype& out;
	find_operation  operation;

	find_index(general_matrix <matrix_base<const ScalarType>> x, kernel_inttype& out, find_operation operation)
	:	x          { std::move(x) }
	,	out        { out }
	,	operation  { std::move(operation) }
	{	}
}; // struct find_index


template<typename ScalarType>
struct l1_norm {
	using scalar_type   = ScalarType;
	using real_type     = remove_complex_t<ScalarType>;
	using a_matrix_type = general_matrix<matrix_base<const ScalarType>>;

	a_matrix_type x;
	real_type&    out;

	l1_norm(a_matrix_type x, real_type& out)
	:	x   { std::move(x) }
	,	out { out }
	{	}
}; // struct l1_norm

template<typename ScalarType>
struct l2_norm {
	using scalar_type   = ScalarType;
	using real_type     = remove_complex_t<ScalarType>;
	using a_matrix_type = general_matrix<matrix_base<const ScalarType>>;

	a_matrix_type x;
	real_type&    out;

	l2_norm(a_matrix_type x, real_type& out)
	:	x   { std::move(x) }
	,	out { out }
	{	}
}; // struct l2_norm


template<typename ScalarType>
struct swap {
	using scalar_type   = ScalarType;
	using a_matrix_type = general_matrix<matrix_base<ScalarType>>;

	a_matrix_type x, y;

	swap(a_matrix_type x, a_matrix_type y)
	:	x   { std::move(x) }
	,       y   { std::move(y) }
	{	}
}; // struct swap

template<typename VectorType, typename ScalarType>
PERFLIBS_LINALG_INLINE kernel_inttype pctx_scale(const rot<VectorType, ScalarType>& pctx) {
	return pctx.n;
}

template<typename ScalarType>
PERFLIBS_LINALG_INLINE kernel_inttype pctx_scale(const find_index<ScalarType>& pctx) {
	return pctx.x.cntg();
}

template<typename ScalarType>
PERFLIBS_LINALG_INLINE kernel_inttype pctx_scale(const l1_norm<ScalarType>& pctx) {
	return pctx.x.cntg();
}

template<typename ScalarType>
PERFLIBS_LINALG_INLINE kernel_inttype pctx_scale(const l2_norm<ScalarType>& pctx) {
	return pctx.x.cntg();
}

template<typename ScalarType>
PERFLIBS_LINALG_INLINE kernel_inttype pctx_scale(const swap<ScalarType>& pctx) {
	return pctx.x.cntg();
}

}  //namespace perflibs::linalg::misc

#endif //PERFLIBS_LINALG_MISC_PROBLEM_CONTEXT_BASES_HPP
