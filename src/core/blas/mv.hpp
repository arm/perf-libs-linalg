/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MV_HPP
#define PERFLIBS_LINALG_MV_HPP

#include "operators/no_op.hpp"

#include "matrix/operations.hpp"

#include "framework/which.hpp"
#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include "perflibs_assert.hpp"

namespace perflibs::linalg::exp {
namespace {

/**
 * Given: @f$y = \alpha Ax + \beta y@f$
 * Computes an axpby with a half of column of @f$A@f$ and the corresponding @f$y@f$.
 * The column of @f$A@f$ is a reflected (cntg) row in the diagonal in physical memory for speed.
 * The \c idx variable specifies the intersection point with the diagonal
 */
template<typename AxpbyKernel>
class axpby_exec {
	AxpbyKernel kernel_;
public:
	PERFLIBS_LINALG_INLINE
	axpby_exec(AxpbyKernel kernel)
	:	kernel_ { std::move(kernel) }
	{	}

	template <typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType& a, BType& b, CType& c, compute_position pos, ScalarType alpha, ScalarType beta, const Args&... args) {
		PERFLIBS_ASSERT(a.cntg() == 1, "a must be vector");
		PERFLIBS_ASSERT(a.strd_step() == 1, "a.strd_step() == 1");

		const auto alpha_x = alpha * b(0, 0);

		kernel_(a.strd(), alpha_x, a.data(), beta, c.data(), a.strd_step(), c.cntg_step());
	}
}; //class axpby_exec

/**
 * Given: @f$ y = \alpha Ax + \beta y @f$.
 * Computes an dot with a half of row of @f$A@f$ and a corresponding @f$y@f$, saved into an element of @f$y@f$.
 * The \c idx variable specifies the intersection point with the diagonal
 */
template<typename DotKernel>
class dot_exec {
	DotKernel kernel_;
public:
	PERFLIBS_LINALG_INLINE
	dot_exec(DotKernel kernel)
	:	kernel_ { std::move(kernel) }
	{	}

	template <typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType& a, BType& b, CType& c, compute_position pos, ScalarType alpha, ScalarType beta, const Args&... args) const {
		PERFLIBS_ASSERT(a.strd() == 1, "a must be vector");
		PERFLIBS_ASSERT(a.cntg() == b.cntg(), "a.cntg() != b.cntg()");
		PERFLIBS_ASSERT(c.cntg() == 1, "c.cntg() != 1");
		PERFLIBS_ASSERT(a.cntg_step() == 1, "a.cntg_step() == 1");

		auto b_mut_ptr = const_cast<ScalarType *>(b.data());

		const auto dot_result = kernel_( a.cntg(), a.data(), b_mut_ptr, a.cntg_step(), b.cntg_step());

		c(0, 0,write) = c(0,0) * beta + alpha * dot_result;
	}
}; //class dot_exec

/**
 * Given a square matrix with triangular form, we will iterate over the cols as vectors
 * the vector will then be passed to 'next' to be processed, first as a 'physical' vector,
 * then that vector will be reflect-transposed and the other next function will be called
 *
 * 1 1 1 1 1 (reflected)
 * 1 2 2 2 2
 * 1 2 3 3 3
 * 1 2 3 4 4
 * 1 2 3 4 5
 * ^
 * physical
 *
 * if the above diagram was a lower triangle symetrix matrix, first the vertical vector marked by '1'
 * would be passed to next0, then the horizontal vector marked by '1' would be.
 *
 * This is repeated for 2 then 3 ... 5
 */
template<typename Next0, typename Next1>
class mv_reflect {
	Next0 next0_;
	Next1 next1_;
public:
	PERFLIBS_LINALG_INLINE
	mv_reflect(Next0 next0, Next1 next1)
	:	next0_ { std::move(next0) }
	,	next1_ { std::move(next1) }
	{	}

	template <typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType& a, BType& b, CType& c, compute_position pos, ScalarType alpha, ScalarType beta, const Args&... args) {
		PERFLIBS_ASSERT(a.cntg() == a.strd(), "a must be square");
		PERFLIBS_ASSERT(a.cntg() == c.cntg(), "y must share dimension with a");
		PERFLIBS_ASSERT(c.strd() == 1, "c must be vector");
		PERFLIBS_ASSERT(b.strd() == 1, "b must be vector");

		if (pos.cntg != 0 || pos.iteration != 0)
			beta = one<ScalarType>;

		if(a.is_upper()) {
			for(kernel_inttype i = 0; i != a.cntg(); ++i) {
				int mod = 0;
				if constexpr(is_triangular_matrix_v<AType>) {
					if(a.is_unit()) {
						c(i,0,write) = beta * c(i,0) + alpha * a(i, i) * b(i, 0);
						mod = 1;
					}
				}
				else if constexpr(is_hermitian_matrix_v<AType>) {
					c(i,0,write) = beta * c(i,0) + alpha * a(i, i) * b(i, 0);
					mod = 1;
				}

				kernel_inttype vlen = a.strd() - i;
				{
					auto a_vec = a.sub_matrix(i, 1, i + mod, vlen - mod);
					auto b_val = get_cntg_panel(b, i, 1);
					auto c_vec = get_cntg_panel(c, i + mod, vlen - mod);

					if(!empty(a_vec))
						next0_(a_vec, b_val, c_vec, pos, alpha, beta, args...);
				}

				beta = one<ScalarType>;
				--vlen;

				//this test isn't actually needed, but doing it means we don't
				//waste time calling sub_matrix a bunch of times
				if constexpr(! is_no_op_v<Next1>) {
					if(vlen > 0) {
						auto a_vec = a.sub_matrix(i + 1, vlen, i, 1);
						auto b_vec = get_cntg_panel(b, i + 1, vlen);
						auto c_val = get_cntg_panel(c, i, 1);
						auto a_rfl = a_vec.reflect_transpose();

						next1_(a_rfl, b_vec, c_val, pos, alpha, beta, args...);
					}
				}
			}
		}
		else {
			for(kernel_inttype j = 0; j != a.cntg(); ++j) {
				kernel_inttype i    = c.cntg() - j - 1;
				kernel_inttype vlen = a.strd() - j;

				int mod = 0;
				if constexpr(is_triangular_matrix_v<AType>) {
					if(a.is_unit()) {
						c(i,0,write) = beta * c(i,0) + alpha * a(i, i) * b(i, 0);
						mod = 1;
					}
				}
				else if constexpr(is_hermitian_matrix_v<AType>) {
					c(i,0,write) = beta * c(i,0) + alpha * a(i, i) * b(i, 0);
					mod = 1;
				}

				{
					auto a_vec = a.sub_matrix(i, 1, 0, vlen - mod);
					auto b_val = get_cntg_panel(b, i, 1);
					auto c_vec = get_cntg_panel(c, 0, vlen - mod);

					if(! empty(a_vec))
						next0_(a_vec, b_val, c_vec, pos, alpha, beta, args...);
				}

				beta = one<ScalarType>;
				--vlen;

				//this test isn't actually needed, but doing it means we don't
				//waste time calling sub_matrix a bunch of times
				if constexpr(! is_no_op_v<Next1>) {
					if(vlen > 0) {
						auto a_vec = a.sub_matrix(0, vlen, i, 1);
						auto b_vec = get_cntg_panel(b, 0, vlen);
						auto c_val = get_cntg_panel(c, i, 1);
						auto a_rfl = a_vec.reflect_transpose();

						next1_(a_rfl, b_vec, c_val, pos, alpha, beta, args...);
					}
				}
			}
		}
	}
}; //triangular_above_below

/**
 * Given two operations, takes these and performs them on the 'square' and 'panel' sections of a triangular
 * matrix. The 'square' is a sub-matrix whose diagonal coincides with the diagonal of the original matrix. The
 * 'panel' spans the physical half of the matrix.
 *
 * E.g.
 *
 *     |.....|-----------|
 *     |xx...|-----------|
 *     |xxxx.|-----------|
 *           |.....|-----|        ----    where `next_panel_` is performed
 *           |xx...|-----|        xxxx    are virtual entries in the matrix
 *           |xxxx.|-----|        ....    is where `next_square_` is performed
 *                 |.....|
 *                 |xx...|        (diagram originally from `mv_common.hpp`)
 *                 |xxxx.|
 *
 * This operator also intercepts the value of `beta` and sets it to one for the subsequent square and panel
 * operators. This is required by the implementation of the `symv` routine -- we set `beta` to one here
 * because the scaling step is performed in the reduction, not the main workload.
 */
template<typename NextSquare, typename NextPanel>
class symv_square_panel {
	which_dimension dimension_;
	/**
	 * The operation to perform on the square part of the matrix. See the doxygen comment for the
	 * `symv_square_panel` class.
	 */
	NextSquare next_square_;

	/**
	 * The operation to perform on the panel part of the matrix. See the doxygen comment for the
	 * `symv_square_panel` class.
	 */
	NextPanel next_panel_;

public:
	symv_square_panel(which_dimension dimension, NextSquare next_square, NextPanel next_panel)
	:	dimension_   { dimension }
	,	next_square_ { std::move(next_square) }
	,	next_panel_  { std::move(next_panel) }
	{	}

	/**
	 * Perform the square and panel operation.
	 *
	 * @tparam AType The type of the matrix `a`.
	 * @tparam BType The type of the matrix `b`.
	 * @tparam CType The type of the matrix `c`.
	 * @param a [in,out] The `a` matrix. Submatrices of `a` (the square and panel) are passed to the next
	 *                   operators which may modify the referenced object.
	 * @param b [in,out] The `b` matrix. This matrix is passed to the next operators which may modify the
	 *                   referenced object.
	 * @param c [in,out] The `c` matrix. Submatrices of `c` (the square and panel) are passed to the next
	 *                   operators which may modify the referenced object.
	 * @param pos [in] The compute position.
	 * @param alpha [in] The `alpha` variable in the BLAS problem. Passed to the next operators.
	 */
	template<typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	void operator()(AType &a, BType &b, CType &c, compute_position pos, ScalarType alpha, Args... args) {
		static_assert(is_triangular_form_v<AType>);

		if(dimension_ == which_dimension::a_strd) {
			/*
			 * First, find the square part of the matrices `a` and `c` and perform the `next_square_`
			 * operator on them.
			 */
			const auto [strd_first0, strd_last0] = get_non_virtual_strd_bounds_for_cntg(a, 0);
			const auto square_start_index = a.is_upper() ? strd_first0 : strd_last0 - 1;
			const auto square_size = a.cntg();

			auto square_a = get_strd_panel(a, square_start_index, square_size);
			auto square_c = get_cntg_panel(c, square_start_index, square_size);

			PERFLIBS_ASSERT(square_a.cntg() == square_a.strd(), "`square_a`'s dimensions are non-square.");

			next_square_(square_a, b, square_c, advance(pos, square_start_index, 0, 0), alpha, args...);

			/*
			 * Then, find the panel part of the submatrices `a` and `c` and perform the `next_panel_`
			 * operator on them. Note that it _is_ possible for the panel to be empty in the case when the
			 * entire matrix `a` is a square. In that case, we do nothing.
			 */
			const auto panel_start_index = a.is_upper()
			                             ? square_start_index + square_size
			                             : 0;

			const auto panel_size = a.is_upper()
			                      ? a.strd() - square_start_index - square_size
			                      : square_start_index;
			auto panel_a = get_strd_panel(a, panel_start_index, panel_size);

			if(!empty(panel_a)) {
				auto panel_c = get_cntg_panel(c, panel_start_index, panel_size);
				next_panel_(panel_a, b, panel_c, advance(pos, panel_start_index, 0, 0), alpha, args...);
			}
		}
		/**
		 * Same as above, but this time we are separating along the a.cntg dimension
		 */
		else if(dimension_ == which_dimension::cntg) {
			/*
			 * First, find the square part of the matrices `a` and `b` and perform the `next_square_`
			 * operator on them.
			 */
			const auto [cntg_first0, cntg_last0] = get_non_virtual_cntg_bounds_for_strd(a, 0);

			const auto square_start_index = a.is_lower()
			                              ? cntg_first0
			                              : cntg_last0 - 1;

			const auto square_size = a.strd();

			auto square_a = get_cntg_panel(a, square_start_index, square_size);
			auto square_b = get_cntg_panel(b, square_start_index, square_size);

			PERFLIBS_ASSERT(square_a.cntg() == square_a.strd(), "`square_a`'s dimensions are non-square.");

			//next_square_(square_a, square_b, c, advance(pos, 0, 0, square_start_index), alpha, args...);

			/* Then, find the panel part of the submatrices `a` and `b` and perform the `next_panel_`
			 * operator on them. Note that it _is_ possible for the panel to be empty in the case when the
			 * entire matrix `a` is a square. In that case, we do nothing.
			 */
			const auto panel_start_index = a.is_lower()
			                             ? square_start_index + square_size
                                         : 0;

			const auto panel_size = a.is_lower()
			                      ? a.cntg() - square_start_index - square_size
                                  : square_start_index;

			auto panel_a = get_cntg_panel(a, panel_start_index, panel_size);
			auto panel_b = get_cntg_panel(b, panel_start_index, panel_size);

			next_square_(square_a, square_b, c, advance(pos, 0, 0, 0, 0), alpha, args...);
			if(!empty(panel_a))
				next_panel_(panel_a, panel_b, c, advance(pos, 0, 0, 0, 1), alpha, args...);
		}
	}
}; // symv_square_panel


template<typename Next>
class trmv_vector_split {
	Next next_;
public:
	trmv_vector_split(Next next)
	:	next_ { std::move(next) }
	{	}

	template<typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(AType &a, BType &b, CType &c, compute_position pos, ScalarType alpha, ScalarType beta, Args... args)const {
		if(a.is_lower()) {
			for(kernel_inttype i = 0; i!=a.strd(); ++i) {
				auto beta2 = beta;

				int mod = 0;

				if constexpr(is_triangular_matrix_v<AType>) {
					if(a.is_unit()) {
						c(i,0,write) = beta * c(i,0) + alpha * a(i, i) * b(i, 0);
						mod = 1;
						beta2 = 1.0;
					}
				}

				const auto vlen = a.cntg() - i - mod;

				auto a_v = a.sub_matrix(i + mod, vlen, i, 1);
				auto b_v = get_cntg_panel(b, i + mod, vlen);
				auto c_v = get_cntg_panel(c, i, 1);

				if(!empty(a_v))
					next_(a_v, b_v, c_v, pos, alpha, beta2, args...);
			}
		}
		else {
			for(kernel_inttype i = 0; i!=a.strd(); ++i) {
				auto beta2 = beta;

				int mod = 1;

				if constexpr(is_triangular_matrix_v<AType>) {
					if(a.is_unit()) {
						c(i,0,write) = beta * c(i,0) + alpha * a(i, i) * b(i, 0);
						mod = 0;
						beta2 = 1.0;
					}
				}

				const auto vlen = i + mod;

				auto a_v = a.sub_matrix(0, vlen, i, 1);
				auto b_v = get_cntg_panel(b, 0, vlen);
				auto c_v = get_cntg_panel(c, i, 1);

				if(!empty(a_v))
					next_(a_v, b_v, c_v, pos, alpha, beta2, args...);
			}
		}
	}
}; //class trmv_vector_split

} // namespace anon
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MV_HPP
