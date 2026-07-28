/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_GEMV_FALLBACK_KERNELS_HPP
#define PERFLIBS_LINALG_GEMV_FALLBACK_KERNELS_HPP

#include "framework/linalg_util.hpp"
#include "perflibs_complex.hpp"

#include <array>

namespace perflibs::linalg {

/**
 * @tparam UnrollCntg number of elements to unroll along the cntg dimension
 * @tparam UnrollStrd number of elements to unroll along the strd dimension
 */
template<
	kernel_inttype UnrollCntg, kernel_inttype UnrollStrd,
	bool IsConj,
	typename ArchitectureSpec,
	typename AlphaType,
	typename AType, typename ACntgStepType, typename AStrdStepType,
	typename XType, typename XCntgStepType,
	typename BetaType,
	typename YType, typename YCntgStepType
>
PERFLIBS_LINALG_INLINE
void gemv_a_strd_first_strd_loop(
	const kernel_inttype a_strd,
	AlphaType alpha,
	const AType *a, const ACntgStepType a_cntg_step, const AStrdStepType a_strd_step,
	const XType *x, const XCntgStepType x_cntg_step,
	BetaType beta,
	      YType *y, const YCntgStepType y_cntg_step) {

	using alpha_type   = compute_type_t< AlphaType >;
	using compute_type = promote_t<alpha_type, XType>;

	std::array<compute_type, UnrollCntg> x_vals;
	#ifdef __clang__
		#pragma unroll(UnrollCntg)
	#endif
	for(kernel_inttype i = 0; i!=UnrollCntg; ++i) {
		x_vals[i] = static_cast<compute_type>( x[ i * x_cntg_step ] ) * alpha;
	}

	kernel_inttype as = 0;
	for(; as <= (a_strd - UnrollStrd); as += UnrollStrd) {
		//init to zero
		std::array<compute_type, UnrollStrd> y_vals { };

		#ifdef __clang__
			#pragma unroll(UnrollCntg)
		#endif
		for(kernel_inttype i = 0; i != UnrollCntg; ++i) {
			#ifdef __clang__
				#pragma unroll(UnrollStrd)
			#endif
			for(kernel_inttype j = 0; j != UnrollStrd; ++j) {
				if constexpr(IsConj) {
					y_vals[j] += x_vals[i] * static_cast<compute_type>( conj(a[ (as + j) * a_strd_step + i * a_cntg_step ]) );
				}
				else {
					y_vals[j] += x_vals[i] * static_cast<compute_type>( a[ (as + j) * a_strd_step + i * a_cntg_step ] );
				}
			}
		}
		#ifdef __clang__
			#pragma unroll(UnrollStrd)
		#endif
		for(kernel_inttype j = 0; j != UnrollStrd; ++j) {
			y[(as + j) * y_cntg_step ] = y[(as + j) * y_cntg_step ] * beta + y_vals[j];
		}
	}

	const kernel_inttype a_strd_rem = a_strd - as;
	std::array<compute_type, UnrollStrd> y_vals;

	#ifdef __clang__
		#pragma unroll(UnrollStrd)
	#endif
	for(kernel_inttype i = 0; i!= a_strd_rem; ++i)
		y_vals[i] = zero<compute_type>;

	#ifdef __clang__
		#pragma unroll(UnrollCntg)
	#endif
	for(kernel_inttype i = 0; i != UnrollCntg; ++i) {
		#ifdef __clang__
			#pragma unroll(UnrollStrd)
		#endif
		for(kernel_inttype j = 0; j != a_strd_rem; ++j) {
			if constexpr(IsConj) {
				y_vals[j] += x_vals[i] * static_cast<compute_type>(conj(a[ (as + j) * a_strd_step + i * a_cntg_step ]));
			}
			else {
				y_vals[j] += x_vals[i] * static_cast<compute_type>(a[ (as + j) * a_strd_step + i * a_cntg_step ]);
			}
		}
	}
	#ifdef __clang__
		#pragma unroll(UnrollStrd)
	#endif
	for(kernel_inttype j = 0; j != a_strd_rem; ++j) {
		y[(as + j) * y_cntg_step] = y[ (as+ j) * y_cntg_step ] * beta + y_vals[j];
	}
}

template<
	bool IsConj,
	typename ArchitectureSpec,
	typename AType, typename ACntgStepType, typename AStrdStepType,
	typename XType, typename XCntgStepType,
	typename YType, typename YCntgStepType,
	typename ScalType
>
__attribute__((noinline))
void gemv_a_strd_first_impl(
	const kernel_inttype a_cntg, const kernel_inttype a_strd,
	ScalType alpha,
	const AType *a, const ACntgStepType a_cntg_step, const AStrdStepType a_strd_step,
	const XType *x, const XCntgStepType x_cntg_step,
	ScalType beta,
	      YType *y, const YCntgStepType y_cntg_step) {

	//tested all obvious combos with dgemv, 4x4 provided best performance across the widest range of problems
	constexpr kernel_inttype unroll_cntg = 4;
	constexpr kernel_inttype unroll_strd = 4;

	kernel_inttype ac = 0;
	for(; ac <= ( a_cntg - unroll_cntg ); ac += unroll_cntg) {
		//only apply beta once, ie when ac is 0;
		if(ac == 0) {
			gemv_a_strd_first_strd_loop<unroll_cntg, unroll_strd, IsConj, ArchitectureSpec>(
				a_strd,
				alpha,
				a + ac * a_cntg_step, a_cntg_step, a_strd_step,
				x + ac * x_cntg_step, x_cntg_step,
				beta,
				y, y_cntg_step);
		}
		else {
			gemv_a_strd_first_strd_loop<unroll_cntg, unroll_strd, IsConj, ArchitectureSpec>(
				a_strd,
				alpha,
				a + ac * a_cntg_step, a_cntg_step, a_strd_step,
				x + ac * x_cntg_step, x_cntg_step,
				float_one<ScalType>{}, //in place of beta
				y, y_cntg_step);
		}
	}
	for(; ac < a_cntg; ++ac) {
		//if the unrolled loop is only used once then, then we still need to apply beta once
		if(ac == 0) {
			gemv_a_strd_first_strd_loop<1, unroll_strd, IsConj, ArchitectureSpec>(
				a_strd,
				alpha,
				a + ac * a_cntg_step, a_cntg_step, a_strd_step,
				x + ac * x_cntg_step, x_cntg_step,
				beta,
				y, y_cntg_step);
		}
		else {
			gemv_a_strd_first_strd_loop<1, unroll_strd, IsConj, ArchitectureSpec>(
				a_strd,
				alpha,
				a + ac * a_cntg_step, a_cntg_step, a_strd_step,
				x + ac * x_cntg_step, x_cntg_step,
				float_one<ScalType>{}, //in place of beta
				y, y_cntg_step);
		}
	}
}

/**
 * A GEMV kernel which iterates over strd in the inner loop - this should give best performance
 * when a_strd_step is 1, although you should always test empirically
 *
 * WARNING: Netlib's GEMV parameters are inconsistent with other BLAS routines:
 * what M & N refer to depends on the trans parameter. Please pay attention to
 * the parameters in this routine.
 *
 * @param a_cntg the shared dimension of A & X (B)
 * @param a_strd the shared dimension of A & Y (C)
 * @param alpha the scalar by which the dot products of A & X are scaled by
 * @param a pointer to the first element of A
 * @param a_cntg_step the increace need to go between adjacent elements in the cntg dimension
 * @param a_strd_step the increace need to go between adjacent elements in the strd dimension
 * @param x pointer to the first element of X, note if x_cntg_step is negative, this should be the first element we want to process- not the first element in memory - this differs to Fortran
 * @param x_cntg_step the increment between adjacement element of x
 * @param beta the scalar by which the elements of Y are scaled before accumulating with the dot product of A & X
 * @param y pointer to the first element of Y, note if y_cntg_step is negative, this should be the first element we want to process- not the first element in memory - this differs to Fortran
 * @param y_cntg_step the increment between adjacement element of y
 * @param y pointer to the first element of Y, note if y_cntg_step is negative, this should be the first element we want to process- not the first element in memory - this differs to Fortran
 *
 * Implementation detail. This function will create specialisations of the base algorithms depending on
 * whether any of the step parameters are 0, 1 or N best the compiler is (usually) capable of producing
 * a much better code gen when it knows that a stride will be 0 or 1.
 *
 * It does this by building up a bitset using 2 bits per step parameter to describe whether
 * each is 0, 1 or N and then invokes a unique version of the routine based on that
 */
template<bool IsConj, typename ArchitectureSpec, typename AType, typename XType, typename YType, typename ScalType>
__attribute__((noinline))
void gemv_a_strd_first(
	const kernel_inttype a_cntg, const kernel_inttype a_strd,
	ScalType alpha,
	const AType *a, const kernel_inttype a_cntg_step, const kernel_inttype a_strd_step,
	const XType *x, const kernel_inttype x_cntg_step,
	ScalType beta,
	      YType *y, const kernel_inttype y_cntg_step) {

	constexpr step_val_fixed<0> zero;
	constexpr step_val_fixed<1> one;

	const auto mask = [](auto val) -> std::uint8_t { return val == 0 ? 0 : val == 1 ? 1 : 2; };

	const std::uint8_t val
		= mask(a_cntg_step)
		| mask(a_strd_step) << 2
		| mask(x_cntg_step) << 4
		| mask(y_cntg_step) << 6;

	/*
	 *  Commented-out cases are intentional. We do not expect to hit them because
	 *  either the BLAS interface does not support them, or they are expected to
	 *  route to gemv_a_cntg_first (see gemv_kernels.hpp).
	 *
	 *  The code is still valid and retained for implementation comparisons; some
	 *  cases may perform better than expected (for example, small in-cache sizes).
	 */
	switch(val) {
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,         one, beta, y,        zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,         one, beta, y,        zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,         one, beta, y,        zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x, x_cntg_step, beta, y,        zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x, x_cntg_step, beta, y,        zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x, x_cntg_step, beta, y,        zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x, x_cntg_step, beta, y,        zero);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,        zero, beta, y,         one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,        zero, beta, y,         one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,        zero, beta, y,         one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,         one, beta, y,         one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,         one, beta, y,         one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,         one, beta, y,         one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x, x_cntg_step, beta, y,         one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x, x_cntg_step, beta, y,         one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x, x_cntg_step, beta, y,         one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x, x_cntg_step, beta, y,         one);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,        zero, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,        zero, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,        zero, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,         one, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,         one, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,         one, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x, x_cntg_step, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x, x_cntg_step, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x, x_cntg_step, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x, x_cntg_step, beta, y, y_cntg_step);
	default:
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_strd_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x, x_cntg_step, beta, y, y_cntg_step);
	}
}

template<
	bool IsConj,
	typename ArchitectureSpec,
	typename AType, typename ACntgStepType, typename AStrdStepType,
	typename XType, typename XCntgStepType,
	typename YType, typename YCntgStepType,
	typename ScalType
>
PERFLIBS_LINALG_INLINE
void gemv_a_cntg_first_impl(
	const kernel_inttype a_cntg, const kernel_inttype a_strd,
	ScalType alpha,
	const AType *a, const ACntgStepType a_cntg_step, const AStrdStepType a_strd_step,
	const XType *x, const XCntgStepType x_cntg_step,
	ScalType beta,
	      YType *y, const YCntgStepType y_cntg_step) {
	using alpha_type   = compute_type_t< ScalType >;
	using compute_type = promote_t<alpha_type, XType>;

	constexpr kernel_inttype unroll = 8;

	//inner loop moves down Y
	kernel_inttype as;
	for(as = 0; as <= ( a_strd - unroll ); as += unroll) {
		const auto as_col = a + as * a_strd_step;

		std::array<compute_type, unroll> tmp {};

		for(kernel_inttype ac = 0; ac != a_cntg; ++ac) {
			if constexpr(IsConj) {
				const auto xval = static_cast<compute_type>(x[ ac * x_cntg_step ]);

				tmp[0] += xval * static_cast<compute_type>(conj(as_col[ 0 * a_strd_step + ac * a_cntg_step ]));
				tmp[1] += xval * static_cast<compute_type>(conj(as_col[ 1 * a_strd_step + ac * a_cntg_step ]));
				tmp[2] += xval * static_cast<compute_type>(conj(as_col[ 2 * a_strd_step + ac * a_cntg_step ]));
				tmp[3] += xval * static_cast<compute_type>(conj(as_col[ 3 * a_strd_step + ac * a_cntg_step ]));
				tmp[4] += xval * static_cast<compute_type>(conj(as_col[ 4 * a_strd_step + ac * a_cntg_step ]));
				tmp[5] += xval * static_cast<compute_type>(conj(as_col[ 5 * a_strd_step + ac * a_cntg_step ]));
				tmp[6] += xval * static_cast<compute_type>(conj(as_col[ 6 * a_strd_step + ac * a_cntg_step ]));
				tmp[7] += xval * static_cast<compute_type>(conj(as_col[ 7 * a_strd_step + ac * a_cntg_step ]));
			}
			else {
				const auto xval = static_cast<compute_type>(x[ ac * x_cntg_step ]);

				tmp[0] += xval * static_cast<compute_type>(as_col[ 0 * a_strd_step + ac * a_cntg_step ]);
				tmp[1] += xval * static_cast<compute_type>(as_col[ 1 * a_strd_step + ac * a_cntg_step ]);
				tmp[2] += xval * static_cast<compute_type>(as_col[ 2 * a_strd_step + ac * a_cntg_step ]);
				tmp[3] += xval * static_cast<compute_type>(as_col[ 3 * a_strd_step + ac * a_cntg_step ]);
				tmp[4] += xval * static_cast<compute_type>(as_col[ 4 * a_strd_step + ac * a_cntg_step ]);
				tmp[5] += xval * static_cast<compute_type>(as_col[ 5 * a_strd_step + ac * a_cntg_step ]);
				tmp[6] += xval * static_cast<compute_type>(as_col[ 6 * a_strd_step + ac * a_cntg_step ]);
				tmp[7] += xval * static_cast<compute_type>(as_col[ 7 * a_strd_step + ac * a_cntg_step ]);
			}
		}

		y[ ( as + 0 ) * y_cntg_step ] = y[ ( as + 0 ) * y_cntg_step ] * beta + alpha * tmp[0];
		y[ ( as + 1 ) * y_cntg_step ] = y[ ( as + 1 ) * y_cntg_step ] * beta + alpha * tmp[1];
		y[ ( as + 2 ) * y_cntg_step ] = y[ ( as + 2 ) * y_cntg_step ] * beta + alpha * tmp[2];
		y[ ( as + 3 ) * y_cntg_step ] = y[ ( as + 3 ) * y_cntg_step ] * beta + alpha * tmp[3];
		y[ ( as + 4 ) * y_cntg_step ] = y[ ( as + 4 ) * y_cntg_step ] * beta + alpha * tmp[4];
		y[ ( as + 5 ) * y_cntg_step ] = y[ ( as + 5 ) * y_cntg_step ] * beta + alpha * tmp[5];
		y[ ( as + 6 ) * y_cntg_step ] = y[ ( as + 6 ) * y_cntg_step ] * beta + alpha * tmp[6];
		y[ ( as + 7 ) * y_cntg_step ] = y[ ( as + 7 ) * y_cntg_step ] * beta + alpha * tmp[7];
	}

	//inner loop moves down Y
	for(; as != a_strd; ++as) {
		const auto as_col = a + as * a_strd_step;

	 	auto tmp = zero<compute_type>;

		for(kernel_inttype ac = 0; ac != a_cntg; ++ac) {
			if constexpr(IsConj) {
				tmp += static_cast<compute_type>(x[ ac * x_cntg_step ])  * static_cast<compute_type>(conj(as_col[ ac * a_cntg_step ]));
			}
			else {
				tmp += static_cast<compute_type>(x[ ac * x_cntg_step ])  * static_cast<compute_type>(as_col[ ac * a_cntg_step ]);
			}
		}

		y[ as *  y_cntg_step ] = y[ as *  y_cntg_step ] * beta + alpha * tmp;
	}
}

/**
 * A GEMV kernel which iterates over cntg in the inner loop - this should give best performance
 * when a_cntg_step is 1, although you should always test empirically
 *
 * WARNING: Netlib's GEMV parameters are inconsistent with other BLAS routines:
 * what M & N refer to depends on the trans parameter. Please pay attention to
 * the parameters in this routine.
 *
 * @param a_cntg the shared dimension of A & X (B)
 * @param a_strd the shared dimension of A & Y (C)
 * @param alpha the scalar by which the dot products of A & X are scaled by
 * @param a pointer to the first element of A
 * @param a_cntg_step the increace need to go between adjacent elements in the cntg dimension
 * @param a_strd_step the increace need to go between adjacent elements in the strd dimension
 * @param x pointer to the first element of X, note if x_cntg_step is negative, this should be the first element we want to process- not the first element in memory - this differs to Fortran
 * @param x_cntg_step the increment between adjacement element of x
 * @param beta the scalar by which the elements of Y are scaled before accumulating with the dot product of A & X
 * @param y pointer to the first element of Y, note if y_cntg_step is negative, this should be the first element we want to process- not the first element in memory - this differs to Fortran
 * @param y_cntg_step the increment between adjacement element of y
 * @param y pointer to the first element of Y, note if y_cntg_step is negative, this should be the first element we want to process- not the first element in memory - this differs to Fortran
 *
 * Implementation detail. This function will create specialisations of the base algorithms depending on
 * whether any of the step parameters are 0, 1 or N best the compiler is (usually) capable of producing
 * a much better code gen when it knows that a stride will be 0 or 1.
 *
 * It does this by building up a bitset using 2 bits per step parameter to describe whether
 * each is 0, 1 or N and then invokes a unique version of the routine based on that
 */
template<bool IsConj, typename ArchitectureSpec, typename AType, typename XType, typename YType, typename ScalType>
__attribute__((noinline))
void gemv_a_cntg_first(
	const kernel_inttype a_cntg, const kernel_inttype a_strd,
	ScalType alpha,
	const AType *a, const kernel_inttype a_cntg_step, const kernel_inttype a_strd_step,
	const XType *x, const kernel_inttype x_cntg_step,
	ScalType beta,
	      YType *y, const kernel_inttype y_cntg_step) {

	constexpr step_val_fixed<0> zero;
	constexpr step_val_fixed<1> one;

	//builds a 2 bit value depending on whether the step param is 0, 1 or N
	const auto mask = [](auto val) -> std::uint8_t { return val == 0 ? 0 : val == 1 ? 1 : 2; };

	//get the 2bit patterns and shift them into a unique position in a 8bit int
	const std::uint8_t val
		= mask(a_cntg_step)
		| mask(a_strd_step) << 2
		| mask(x_cntg_step) << 4
		| mask(y_cntg_step) << 6;

	/*
	 * each permutation of the step parameter states will produce a unique number, we can then switch on these numbers
	 *
	 *  Commented-out cases are intentional. We do not expect to hit them because
	 *  either the BLAS interface does not support them, or they are expected to
	 *  route to gemv_a_cntg_first (see gemv_kernels.hpp).
	 *
	 *  The code is still valid and retained for implementation comparisons; some
	 *  cases may perform better than expected (for example, small in-cache sizes).
	 */
	switch(val) {
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,        zero, beta, y,        zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,        zero, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,         one, beta, y,        zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,         one, beta, y,        zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,         one, beta, y,        zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,         one, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x, x_cntg_step, beta, y,        zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x, x_cntg_step, beta, y,        zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x, x_cntg_step, beta, y,        zero);
	//case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x, x_cntg_step, beta, y,        zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x, x_cntg_step, beta, y,        zero);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,        zero, beta, y,         one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,        zero, beta, y,         one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,        zero, beta, y,         one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,        zero, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,         one, beta, y,         one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,         one, beta, y,         one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,         one, beta, y,         one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,         one, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x, x_cntg_step, beta, y,         one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x, x_cntg_step, beta, y,         one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x, x_cntg_step, beta, y,         one);
	//case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x, x_cntg_step, beta, y,         one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x, x_cntg_step, beta, y,         one);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,        zero, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,        zero, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,        zero, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,        zero, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x,         one, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x,         one, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x,         one, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x,         one, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,        zero, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,        zero, x, x_cntg_step, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,        zero, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero,         one, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one,         one, x, x_cntg_step, beta, y, y_cntg_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step,         one, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,        zero, a_strd_step, x, x_cntg_step, beta, y, y_cntg_step);
	//case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a,         one, a_strd_step, x, x_cntg_step, beta, y, y_cntg_step);
	default:
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return gemv_a_cntg_first_impl<IsConj, ArchitectureSpec>(a_cntg, a_strd, alpha, a, a_cntg_step, a_strd_step, x, x_cntg_step, beta, y, y_cntg_step);
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_GEMV_FALLBACK_KERNELS_HPP
