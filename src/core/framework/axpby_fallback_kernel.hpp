/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_AXPBY_FALLBACK_KERNEL_HPP
#define PERFLIBS_LINALG_AXPBY_FALLBACK_KERNEL_HPP

#include "framework/axpby_fallback_kernel_fwd.hpp"
#include "perflibs_assert.hpp"

namespace perflibs::linalg {
namespace {

template<bool IsConj, typename AlphaType, typename XType, typename XStepType, typename BetaType, typename YType, typename YStepType>
[[maybe_unused]]
void axpby_fallback_impl(kernel_inttype n, AlphaType alpha, const XType *x, XStepType x_step, BetaType beta, YType *y,  YStepType y_step) {

	using alpha_type   = compute_type_t< AlphaType >;
	using compute_type = promote_t<alpha_type, XType>;


	for(kernel_inttype i = 0; i != n; ++i) {
		if constexpr(IsConj) {
			y[ i * y_step ] =  y[ i * y_step ] * beta + static_cast<compute_type>( conj(x[ i * x_step ]) ) * alpha;
		}
		else {
			y[ i * y_step ] =  y[ i * y_step ] * beta + static_cast<compute_type>( x[ i * x_step ] ) * alpha;
		}
	}
}

} // namespace

template<bool IsConj, typename ArchitectureSpec, typename XType, typename YType, typename ScalType, zero_mode AlphaZero, zero_mode BetaZero>
void axpby_fallback(kernel_inttype n, ScalType alpha, const XType *x, ScalType beta, YType *y, kernel_inttype x_step, kernel_inttype y_step) {

	constexpr float_zero<ScalType> fp_zero;
	constexpr float_one<ScalType>  fp_one;
	constexpr step_val_fixed<0>    i_zero;
	constexpr step_val_fixed<1>    i_one;

	const auto step_mask  = [](auto val) -> std::uint8_t { return val == 0 ? 0 : val == 1 ? 1 : 2; };

	// If AlphaZero == zero_mode::set then we use the compile-time alpha=0 instantiation which does not propagate NaNs in X
	const auto alpha_mask = [](auto val) -> std::uint8_t { return val == zero<ScalType> && AlphaZero == zero_mode::set ? 0 : val == one<ScalType> ? 1 : 2; };
	// If BetaMode  == zero_mode::set then we use the compile-time beta=0  instantiation which does not propagate NaNs in Y
	const auto beta_mask  = [](auto val) -> std::uint8_t { return val == zero<ScalType> && BetaZero  == zero_mode::set ? 0 : val == one<ScalType> ? 1 : 2; };

	const std::uint8_t val
		= step_mask(x_step)
		| step_mask(y_step) << 2
		| alpha_mask(alpha) << 4
		| beta_mask(beta)   << 6;

	switch(val) {
	default: PERFLIBS_ASSERT(false, "axpby case no handled");
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step, fp_zero, y, i_zero);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step, fp_zero, y,  i_one);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one, fp_zero, y, y_step);
	case ( 0 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step, fp_zero, y, y_step);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step,  fp_one, y, i_zero);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step,  fp_one, y,  i_one);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one,  fp_one, y, y_step);
	case ( 1 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step,  fp_one, y, y_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, i_zero,    beta, y, y_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x,  i_one,    beta, y, y_step);
	case ( 2 << 6 ) | ( 0 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n, fp_zero, x, x_step,    beta, y, y_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, i_zero,    beta, y, y_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x,  i_one,    beta, y, y_step);
	case ( 2 << 6 ) | ( 1 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,  fp_one, x, x_step,    beta, y, y_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 0 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step,    beta, y, i_zero);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 1 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step,    beta, y,  i_one);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 0 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, i_zero,    beta, y, y_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 1 ): return axpby_fallback_impl<IsConj>(n,   alpha, x,  i_one,    beta, y, y_step);
	case ( 2 << 6 ) | ( 2 << 4 ) | ( 2 << 2 ) | ( 2 ): return axpby_fallback_impl<IsConj>(n,   alpha, x, x_step,    beta, y, y_step);
	}
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_AXPBY_FALLBACK_KERNEL_HPP
