/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_AXPBY_FALLBACK_KERNEL_FWD_HPP
#define PERFLIBS_LINALG_AXPBY_FALLBACK_KERNEL_FWD_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<bool IsConj, typename ArchitectureSpec, typename XType, typename YType, typename ScalType,
         zero_mode AlphaZero = zero_mode::set, zero_mode BetaZero = zero_mode::set>
void axpby_fallback(kernel_inttype n, ScalType alpha, const XType *x, ScalType beta, YType *y,
                    kernel_inttype x_step, kernel_inttype y_step);

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_AXPBY_FALLBACK_KERNEL_FWD_HPP
