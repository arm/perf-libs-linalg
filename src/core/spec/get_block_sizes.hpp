/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPEC_GET_BLOCK_SIZES_HPP
#define PERFLIBS_LINALG_SPEC_GET_BLOCK_SIZES_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg::spec {

template<typename Spec>
PERFLIBS_LINALG_INLINE
auto get_l1_cntg(const Spec& spec) {
	return spec.l1_cntg * spec.kernel_spec.cntg_unroll;
}

template<typename Spec>
PERFLIBS_LINALG_INLINE
auto get_l1_strd(const Spec& spec) {
	return spec.l1_strd * spec.kernel_spec.a_interleave_spec.strd_interleave() * spec.kernel_spec.a_interleave_spec.strd_unroll;
}

template<typename Spec>
PERFLIBS_LINALG_INLINE
auto get_l2_strd(const Spec& spec) {
	return spec.l2_strd * spec.kernel_spec.b_interleave_spec.strd_interleave() * spec.kernel_spec.b_interleave_spec.strd_unroll;
}

} //namespace perflibs::linalg::spec {

#endif // PERFLIBS_LINALG_SPEC_GET_BLOCK_SIZES_HPP
