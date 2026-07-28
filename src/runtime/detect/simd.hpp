/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_SIMD_HPP
#define PERFLIBS_SIMD_HPP

namespace perflibs {

namespace linalg {

namespace simd {

#ifdef __ARM_NEON
inline constexpr bool is_neon = true;
#else
inline constexpr bool is_neon = false;
#endif

#if defined(__ARM_FEATURE_SVE) || defined(PERFLIBS_IFUNC_SVE)
inline constexpr bool is_sve = true;
#else
inline constexpr bool is_sve = false;
#endif

/**
 * if platform has neon, but no SVE
 * ie neon only
 */
inline constexpr bool is_neon_not_sve() {
	return is_neon && (!is_sve);
}

} // namespace simd

} // namespace linalg

} // namespace perflibs

#endif
