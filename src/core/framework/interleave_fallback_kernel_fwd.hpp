/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_FWD_HPP
#define PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_FWD_HPP

#include "linalg_util.hpp"

namespace perflibs::linalg {

enum class interleave_flags : kernel_inttype {
	None      = 0ul,		   // 00000000
	ZeroPad   = 1ul << 0,   // 00000001
	IsConj    = 1ul << 1,	// 00000010
	DiagVirt  = 1ul << 2,  // 00000100
	DiagUnit  = 1ul << 3,  // 00001000
	DiagReal  = 1ul << 4,  // 00010000
	LowerVirt = 1ul << 5, // 00100000
	UpperVirt = 1ul << 6, // 01000000
	VirtZero  = 1ul << 7,  // 10000000
};

constexpr PERFLIBS_LINALG_INLINE kernel_inttype operator|(const enum interleave_flags l_value,
                                                     const enum interleave_flags r_value) {
	return kernel_inttype(kernel_inttype(l_value) | kernel_inttype(r_value));
}
constexpr PERFLIBS_LINALG_INLINE kernel_inttype operator|(const kernel_inttype l_value,
                                                     const enum interleave_flags r_value) {
	return kernel_inttype(l_value | kernel_inttype(r_value));
}
constexpr PERFLIBS_LINALG_INLINE kernel_inttype operator&(const kernel_inttype l_value,
                                                     const enum interleave_flags r_value) {
	return l_value & kernel_inttype(r_value);
}

template<std::size_t Interleave, kernel_inttype Flags, typename TSrc, typename TDst, typename MachineSpec>
void n_cpp_interleave(
	std::size_t src_cntg, std::size_t src_strd, const TSrc *__restrict src, std::size_t src_cntg_stride, std::size_t src_strd_stride,
	std::size_t dst_cntg, std::size_t dst_strd,       TDst *__restrict dst,                              std::size_t dst_srd_stride,
	kernel_inttype submat_cntg, kernel_inttype submat_strd);

template<std::size_t Interleave, kernel_inttype Flags, typename TSrc, typename TDst, typename MachineSpec>
void t_cpp_interleave(
	std::size_t src_cntg, std::size_t src_strd, const TSrc *__restrict src, std::size_t src_cntg_stride, std::size_t src_strd_stride,
	std::size_t dst_cntg, std::size_t dst_strd,       TDst *__restrict dst,                              std::size_t dst_strd_stride,
	kernel_inttype submat_cntg, kernel_inttype submat_strd);

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_FWD_HPP
