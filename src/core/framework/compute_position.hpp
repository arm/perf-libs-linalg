/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_COMPUTE_POSITION_HPP
#define PERFLIBS_LINALG_FRAMEWORK_COMPUTE_POSITION_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

struct compute_position {
	/// m, n & k in old money
	/// iteration is used to denote iteration number - used for syr2k and frieds which does multiple passes
	/// block is used to uniquely label the submatrix being considered when parallelising
	/// blocks is the total number of blocks for this thread when parallelising
	kernel_inttype a_strd, b_strd, cntg, iteration { 0 }, block { 0 }, blocks { 0 }, thread_num { 0 };
}; //struct compute_position

namespace {

inline compute_position advance(const compute_position &pos,
	kernel_inttype a_strd,        kernel_inttype b_strd,    kernel_inttype cntg,
	kernel_inttype iteration = 0, kernel_inttype block = 0, kernel_inttype blocks = 0,
	kernel_inttype thread_num = 0) {

	return {
		pos.a_strd     + a_strd,
		pos.b_strd     + b_strd,
		pos.cntg       + cntg,
		pos.iteration  + iteration,
		pos.block      + block,
		pos.blocks     + blocks,
		pos.thread_num + thread_num
	};
}

PERFLIBS_LINALG_INLINE
compute_position set_threads(compute_position pos, kernel_inttype thread_num) {
	pos.thread_num = thread_num;
	return pos;
}

} // namespace anon
} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_COMPUTE_POSITION_HPP
