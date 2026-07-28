/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_PARALLEL_SPAWN_HPP
#define PERFLIBS_LINALG_OPERATORS_PARALLEL_SPAWN_HPP

#include "framework/parallel.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

/**
 * A simple parallelism operator which invokes `next`
 * max_threads  times in  parallel
 *
 * Note: this operator is designed to be used with a parallelism
 * strategy operator which is responsible for partitioning the compute
 * space in accordance with thread index
 */
template<typename Next>
struct parallel_spawn {
	kernel_inttype max_threads;
	Next next;

	PERFLIBS_LINALG_INLINE
	void operator()(const auto& a, const auto& b, const auto& c, const compute_position& pos, const auto&... args) {
		parallel(max_threads, [&](kernel_inttype thread_num) {
			next(a, b, c, set_threads(pos, thread_num), args...);
		});
	}
}; ///class parallel_spawn

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_OPERATORS_PARALLEL_SPAWN_HPP
