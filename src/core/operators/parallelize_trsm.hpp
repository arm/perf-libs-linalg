/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_PARALLEL_TRSM_HPP
#define PERFLIBS_LINALG_PARALLEL_TRSM_HPP

#include "framework/parallel.hpp"
#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<typename Next>
class parallelize_trsm {
	kernel_inttype max_threads_;
	kernel_inttype interleave_;
	Next next_;

public:
	parallelize_trsm(kernel_inttype max_threads, kernel_inttype interleave, Next next)
	:	max_threads_ { max_threads     }
	,	interleave_  { interleave      }
	,	next_        { std::move(next) }
	{	}

	template <typename AMatrixType, typename BMatrixType>
	inline
	void operator()(const AMatrixType& a, BMatrixType& b) {
		if (max_threads_ > 1) {
			const auto split = make_parallel_split(b.strd(), interleave_, max_threads_);

			parallel(split.threads, [=, this](auto thread_num) mutable {
				const auto[chunk_start, chunk_size] = work_distribution(thread_num, split);
				auto b_panel = b.sub_matrix(0, b.cntg(), chunk_start, chunk_size);

				next_(a, b_panel);
			});
		}
		else {
			// pass through to the next level in the single threaded case
			next_(a, b);
		}
	}
}; // class parallelize_trsm

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_PARALLEL_TRSM_HPP
