/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MISC_STRATEGIES_FIND_INDEX_HPP
#define PERFLIBS_LINALG_MISC_STRATEGIES_FIND_INDEX_HPP

#include "matrix/matrix.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include "detect/omp.hpp"
#include "perflibs_type_traits.hpp"

#include "perflibs_util.hpp"

namespace perflibs::linalg::misc {

class find_index_strategy {
public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<find_index_strategy>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using real_type = remove_complex_t<typename ProblemContext::scalar_type>;

		// Retrieve the machine tuned parameters for a problem described in the above struct
		const auto spec = get_spec(spec::strategy_tag<find_index_strategy>{}, pctx);

		if constexpr(omp::is_mp) {
			if (spec.max_threads > 1) {
				// Partition the threads to the work, note thread throttling is done in `get_problem_spec` by altering `max_thread`
				const auto split =  make_parallel_split(pctx.x.cntg(), 1, spec.max_threads, pctx.x.cntg());
				// 224 is the biggest number of thread I can envision, dyn array will grow if more are needed
				dyn_array<perflibs::trivial_pair<kernel_inttype, real_type>, 224> thread_values (split.threads);
				// Spawn a parallel region
				parallel(split.threads, [&](kernel_inttype thread_num) {
						const auto[ start, work ] = work_distribution(thread_num, split, 1);

						auto thread_work = get_cntg_panel(pctx.x, start, work);

						//each thread will store its thread local result in its own index in the shared dny_array
						thread_values[ thread_num ] =
								spec.kernel(thread_work.cntg(), thread_work.data(), thread_work.cntg_step());

						thread_values[thread_num].first += start;
					}
				);
				/*
				* the master thread will then traverse the shared array (after all the threads complete)
				* and find the largest or smallest value, and return its index.
				*/
				kernel_inttype idx = thread_values[0].first;
				real_type val = thread_values[0].second;

				for(kernel_inttype i = 1; i!=split.threads; ++i) {
					if (pctx.operation == find_operation::absolute_max) {
						if(thread_values[i].second > val) {
							val = thread_values[i].second;
							idx = thread_values[i].first;
						};
					} else if (pctx.operation == find_operation::absolute_min) {
						if(thread_values[i].second < val) {
							val = thread_values[i].second;
							idx = thread_values[i].first;
						}
					};
				};
				//Set result here and return true:
				pctx.out = idx;
				return true;
			}; //if max_threads > 1
		}; //if omp

		// Sequential
		//Set result here and return true:
		pctx.out = spec.kernel(pctx.x.cntg(), pctx.x.data(), pctx.x.cntg_step()).first;
		return true;
	} // bool operator

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext&) const { return false; }
}; // find_index_strategy
} //namespace perflibs::linalg::misc

#endif //PERFLIBS_LINALG_MISC_STRATEGIES_FIND_INDEX_HPP
