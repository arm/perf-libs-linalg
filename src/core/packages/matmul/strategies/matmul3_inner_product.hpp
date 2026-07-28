/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_INNER_PRODUCT_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_INNER_PRODUCT_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul3_inner_product {
	template<typename ProblemContext, typename T2>
	PERFLIBS_LINALG_INLINE
	void set_result(const ProblemContext& pctx, const T2& res) const {
		if(pctx.beta == zero<T2>) {
			if(pctx.alpha == one<T2>) {
				// TODO replace `c.data()[0]` with `c(0, 0)` once `pctx.c` can be assigned to
				// using `operator()`
				pctx.c.data()[0] = res;
			}
			else {
				pctx.c.data()[0] = pctx.alpha * res;
			}
		}
		else if(pctx.beta == one<T2>) {
			if(pctx.alpha == one<T2>) {
				pctx.c.data()[0] += res;
			}
			else {
				pctx.c.data()[0] += pctx.alpha * res;
			}
		}
		else {
			pctx.c.data()[0] = pctx.beta * pctx.c.data()[0] + pctx.alpha * res;
		}
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	void impl(const ProblemContext& pctx) const {
		if (pctx.a.cntg() < 0) {
			return;
		}

		// we then get all of the configuration details about how we will execute the routine.
		const auto spec = get_spec(spec::strategy_tag<matmul3_inner_product>{}, pctx);

		if constexpr (omp::is_mp) {
			using T2 = typename ProblemContext::c_matrix_type::value_type;

			if (spec.max_threads > 1 && pctx.a.cntg() != 0) {
				// Partition the threads to the work, note thread throttling is done in
				// `get_problem_spec` by altering `max_thread`
				const auto split = make_parallel_split(pctx.a.cntg(), 1, spec.max_threads, pctx.a.cntg());

				// Spawn a parallel region
				const auto result = reduce_add_parallel<T2>(split.threads, [&](kernel_inttype thread_num) {
					const auto [ start, work ] = work_distribution(thread_num, split);

					return spec.kernel(work,
						pctx.a.data() + start * pctx.a.cntg_step(),
						pctx.b.data() + start * pctx.b.cntg_step(),
						pctx.a.cntg_step(),
						pctx.b.cntg_step());
				});

				set_result(pctx, result);
				return;
			} // if max_threads > 1
		}

		const auto result = spec.kernel(pctx.a.cntg(), pctx.a.data(), pctx.b.data(),
		                                pctx.a.cntg_step(), pctx.b.cntg_step());
		set_result(pctx, result);
	}

public:
	static constexpr std::string_view name() { return "matmul3_inner_product"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_inner_product>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		//dot does work when one of the two inputs are conj, but it needs to be A
		if(pctx.b.is_conj()) {
			spec::problem_context pctx_swap_ab {
				matmul::matmul3 {
					pctx.b, pctx.a, pctx.c,
					pctx.alpha, pctx.beta
				},
				pctx.architecture_spec
			};

			this->impl(pctx_swap_ab);
		}
		else {
			this->impl(pctx);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_inner_product>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.a.strd() == 1 && pctx.b.strd() == 1
		    && !( pctx.a.is_conj() && pctx.b.is_conj() );
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_inner_product

} //namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_INNER_PRODUCT_HPP
