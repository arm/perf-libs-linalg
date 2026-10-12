/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_VECTOR_SCALAR_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_VECTOR_SCALAR_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matadd3_vector_scalar {
public:
	static constexpr std::string_view name() { return "matadd3_vector_scalar"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matadd3_vector_scalar>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (!this->can_compute(pctx)) return false;

		const auto spec = get_spec(spec::strategy_tag<matadd3_vector_scalar>{}, pctx);

		const auto elements         =  pctx.c.cntg() * pctx.c.strd();

		const bool same_dimensions  =  pctx.a.cntg()      == pctx.c.cntg()
		                            && pctx.a.strd()      == pctx.c.strd()
		                            && pctx.b.cntg()      == pctx.c.cntg()
		                            && pctx.b.strd()      == pctx.c.strd();

		const bool cntg_contiguous  =  pctx.a.cntg_step() == 1
		                            && pctx.b.cntg_step() == 1
		                            && pctx.c.cntg_step() == 1
		                            && pctx.a.strd_step() == pctx.a.cntg()
		                            && pctx.b.strd_step() == pctx.b.cntg()
		                            && pctx.c.strd_step() == pctx.c.cntg();

		const bool strd_contiguous  =  pctx.a.strd_step() == 1
		                            && pctx.b.strd_step() == 1
		                            && pctx.c.strd_step() == 1
		                            && pctx.a.cntg_step() == pctx.a.strd()
		                            && pctx.b.cntg_step() == pctx.b.strd()
		                            && pctx.c.cntg_step() == pctx.c.strd();

		if (same_dimensions && (cntg_contiguous || strd_contiguous)) {
			if constexpr (omp::is_mp) {
				if (spec.max_threads > 1) {
					const auto split = make_parallel_split(elements, 1, spec.max_threads);

					parallel(split.threads, [=, &pctx, &spec](kernel_inttype thread_num) {
						const auto [start, work] = work_distribution(thread_num, split);

						spec.kernel(
							work,
							pctx.alpha,
							pctx.a.data() + start,
							pctx.beta,
							pctx.b.data() + start,
							pctx.c.data() + start,
							1,
							1,
							1
						);
					});

					return true;
				}
			}

			spec.kernel(
				elements,
				pctx.alpha,
				pctx.a.data(),
				pctx.beta,
				pctx.b.data(),
				pctx.c.data(),
				1,
				1,
				1
			);

			return true;
		}

		const bool cntg_output = pctx.c.cntg_step() == 1;
		const auto vectors     = cntg_output ? pctx.c.strd()
		                                     : pctx.c.cntg();

		const auto impl = [&](kernel_inttype start, kernel_inttype work) {
			if (cntg_output) {
				for (kernel_inttype j = start; j != start + work; ++j) {
					spec.kernel(
						pctx.c.cntg(),
						pctx.alpha,
						pctx.a.data() + j * pctx.a.strd_step(),
						pctx.beta,
						pctx.b.data() + j * pctx.b.strd_step(),
						pctx.c.data() + j * pctx.c.strd_step(),
						pctx.a.cntg_step(),
						pctx.b.cntg_step(),
						pctx.c.cntg_step()
					);
				}
			}
			else {
				for (kernel_inttype i = start; i != start + work; ++i) {
					spec.kernel(
						pctx.c.strd(),
						pctx.alpha,
						pctx.a.data() + i * pctx.a.cntg_step(),
						pctx.beta,
						pctx.b.data() + i * pctx.b.cntg_step(),
						pctx.c.data() + i * pctx.c.cntg_step(),
						pctx.a.strd_step(),
						pctx.b.strd_step(),
						pctx.c.strd_step()
					);
				}
			}
		};

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1) {
				const auto split = make_parallel_split(vectors, 1, spec.max_threads);

				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split);

					impl(start, work);
				});

				return true;
			}
		}

		impl(0, vectors);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matadd3_vector_scalar>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {

		const bool cntg_output =   pctx.c.cntg_step() == 1;
		const bool strd_output =   pctx.c.strd_step() == 1
							   && (pctx.a.strd_step() == 1 || pctx.b.strd_step() == 1);

		return  pctx.alpha != zero<>
		    &&  pctx.beta  != zero<>
		    && (cntg_output || strd_output)
		    &&  pctx.a.is_physical()
		    &&  pctx.b.is_physical();
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_VECTOR_SCALAR_HPP
