/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_ZERO_SCALAR_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_ZERO_SCALAR_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matadd3_zero_scalar {
public:
	static constexpr std::string_view name() { return "matadd3_zero_scalar"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matadd3_zero_scalar>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (!this->can_compute(pctx)) return false;

		using scalar_type = typename ProblemContext::scalar_type;

		const bool beta_is_zero = pctx.beta == zero<scalar_type>;

		const auto spec = get_spec(spec::strategy_tag<matadd3_zero_scalar>{}, pctx);

		if (beta_is_zero) {
			impl(spec, pctx.alpha, pctx.a, pctx.c);
		}
		else {
			impl(spec, pctx.beta, pctx.b, pctx.c);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		using scalar_type = typename ProblemContext::scalar_type;

		return pctx.alpha == zero<scalar_type>
		    || pctx.beta == zero<scalar_type>;
	}

private:
	template<typename Spec, typename Scalar, typename SrcMatrix, typename DstMatrix>
	PERFLIBS_LINALG_INLINE
	void impl(
		const Spec& spec,
		const Scalar scalar,
		const SrcMatrix& src,
		const DstMatrix& dst
	) const {
		const auto elements        =  dst.cntg() * dst.strd();

		const bool same_dimensions =  src.cntg()      == dst.cntg()
		                           && src.strd()      == dst.strd();

		const bool cntg_contiguous =  src.cntg_step() == 1
		                           && dst.cntg_step() == 1
		                           && src.strd_step() == src.cntg()
		                           && dst.strd_step() == dst.cntg();

		const bool strd_contiguous =  src.strd_step() == 1
		                           && dst.strd_step() == 1
		                           && src.cntg_step() == src.strd()
		                           && dst.cntg_step() == dst.strd();

		if (same_dimensions && (cntg_contiguous || strd_contiguous)) {
			if constexpr (omp::is_mp) {
				if (spec.max_threads > 1) {
					const auto split = make_parallel_split(elements, 1, spec.max_threads);

					parallel(split.threads, [=, &spec, &src, &dst](kernel_inttype thread_num) {
						const auto [start, work] = work_distribution(thread_num, split);

						spec.kernel(
							work,
							scalar,
							src.data() + start,
							zero<Scalar>,
							dst.data() + start,
							1,
							1
						);
					});

					return;
				}
			}

			spec.kernel(
				elements,
				scalar,
				src.data(),
				zero<Scalar>,
				dst.data(),
				1,
				1
			);

			return;
		}

		const bool cntg_output = dst.cntg_step() == 1;
		const auto vectors     = cntg_output ? dst.strd()
		                                     : dst.cntg();

		const auto compute = [&](kernel_inttype start, kernel_inttype work) {
			if (cntg_output) {
				for (kernel_inttype j = start; j != start + work; ++j) {
					spec.kernel(
						dst.cntg(),
						scalar,
						src.data() + j * src.strd_step(),
						zero<Scalar>,
						dst.data() + j * dst.strd_step(),
						src.cntg_step(),
						dst.cntg_step()
					);
				}
			}
			else {
				for (kernel_inttype i = start; i != start + work; ++i) {
					spec.kernel(
						dst.strd(),
						scalar,
						src.data() + i * src.cntg_step(),
						zero<Scalar>,
						dst.data() + i * dst.cntg_step(),
						src.strd_step(),
						dst.strd_step()
					);
				}
			}
		};

		if constexpr (omp::is_mp) {
			if (spec.max_threads > 1) {
				const auto split = make_parallel_split(vectors, 1, spec.max_threads);

				parallel(split.threads, [&](kernel_inttype thread_num) {
					const auto [start, work] = work_distribution(thread_num, split);

					compute(start, work);
				});

				return;
			}
		}

		compute(0, vectors);
	}
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD3_ZERO_SCALAR_HPP
