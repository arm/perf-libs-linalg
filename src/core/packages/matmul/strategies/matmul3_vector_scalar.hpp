/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_VECTOR_SCALAR_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_VECTOR_SCALAR_HPP

#include "framework/linalg_util.hpp"
#include "framework/parallel.hpp"

#include "matrix/matrix.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/operations.hpp"

#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul3_vector_scalar {
	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	void impl(const ProblemContext& pctx) const {
		const auto spec = get_spec(spec::strategy_tag<matmul3_vector_scalar>{}, pctx);

		if constexpr (omp::is_mp) {
			if(spec.max_threads > 1 && pctx.c.cntg_step() != 0_ki) {
				const auto split = make_parallel_split(pctx.a.strd(), 1, spec.max_threads);

				parallel(split.threads, [&](auto thread_num) {
					const auto [ start, work ] = work_distribution(thread_num, split);

					auto x_panel = get_strd_panel(pctx.a, start, work);
					auto y_panel = get_cntg_panel(pctx.c, start, work);

					spec.kernel(y_panel.cntg(), pctx.alpha, x_panel.data(), pctx.beta,
					            y_panel.data(), x_panel.strd_step(), y_panel.cntg_step());
				});

				return;
			}
		}
		spec.kernel(pctx.a.strd(), pctx.alpha, pctx.a.data(), pctx.beta, pctx.c.data(),
		            pctx.a.strd_step(), pctx.c.cntg_step());
	}

public:
	static constexpr std::string_view name() { return "matmul3_vector_scalar"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_vector_scalar>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if (! this->can_compute(pctx) ) return false;

		if (pctx.b.strd() == 1 && pctx.b(0,0) == one<>) {
			//simple matmul3_vector_scalar case
			impl(pctx);
		}
		else {
			/*
			 * Problem can be computed with MATMUL3_VECTOR_SCALAR but isn't in the ideal form
			 * transpose and remove B by updating alpha
			 */
			auto pctx2 = pctx;

			if (pctx2.b.strd() != 1) {
				using std::swap;
				swap(pctx2.a, pctx2.b);
				pctx2.c = pctx2.c.transpose();
			}

			PERFLIBS_ASSERT(pctx2.c.cntg() == pctx2.a.strd());
			PERFLIBS_ASSERT(pctx2.b.strd() == 1);
			PERFLIBS_ASSERT(pctx2.c.strd() == 1);

			if(pctx2.b(0,0) != one<>) {
				using b_value_type = ProblemContext::b_matrix_type::value_type;
				pctx2.alpha *= pctx2.b(0, 0);
				pctx2.b = general_matrix { matrix_base { &one<b_value_type>, 1, 1, 0, 0 } };
			}

			impl(pctx2);
		}
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_vector_scalar>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.a.cntg() == 1 && pctx.b.cntg() == 1
		    && ( (pctx.a.strd() == 1) || (pctx.b.strd() == 1) );
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_vector_scalar

} //namespace perflibs::linalg::strat

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_VECTOR_SCALAR_HPP
