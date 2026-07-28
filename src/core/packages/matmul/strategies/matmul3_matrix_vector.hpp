/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_GEMV_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_GEMV_HPP

#include "framework/alloc.hpp"
#include "framework/linalg_util.hpp"
#include "operators/parallelize.hpp"
#include "spec/strategy_tag.hpp"

#include "perflibs_assert.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

template<typename GemvKernel>
class gemv_kernel_exec {
	GemvKernel kernel_;
	bool       apply_beta_;
public:
	PERFLIBS_LINALG_INLINE
	gemv_kernel_exec(GemvKernel kernel, bool apply_beta)
	:	kernel_     { kernel     }
	,	apply_beta_ { apply_beta }
	{	}

	template <typename AType, typename BType, typename CType, typename ScalarType>
	PERFLIBS_LINALG_INLINE
	void operator()(const AType& a, const BType& b, CType& c, const compute_position& pos, ScalarType alpha, ScalarType beta) const {
		PERFLIBS_ASSERT(a.cntg() == b.cntg(), "a.cntg() != b.cntg()");
		PERFLIBS_ASSERT(a.strd() == c.cntg(), "a.strd() != c.cntg()");
		PERFLIBS_ASSERT(b.strd() == c.strd(), "b.strd() != c.strd()");
		PERFLIBS_ASSERT(b.strd() == 1,        "b must be vector");
		PERFLIBS_ASSERT(c.strd() == 1,        "c must be vector");

		if(beta == zero<ScalarType>) {
			set(beta, c);
			beta = one<ScalarType>;
		}
		else if(apply_beta_ && beta != one<ScalarType>) {
			scale(beta, c);
			beta = one<ScalarType>;
		}

		kernel_(
			a.cntg(), a.strd(),
			alpha,
			a.data(), a.cntg_step(), a.strd_step(),
			b.data(), b.cntg_step(),
			beta,
			c.data(), c.cntg_step());
	}
}; //class gemv_kernel

/**
 * If either a.strd() or b.strd() is 1 then then we are computing a matrix-vector problem and can
 * use GEMV to do the work rather than a full GEMM
 */
class matmul3_matrix_vector {
public:
	static constexpr std::string_view name() { return "matmul3_matrix_vector"; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	void impl(const ProblemContext& pctx) const {
		using scalar_type = typename ProblemContext::scalar_type;

		if(empty(pctx.a)) {
			return;
		}

		if(pctx.alpha == zero<scalar_type>) {
			if(pctx.beta == zero<scalar_type>) {
				set(pctx.beta, pctx.c);
			}
			else if(pctx.beta != one<scalar_type>) {
				scale(pctx.beta, pctx.c);
			}
			return;
		}

		const auto spec = get_spec(spec::strategy_tag<matmul3_matrix_vector>{}, pctx);

		//build the driver
		const compute_position pos { 0, 0, 0 };

		auto driver =
			/*
			 * Parallelise over the a.strd() / c.cntg() dimension
			 * this will ensure there are no race conditions as each thread will
			 * get its own chunk of the destination vector
			 */
			parallelize      { general_parallel_strat, a_strd, spec.max_threads, spec.kernel.num_cols,
			gemv_kernel_exec { spec.kernel.gemv_kernel, spec.kernel.apply_beta } };

		if(spec.kernel.scale_b) {
			// Replace the user's B with an internal scaled copy.
			using b_data_type = std::remove_cv_t<typename ProblemContext::b_matrix_type::value_type>;
			auto buffer = get_memory<b_data_type, memory_bank::gemv>(pctx.b.cntg());

			//create a new matrix, using the new buffer with the same dimensions as the input
			general_matrix scaled_b { matrix_base { buffer, pctx.b.cntg(), pctx.b.strd(), 1, pctx.b.cntg() } };

			//do an out of place scale from the input buffer to our allocated buffer
			scale(static_cast<b_data_type>(pctx.alpha), pctx.b, scaled_b);

			//because the scale has already been applied, we can set alpha to 1 now.
			auto alpha = one<scalar_type>;

			driver(pctx.a, scaled_b, pctx.c, pos, alpha, pctx.beta);

			return_memory<b_data_type, memory_bank::gemv>(scaled_b.data());
		}
		else {
			driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);
		}
	}

public:
	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		//we can do a GEMV if b is the matrix, but only if swap A & B and transpose C
		if (pctx.b.strd() > 1) {
			const spec::problem_context pctx_transpose {
				matmul::matmul3 {
					pctx.b,
					pctx.a,
					pctx.c.transpose(),
					pctx.alpha, pctx.beta
				},
				pctx.architecture_spec
			};
			this->impl(pctx_transpose);
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
	requires spec::has_get_spec<spec::strategy_tag<matmul3_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		if(pctx.b.strd() > 1) {
			return ( ( pctx.b.cntg_step() == 1 && pctx.b.strd_step() >= 1 )
			      || ( pctx.b.strd_step() == 1 && pctx.b.cntg_step() >= 1 && !pctx.b.is_conj()) )
			    && !pctx.a.is_conj() && pctx.a.strd() == 1;
		}
		else {
			return ( ( pctx.a.cntg_step() == 1 && pctx.a.strd_step() >= 1 )
			      || ( pctx.a.strd_step() == 1 && pctx.a.cntg_step() >= 1 && !pctx.a.is_conj()) )
			    && !pctx.b.is_conj() && pctx.b.strd() == 1;
		}
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_matrix_vector

} //namespace perflibs::linalg::matmul
#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_GEMV_HPP
