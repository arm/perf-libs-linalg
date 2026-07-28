/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_K_UPDATE_BASIC_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_K_UPDATE_BASIC_HPP

#include "spec/strategy_tag.hpp"

#include "framework/buffer_pool.hpp"
#include "framework/alloc.hpp"

#include "operators/pack.hpp"
#include "operators/tri_resident.hpp"
#include "operators/partial_separate.hpp"
#include "operators/kernel_exec.hpp"
#include "operators/copy_matrix.hpp"

#include <string_view>

/**
 * Performs A SYRK or HERK without any parallelism
 *
 */
namespace perflibs::linalg::matmul {

class rank_update_generic;

class rank_k_update_basic {
public:
	static constexpr std::string_view name() { return "rank_k_update_basic"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_update_generic>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if( !this->can_compute(pctx) ) return false;

		auto pctx_local = pctx;

		pctx_local.a.estrange_parent();
		pctx_local.b.estrange_parent();
		pctx_local.c.estrange_parent();

		const auto spec         = get_spec(spec::strategy_tag<rank_update_generic>{}, pctx_local);
		const auto kspec        = spec.kernel_spec;
		using kernel_value_type = typename decltype(kspec)::value_type;

		const kernel_inttype a_buf_sz    = spec.a_convert.elements_required(pctx_local.a.cntg(), pctx_local.a.strd());
		const kernel_inttype b_buf_sz    = spec.b_convert.elements_required(pctx_local.a.cntg(), pctx_local.b.strd());
		const kernel_inttype c_buf_sz    = pctx_local.c.cntg() * pctx_local.c.strd();
		const kernel_inttype buffer_size = a_buf_sz + b_buf_sz + c_buf_sz;
		auto buffer                      = get_memory<kernel_value_type>(buffer_size);

		buffer_pool a_buf { buffer,                       1, buffer_size };
		buffer_pool b_buf { buffer + a_buf_sz,            1, buffer_size };
		buffer_pool c_buf { buffer + a_buf_sz + b_buf_sz, 1, buffer_size };


		const kernel_inttype a_unroll = kspec.a_interleave_spec.strd_unroll * kspec.a_interleave_spec.strd_interleave();
		const kernel_inttype b_unroll = kspec.b_interleave_spec.strd_unroll * kspec.b_interleave_spec.strd_interleave();

		auto driver =
			pack             { a_matrix, a_buf	, spec.a_convert,
			tri_resident     { c_matrix, b_strd, pctx_local.c.is_lower(), b_unroll, a_unroll,
			pack             { b_matrix, b_buf, spec.b_convert,
			partial_separate { kspec.a_interleave_spec.strd_unroll, kspec.b_interleave_spec.strd_unroll,
				//when process rectangular sectoins of matrix, then just use the normal gemm
				kernel_exec  { kspec.kernel, kspec.apply_beta },

				//when processing triangular sections then wrap the gemm in a copy
				copy_matrix { c_matrix, c_buf, general_cntg_contig_generator{},
				kernel_exec { kspec.kernel, kspec.apply_beta }}
			}}}};

		driver(pctx_local.a, pctx_local.b, pctx_local.c, compute_position{0, 0, 0}, pctx_local.alpha, pctx_local.beta);

		if constexpr (is_hermitian_matrix_v<typename ProblemContext::c_matrix_type>) {
			zero_diag_imag(pctx_local.c);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<rank_update_generic>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		return pctx.alpha != zero<typename ProblemContext::scalar_type>
		    && pctx.beta_zero_mode != zero_mode::scale
		    && pctx.a.cntg_step() >= 1 && pctx.a.strd_step() >= 1
		    && pctx.b.cntg_step() >= 1 && pctx.b.strd_step() >= 1
		    && pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class rank_k_update_basic
} // perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_RANK_K_UPDATE_BASIC_HPP
