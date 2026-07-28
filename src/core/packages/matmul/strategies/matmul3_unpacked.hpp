/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_UNPACKED_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_UNPACKED_HPP

#include "blas/gemm_small_impl.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matmul3_unpacked {
public:
	static constexpr std::string_view name() { return "matmul3_unpacked"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (ProblemContext& pctx) const {
		using a_value_type = typename ProblemContext::a_matrix_type::value_type;
		using architecture_spec_type = typename ProblemContext::architecture_spec_type;

		if( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<matmul3_interleaved_strategy_tag>{}, pctx);

		gemm_small_impl<a_value_type, architecture_spec_type>::get_kernel()(
			spec.max_threads,
			c_to_trans(gemm_transa(pctx)), c_to_trans(gemm_transb(pctx)),
			gemm_m(pctx), gemm_n(pctx), gemm_k(pctx),
			pctx.alpha,
			pctx.a.data(), gemm_lda(pctx),
			pctx.b.data(), gemm_ldb(pctx),
			pctx.beta,
			pctx.c.data(), gemm_ldc(pctx));

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		using a_value_type = typename ProblemContext::a_matrix_type::value_type;
		using architecture_spec_type = typename ProblemContext::architecture_spec_type;

		return gemm_small_impl<a_value_type, architecture_spec_type>::can_compute(pctx);
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_unpacked

} // namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_MATMUL3_UNPACKED_HPP
