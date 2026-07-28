/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_OUTER_PRODUCT_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_OUTER_PRODUCT_HPP

#include "framework/alloc.hpp"
#include "framework/buffer_pool.hpp"
#include "matrix/operations.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/outer_product.hpp"
#include "operators/parallelize.hpp"
#include "operators/residents.hpp"
#include "operators/if_then_else.hpp"
#include "operators/set_scalars.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

//we need to have two different axpby kernels for fully efficient outer-prouduct
class matmul3_outer_product_beta_one { };

class matmul3_outer_product {
public:
	static constexpr std::string_view name() { return "matmul3_outer_product"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_outer_product>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using a_data_type = std::remove_cv_t< typename ProblemContext::a_matrix_type::value_type >;

		if( !this->can_compute(pctx) ) return false;

		const auto spec        = get_spec(spec::strategy_tag<matmul3_outer_product>{}, pctx);
		const auto axpy_kernel = get_spec(axpby_kernel_tag { spec::strategy_tag<matmul3_outer_product_beta_one>{} }, pctx);

		const auto buf_size = min(pctx.a.strd(), spec.a_strd_block_size);

		auto *buffer = pctx.a.is_physical() && is_strd_contig(pctx.a)
		             ? nullptr
		             : get_memory<a_data_type, memory_bank::outer_product>(buf_size * spec.max_threads);

		buffer_pool buffer_pool { buffer, spec.max_threads, buf_size };

		auto driver =
			parallelize  { general_parallel_strat, b_strd, spec.max_threads,
			resident     { a_matrix, 1, spec.a_strd_block_size, true,
			if_then_else {
				[](const auto& a, const auto&... _) {
					return a.absolute_cntg() == 0;
				},

				//IF - is first iteration along cntg (k)
				copy_matrix            { a_matrix, buffer_pool, general_strd_contig_generator{},
				outer_product_terminal { spec.kernel_axpby } },

				//ELSE - don't reapply beta
				set_scalar             { one< typename ProblemContext::scalar_type >, true,
				copy_matrix            { a_matrix, buffer_pool, general_strd_contig_generator{},
				outer_product_terminal { axpy_kernel       } }}
			}}};

		driver(pctx.a, pctx.b, pctx.c, compute_position{}, pctx.alpha, pctx.beta);

		if (buffer != nullptr) {
			return_memory<a_data_type, memory_bank::outer_product>(buffer);
		}

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_outer_product>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const {
		//return true;
		return pctx.alpha != zero<> && pctx.a.cntg() > 0;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_outer_product

} // namespace perflibs::linalg::matmul

#endif //PERFLIBS_LINALG_MATMUL_STRATEGIES_OUTER_PRODUCT_HPP
