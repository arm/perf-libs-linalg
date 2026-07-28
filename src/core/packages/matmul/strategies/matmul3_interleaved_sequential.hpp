/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_SEQUENTIAL_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_SEQUENTIAL_HPP

#include "spec/strategy_tag.hpp"

#include "spec/get_block_sizes.hpp"

#include "framework/alloc.hpp"

#include "operators/residents.hpp"
#include "operators/pack.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/kernel_exec.hpp"

#include <string_view>

/**
 * Performs a GEMM with cache blocking but without any parallelism
 */
namespace perflibs::linalg::matmul {

class matmul3_interleaved_sequential {
public:
	static constexpr std::string_view name() { return "matmul3_interleaved_sequential"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		const auto spec  = get_spec(spec::strategy_tag<matmul3_interleaved_strategy_tag>{}, pctx);
		const auto kspec = spec.kernel_spec;

		using kernel_spec_type   = typename std::remove_cvref_t<decltype(kspec)>;
		using a_kernel_data_type = typename kernel_spec_type::a_data_type;
		using b_kernel_data_type = typename kernel_spec_type::b_data_type;
		using c_kernel_data_type = typename kernel_spec_type::c_data_type;


		/*
		 * Is the datatype of kernel the same as the input types? It might be different
		 * if this is a mixed precision GEMM or for a "faked" hgemm
		 */
		constexpr bool is_homogeneous       = std::is_same_v<typename ProblemContext::c_matrix_type::value_type, c_kernel_data_type>;

		//block_sizes
		const auto l1_cntg                  = get_l1_cntg(spec);
		const auto l1_strd                  = get_l1_strd(spec);
		const auto l2_strd                  = get_l2_strd(spec);

		const kernel_inttype a_buf_sz_bytes =                         spec.a_align*2_ki + iround_if_not_zero(spec.a_convert.bytes_required(l1_cntg, l1_strd), spec.a_align);
		const kernel_inttype b_buf_sz_bytes =                         spec.b_align*2_ki + iround_if_not_zero(spec.b_convert.bytes_required(l1_cntg, l2_strd), spec.b_align);
		const kernel_inttype c_buf_sz_bytes = is_homogeneous ? 0_ki : spec.c_align*2_ki + iround_if_not_zero(l1_strd * l2_strd * sizeof(c_kernel_data_type),  spec.c_align);

		auto a_ptr_bytes                    = get_memory<std::uint8_t>(a_buf_sz_bytes + b_buf_sz_bytes + c_buf_sz_bytes);
		auto b_ptr_bytes                    = a_ptr_bytes + a_buf_sz_bytes;
		auto c_ptr_bytes                    = b_ptr_bytes + b_buf_sz_bytes;

		auto a_ptr_elems                    = align_to(reinterpret_cast<a_kernel_data_type*>(a_ptr_bytes), spec.a_align);
		auto b_ptr_elems                    = align_to(reinterpret_cast<b_kernel_data_type*>(b_ptr_bytes), spec.b_align);
		auto c_ptr_elems                    = align_to(reinterpret_cast<c_kernel_data_type*>(c_ptr_bytes), spec.c_align);

		auto driver =
			resident    { b_matrix, l1_cntg, l2_strd, spec.l2_cntg_first,
			pack        { b_matrix, buffer_pool { b_ptr_elems }, spec.b_convert,
			resident    { a_matrix, l1_cntg, l1_strd, spec.l1_cntg_first,
			pack        { a_matrix, buffer_pool { a_ptr_elems }, spec.a_convert,
			copy_matrix { c_matrix, buffer_pool { c_ptr_elems }, general_cntg_contig_generator{},
			kernel_exec { kspec.kernel, kspec.apply_beta } } } } } };

		compute_position pos { 0, 0, 0, 0 };

		driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);

		return_memory(a_ptr_bytes);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool can_compute(const ProblemContext& pctx) const {
		return pctx.alpha != zero<typename ProblemContext::scalar_type>
		    && pctx.beta_zero_mode != zero_mode::scale
		    && pctx.a.cntg_step() >= 1 && pctx.a.strd_step() >= 1
		    && pctx.b.cntg_step() >= 1 && pctx.b.strd_step() >= 1
		    && pctx.c.cntg_step() == 1 && pctx.c.strd_step() >= 1;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext& pctx) const { return false; }
}; //class matmul3_interleaved_sequential

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_SEQUENTIAL_HPP
