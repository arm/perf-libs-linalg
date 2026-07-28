/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_BASIC_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_BASIC_HPP

#include "operators/no_inline.hpp"
#include "operators/pack.hpp"
#include "operators/copy_matrix.hpp"
#include "operators/kernel_exec.hpp"
#include "framework/alloc.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

/**
 * Implements the most basic linalg GEMM stack, packs the data such that the kernel
 * can process it then executes the kernel. No blocking, no parallelism.
 *
 * Designed for cases where the overheads in the framework are detrimental to the performance
 */
class matmul3_interleaved_basic {
public:
	static constexpr std::string_view name() { return "matmul3_interleaved_basic"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<matmul3_interleaved_strategy_tag>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		// basic uses the same tuning as sequential
		const auto spec  = get_spec(spec::strategy_tag<matmul3_interleaved_strategy_tag>{}, pctx);
		const auto kspec = spec.kernel_spec;

		using kernel_spec_type   = typename std::remove_cvref_t<decltype(kspec)>;
		using a_kernel_data_type = typename kernel_spec_type::a_data_type;
		using b_kernel_data_type = typename kernel_spec_type::b_data_type;
		using c_kernel_data_type = typename kernel_spec_type::c_data_type;

		if( !this->can_compute(pctx) ) return false;

		/*
		 * Is the datatype of kernel the same as the input types? It might be different
		 * if this is a mixed precision GEMM or for a "faked" hgemm
		 */
		constexpr bool is_homogeneous       = std::is_same_v<typename ProblemContext::c_matrix_type::value_type, c_kernel_data_type>;

		const kernel_inttype a_buf_sz_bytes =                         spec.a_align*2_ki + iround_if_not_zero(spec.a_convert.bytes_required(pctx.a.cntg(), pctx.a.strd()), spec.a_align);
		const kernel_inttype b_buf_sz_bytes =                         spec.b_align*2_ki + iround_if_not_zero(spec.b_convert.bytes_required(pctx.b.cntg(), pctx.b.strd()), spec.b_align);
		const kernel_inttype c_buf_sz_bytes = is_homogeneous ? 0_ki : spec.c_align*2_ki + iround_if_not_zero(pctx.c.cntg() * pctx.c.strd() * sizeof(c_kernel_data_type),  spec.c_align);

		auto a_ptr_bytes                    = get_memory<std::uint8_t>(a_buf_sz_bytes + b_buf_sz_bytes + c_buf_sz_bytes);
		auto b_ptr_bytes                    = a_ptr_bytes + a_buf_sz_bytes;

		auto a_ptr_elems                    = align_to(reinterpret_cast<a_kernel_data_type*>(a_ptr_bytes), spec.a_align);
		auto b_ptr_elems                    = align_to(reinterpret_cast<b_kernel_data_type*>(b_ptr_bytes), spec.b_align);

		const compute_position pos { 0, 0, 0, 0 };

		if constexpr(is_homogeneous) {
			auto driver =
				pack        { b_matrix, buffer_pool { b_ptr_elems }, spec.b_convert,
				pack        { a_matrix, buffer_pool { a_ptr_elems }, spec.a_convert,
				kernel_exec { kspec.kernel, kspec.apply_beta } } };

			driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);
		}
		else {
			auto c_ptr_bytes = b_ptr_bytes + b_buf_sz_bytes;
			auto c_ptr_elems = align_to(reinterpret_cast<c_kernel_data_type*>(c_ptr_bytes), spec.c_align);

			auto driver =
				pack        { b_matrix, buffer_pool { b_ptr_elems }, spec.b_convert,
				pack        { a_matrix, buffer_pool { a_ptr_elems }, spec.a_convert,
				copy_matrix { c_matrix, buffer_pool { c_ptr_elems }, general_cntg_contig_generator{},
				kernel_exec { kspec.kernel, kspec.apply_beta } } } };

			driver(pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);
		}

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
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class matmul3_interleaved_basic

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_BASIC_HPP
