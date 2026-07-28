/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ARCHITECTURES_NEON_GET_KERNEL_SPEC_HPP
#define PERFLIBS_LINALG_ARCHITECTURES_NEON_GET_KERNEL_SPEC_HPP

#include "matrix/interleaved_tile_matrix_base.hpp"

#include "spec/routine_specs.hpp"
#include "spec/problem_context.hpp"

#include "kernel_specs/matmul3_kernels.hpp"
#include "kernel_specs/interleave_kernels.hpp"

#include "framework/which.hpp"

#include "packages/matmul/fwd.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_strategy_tag.hpp"
#include "packages/misc/problem_context_bases.hpp"

#include "perflibs_numeric_utils.hpp"
#include "perflibs_assert.hpp"

#include "atomic_kernels.hpp"

#include "detect/os.hpp"
#include "detect/cpu_info.hpp" //features.neon_bf16

#include <concepts>

namespace perflibs::linalg::spec {

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, ArchitectureSpec>>, r16>{

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r16, r16, r16>, ArchitectureSpec> [ hgemm_8_2_kernel_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, ArchitectureSpec>>, bf16> {

	if(machine::cpu_info::get_cpu_features().neon_bf16) {
		return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, bf16>, ArchitectureSpec> [ a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8_idx ];
	}
	else {
		return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, bf16>, ArchitectureSpec> [ bgemm_fp32_fmla_acle_nobf16exten_idx ];
	}
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, ArchitectureSpec>>, r32> {

	if constexpr(std::is_same_v<bf16, std::remove_cv_t<typename ProblemContextBase::a_matrix_type::value_type>>) {

		if(machine::cpu_info::get_cpu_features().neon_bf16) {
			return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, r32>, ArchitectureSpec> [ a64_interleaved_nomerge_bf16fp32_mmla_6x8_idx ];
		}

		return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, r32>, ArchitectureSpec> [ sbgemm_fp32_fmla_acle_idx ];
	}
	else {
		// The assembly SGEMM kernel uses more GPRs than we have available on Arm64EC
		// so we cannot use it there and must instead use an ACLE fallback
		if constexpr(os::windows_arm64ec) {
			return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<r32, r32, r32>, ArchitectureSpec> [ sgemm_vanilla_big_acle_idx ];
		}

		if constexpr(std::is_same_v<StrategyTag, strategy_tag<matmul::matmul3_interleaved_strategy_tag_accelerator>>) {
				return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<r32, r32, r32>, ArchitectureSpec> [ sme2_interleaved_nomerge_fp32_mopa_2VLx2VL_idx ];
		}

		return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r32, r32, r32>, ArchitectureSpec> [ sgemm_vanilla_big_idx ];
	}
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag> tag, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, ArchitectureSpec>>, r64> {

	if constexpr(std::is_same_v<StrategyTag, strategy_tag<matmul::matmul3_interleaved_strategy_tag_accelerator>>) {
			return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<r64, r64, r64>, ArchitectureSpec> [ sme2_interleaved_nomerge_fp64_mopa_2VLx2VL_idx ];
	}

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r64, r64, r64>, ArchitectureSpec> [ dgemm_vanilla_big_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, ArchitectureSpec>>, c32> {

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<c32, c32, c32>, ArchitectureSpec> [ cgemm_large_kernel_TN_nmk_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, ArchitectureSpec>>, c64> {

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<c64, c64, c64>, ArchitectureSpec> [ zgemm_vanilla_big_TN_idx ];
}

template<typename StrategyTag, typename ProblemContext, typename System>
auto get_dot_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	return get_neon_dot_kernel(pctx);
}

template<typename ProblemContextBase, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
constexpr auto get_atomic_dispatch_kernel(const problem_context<ProblemContextBase, ArchitectureSpec>&) {
	using data_type = typename ProblemContextBase::scalar_type;

	return &dispatch_atomic_neon<data_type>;
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_find_index_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using data_type = typename ProblemContext::scalar_type;

	if (pctx.operation == misc::find_operation::absolute_max) {
		if (pctx.x.cntg_step() == 1) {
			if constexpr(std::is_same_v<data_type, float>) {
				return &isamax_kernel;
			}
			else if constexpr(std::is_same_v<data_type, double>) {
				return &idamax_kernel;
			}
			else if constexpr (std::is_same_v<data_type, std::complex<float>>) {
				return &icamax_kernel;
				}
				else if constexpr(std::is_same_v<data_type, std::complex<double>>) {
				return &izamax_kernel;
			}
		}
		// This means incx != 1, so use the c++ iamax implementation
		return &iamax_fallback<data_type>;
	} //if pctx.operation == find_operation::absolute_max
	else { //(pctx.operation == misc::find_operation::absolute_min)
		if (pctx.x.cntg_step() == 1) {
			if constexpr(std::is_same_v<data_type, float>) {
				return &isamin_kernel;
			}
			else if constexpr(std::is_same_v<data_type, double>) {
				return &idamin_kernel;
			}
			else if constexpr (std::is_same_v<data_type, std::complex<float>>) {
				return &icamin_kernel;
			}
			else if constexpr(std::is_same_v<data_type, std::complex<double>>) {
				return &izamin_kernel;
			}
		}
		// This means incx != 1, so use the c++ iamin implementation
		return &iamin_fallback<data_type>;
	} // if (pctx.operation == find_operation::absolute_min)

}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_l1_norm_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using data_type = typename ProblemContext::scalar_type;

	if (pctx.x.cntg_step() == 1) {
		if constexpr (std::is_same_v<data_type, float>) {
			return &sasum_kernel;
		}
		else if constexpr (std::is_same_v<data_type, double>) {
			return &dasum_kernel;
		}
		else if constexpr (std::is_same_v<data_type, std::complex<float>>) {
			return &scasum_kernel;
		}
		else if constexpr (std::is_same_v<data_type, std::complex<double>>) {
			return &dzasum_kernel;
		}
	}

	return &l1_norm_fallback<data_type>;
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_l2_norm_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using data_type = typename ProblemContext::scalar_type;

	if (pctx.x.cntg_step() == 1) {
		return &l2_norm_neon_kernel<data_type>;
	}

	return &l2_norm_fallback<data_type>;
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_swap_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using data_type = typename ProblemContext::scalar_type;
	if (pctx.x.cntg_step() == 1 && pctx.y.cntg_step() == 1){
		if constexpr (std::is_same_v<data_type, r32>)
			return &sswap_kernel;
		else if constexpr (std::is_same_v<data_type, r64>)
			return &dswap_kernel;
		else if constexpr (std::is_same_v<data_type, c32>)
			return &cswap_kernel;
		else if constexpr (std::is_same_v<data_type, c64>)
			return &zswap_kernel;
	}
	return &swap_fallback<data_type>;
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_waxpby_kernel_system (StrategyTag strat_tag, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System) {
	return get_neon_waxpby_kernel(pctx);
}


template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_trsv_kernel_system(StrategyTag strat_tag, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System) {
	using pctx_type      = std::remove_cv_t<std::remove_reference_t<decltype(pctx)>>;
	using data_type      = typename pctx_type::b_matrix_type::value_type;

	const auto notrans   = pctx.transa == PERFLIBS_NOTRANS;
	const auto conjtrans = pctx.transa == PERFLIBS_CONJTRANS;

	const auto nounit    = !pctx.a.is_unit();
	const auto uplo      = notrans ? lower_flip(pctx.a.uplo()) : pctx.a.uplo();
	const auto upper     = uplo == PERFLIBS_UPPER;

	if (notrans) {
		if (upper) {
			if (nounit) return trsv_notrans_upper<data_type, true  /*nounit*/>;
			else        return trsv_notrans_upper<data_type, false /*nounit*/>;
		}
		else {
			if (nounit) return trsv_notrans_lower<data_type, true  /*nounit*/>;
			else        return trsv_notrans_lower<data_type, false /*nounit*/>;
		}
	}
	else if (conjtrans) {
		if (upper) {
			if (nounit) return trsv_trans_upper  <data_type, true  /*nounit*/, true /*conj*/>;
			else        return trsv_trans_upper  <data_type, false /*nounit*/, true /*conj*/>;
		}
		else {
			if (nounit) return trsv_trans_lower  <data_type, true  /*nounit*/, true  /*conj*/>;
			else        return trsv_trans_lower  <data_type, false /*nounit*/, true  /*conj*/>;
		}
	}
	else {
		if (upper) {
			if (nounit) return trsv_trans_upper  <data_type, true  /*nounit*/, false /*conj*/>;
			else        return trsv_trans_upper  <data_type, false /*nounit*/, false /*conj*/>;
		}
		else {
			if (nounit) return trsv_trans_lower  <data_type, true  /*nounit*/, false /*conj*/>;
			else        return trsv_trans_lower  <data_type, false /*nounit*/, false /*conj*/>;
		}
	}
}

template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec, typename System>
PERFLIBS_LINALG_INLINE
auto get_trsm_kernel_system(StrategyTag strat_tag, const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, System) {
	using pctx_type      = std::remove_cv_t<std::remove_reference_t<decltype(pctx)>>;
	using data_type      = typename pctx_type::b_matrix_type::value_type;

	const auto lside     = pctx.side   == PERFLIBS_LEFT;
	const auto notrans   = pctx.transa == PERFLIBS_NOTRANS;
	const auto conjtrans = pctx.transa == PERFLIBS_CONJTRANS;

	const auto nounit    = !pctx.a.is_unit();
	const auto uplo      = lside
	                       ? notrans ? lower_flip(pctx.a.uplo()) : pctx.a.uplo()
	                       : notrans ? pctx.a.uplo() : lower_flip(pctx.a.uplo());
	const auto upper     = uplo == PERFLIBS_UPPER;

	// trsm_kernel template parameters: data_type, forward, transa, transb, nounit, conja
	if (lside) {
		if (notrans) {
			if (upper) { // left, notrans, upper
				if (nounit) return trsm_kernel<data_type, false, true , true , true , false>;
				else        return trsm_kernel<data_type, false, true , true , false, false>;
			}
			else { // left, notrans, lower
				if (nounit) return trsm_kernel<data_type, true , true , true , true , false>;
				else        return trsm_kernel<data_type, true , true , true , false, false>;
			}
		}
		else if (conjtrans) {
			if (upper) { // left, conjtrans, upper
				if (nounit) return trsm_kernel<data_type, true , false, true , true , true >;
				else        return trsm_kernel<data_type, true , false, true , false, true >;
			}
			else { // left, conjtrans, lower
				if (nounit) return trsm_kernel<data_type, false, false, true , true , true >;
				else        return trsm_kernel<data_type, false, false, true , false, true >;
			}
		}
		else {
			if (upper) { // left, trans, upper
				if (nounit) return trsm_kernel<data_type, true , false, true , true , false>;
				else        return trsm_kernel<data_type, true , false, true , false, false>;
			}
			else { // left, trans, lower
				if (nounit) return trsm_kernel<data_type, false, false, true , true , false>;
				else        return trsm_kernel<data_type, false, false, true , false, false>;
			}
		}
	}
	else {
		if (notrans) {
			if (upper) { // right, notrans, upper
				if (nounit) return trsm_kernel<data_type, true , false, false, true , false>;
				else        return trsm_kernel<data_type, true , false, false, false, false>;
			}
			else { // right, notrans, lower
				if (nounit) return trsm_kernel<data_type, false, false, false, true , false>;
				else        return trsm_kernel<data_type, false, false, false, false, false>;
			}
		}
		else if (conjtrans) {
			if (upper) { // right, conjtrans, upper
				if (nounit) return trsm_kernel<data_type, false, true , false, true , true >;
				else        return trsm_kernel<data_type, false, true , false, false, true >;
			}
			else { // right, conjtrans, lower
				if (nounit) return trsm_kernel<data_type, true , true , false, true , true >;
				else        return trsm_kernel<data_type, true , true , false, false, true >;
			}
		}
		else {
			if (upper) { // right, trans, upper
				if (nounit) return trsm_kernel<data_type, false, true , false, true , false>;
				else        return trsm_kernel<data_type, false, true , false, false, false>;
			}
			else { // right, trans, lower
				if (nounit) return trsm_kernel<data_type, true , true , false, true , false>;
				else        return trsm_kernel<data_type, true , true , false, false, false>;
			}
		}
	}
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_rot_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using scalar_type = typename ProblemContext::scalar_type;
	using vector_type = typename ProblemContext::vector_type;


	if constexpr (is_complex_v<scalar_type>) { // crot, zrot
		if (pctx.incx == 1 && pctx.incy == 1)
			return &rot_neon_kernel_complex<vector_type>;
	}
	return &rot_kernel_fallback<vector_type, scalar_type>; // srot, drot, csrot, zdrot
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_rotg_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using scalar_type = typename ProblemContext::scalar_type;

	return &rotg_kernel<scalar_type>;
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_rotm_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using scalar_type = typename ProblemContext::scalar_type;

	return &rotm_kernel<scalar_type>;
}

template<typename StrategyTag, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_rotmg_kernel_system(StrategyTag strat_tag, const ProblemContext& pctx, System) {
	using scalar_type = typename ProblemContext::scalar_type;

	return &rotmg_kernel<scalar_type>;
}

template <typename StrategyTag, typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_gemv_kernel(StrategyTag strat_tag, const ProblemContext& pctx, const kernel_inttype max_threads) {

	return get_gemv_neon_kernel(pctx, max_threads);
}

template<typename InterleaveKernelSpec>
struct unary_interleave {
	InterleaveKernelSpec interleave_kernel_spec;

	using dst_data_type = typename InterleaveKernelSpec::dst_data_type;

	template<typename SrcMatrixType, typename DstMatrixType>
	PERFLIBS_LINALG_INLINE
	void pack(const SrcMatrixType& src, const DstMatrixType& dst) const {
		if(interleave_kernel_spec.kernel)
			invoke(interleave_kernel_spec.kernel, src, dst);
		else
			copy(src, dst, /* zero_slack */ true);
	}

	PERFLIBS_LINALG_INLINE
	auto cntg_interleave() const {
		return interleave_kernel_spec.cntg_interleave();
	}

	PERFLIBS_LINALG_INLINE
	auto strd_interleave() const {
		return interleave_kernel_spec.strd_interleave();
	}

	PERFLIBS_LINALG_INLINE
	auto cntg_interleave_step() const {
		return interleave_kernel_spec.cntg_interleave_step;
	}

	PERFLIBS_LINALG_INLINE
	auto strd_interleave_step() const {
		return interleave_kernel_spec.strd_interleave_step;
	}

	PERFLIBS_LINALG_INLINE
	auto split_factor() const { return interleave_kernel_spec.split_factor; }
}; //struct unary_interleave

template<typename InterleaveKernelSpec>
struct ternary_interleave {
	InterleaveKernelSpec physical_interleave_kernel_spec;
	InterleaveKernelSpec virtual_interleave_kernel_spec;
	InterleaveKernelSpec general_interleave_kernel_spec;

	using dst_data_type = typename InterleaveKernelSpec::dst_data_type;

	template<typename SrcMatrixType, typename DstMatrixType>
	PERFLIBS_LINALG_INLINE
	auto pack(const SrcMatrixType& src, const DstMatrixType& dst) const {
		if(src.is_physical() && general_interleave_kernel_spec.kernel) {
			invoke(general_interleave_kernel_spec.kernel, src, dst);
		}
		else if(physical_interleave_kernel_spec.kernel && virtual_interleave_kernel_spec.kernel) {
			invoke(physical_interleave_kernel_spec.kernel, src,                     dst);
			invoke(virtual_interleave_kernel_spec.kernel,  src.reflect_transpose(), dst);
		}
		else {
			copy(src, dst, /* zero_slack */ true);
		}
	}

	PERFLIBS_LINALG_INLINE
	auto cntg_interleave() const {
		return physical_interleave_kernel_spec.cntg_interleave();
	}

	PERFLIBS_LINALG_INLINE
	auto strd_interleave() const {
		PERFLIBS_ASSERT(physical_interleave_kernel_spec.strd_interleave() == virtual_interleave_kernel_spec.strd_interleave()
		          && physical_interleave_kernel_spec.strd_interleave() == general_interleave_kernel_spec.strd_interleave());

		return physical_interleave_kernel_spec.strd_interleave();
	}

	PERFLIBS_LINALG_INLINE
	auto cntg_interleave_step() const {
		return physical_interleave_kernel_spec.cntg_interleave_step;
	}

	PERFLIBS_LINALG_INLINE
	auto strd_interleave_step() const {
		return physical_interleave_kernel_spec.strd_interleave_step;
	}

	PERFLIBS_LINALG_INLINE
	auto split_factor() const {
		PERFLIBS_ASSERT(physical_interleave_kernel_spec.split_factor == virtual_interleave_kernel_spec.split_factor
		          && physical_interleave_kernel_spec.split_factor == general_interleave_kernel_spec.split_factor);

		return physical_interleave_kernel_spec.split_factor;
	}
}; //struct ternary_interleave

template<typename KernelInvoker>
struct basic_interleave_format : KernelInvoker {
	kernel_inttype       gemm_cntg_unroll_factor;
	kernel_inttype       gemm_strd_unroll_factor;

	PERFLIBS_LINALG_INLINE
	kernel_inttype elements_required(kernel_inttype max_cntg, kernel_inttype max_strd) const {
		return iround(max_strd, this->strd_interleave() * this->gemm_strd_unroll_factor)
		     * iround(max_cntg, this->gemm_cntg_unroll_factor);
	}
}; //structn basic_interleave_format

template<typename KernelInvoker>
struct interleave_format : basic_interleave_format<KernelInvoker> {

	PERFLIBS_LINALG_INLINE
	kernel_inttype bytes_required(kernel_inttype max_cntg, kernel_inttype max_strd) const {
		return this->elements_required(max_cntg, max_strd)
		     * sizeof( typename interleave_format::dst_data_type );
	}


	template<typename SrcMatrixType, typename DstDataType>
	PERFLIBS_LINALG_INLINE
	interleaved_tile_matrix_base<DstDataType> make_packed(const SrcMatrixType& src, DstDataType *dst_buffer) const {
		const kernel_inttype cntg                = iround(src.cntg(), this->gemm_cntg_unroll_factor);
		const kernel_inttype strd                = src.strd();

		const kernel_inttype cntg_interleave     = this->cntg_interleave();
		const kernel_inttype strd_interleave     = this->strd_interleave();

		const kernel_inttype cntg_interleave_step = this->cntg_interleave_step();
		const kernel_inttype strd_interleave_step = this->strd_interleave_step();

		const kernel_inttype cntg_tile_step       = strd_interleave_step * strd_interleave;
		const kernel_inttype strd_tile_step       = cntg * strd_interleave;

		return {
			dst_buffer,
			cntg_interleave,      strd_interleave,
			cntg,                 strd,
			cntg_interleave_step, strd_interleave_step,
			cntg_tile_step,       strd_tile_step
		};
	}
}; //struct interleave_format

template<typename KernelInvoker>
struct split_complex_format : basic_interleave_format<KernelInvoker> {
	PERFLIBS_LINALG_INLINE
	kernel_inttype bytes_required(kernel_inttype max_cntg, kernel_inttype max_strd) const {
		return this->elements_required(max_cntg, max_strd)
		     * sizeof( typename split_complex_format::dst_data_type );
	}

	template<typename SrcMatrixType, typename DstDataType>
	PERFLIBS_LINALG_INLINE
	split_complex_matrix<DstDataType> make_packed(const SrcMatrixType& src, DstDataType *dst_buffer) const {
		const kernel_inttype cntg_exten = iround(src.cntg(), this->gemm_cntg_unroll_factor);
		const kernel_inttype strd_step  = cntg_exten * this->strd_interleave();

		return { dst_buffer, this->strd_interleave(), this->split_factor(), cntg_exten, src.strd(), strd_step };
	}
}; //struct split_complex_format

template<typename InterleaveObject>
struct convert : InterleaveObject {

	template<typename SrcMatrixType, typename DstDataType>
	PERFLIBS_LINALG_INLINE
	auto operator() (const SrcMatrixType& src, DstDataType *dst_buffer) const {
		const auto dst = this->make_packed(src, dst_buffer);
		this->pack(src, dst);
		return dst;
	}
}; //struct convert

template<typename SrcMatrixType, typename DstMatrixSpec, typename ProblemContext, typename System>
auto get_interleave_kernel(const SrcMatrixType& src, const DstMatrixSpec& dst_matrix_spec, const ProblemContext& pctx, System system) {
	if constexpr(is_symmetric_matrix_v<SrcMatrixType> || is_hermitian_matrix_v<SrcMatrixType>) {
		return ternary_interleave {
			get_physical_interleave_spec(src,                    dst_matrix_spec, pctx, system),
			get_virtual_interleave_spec(src.reflect_transpose(), dst_matrix_spec, pctx, system),
			get_general_interleave_spec(src,                     dst_matrix_spec, pctx, system)
		};
	}
	else {
		return unary_interleave {
			get_physical_interleave_spec(src, dst_matrix_spec, pctx, system)
		};
	}
}

template<typename InterleaveObject, typename DstDataType, typename GemmKernelSpec>
auto get_format_object(InterleaveObject&& object, const matrix_interleave_spec<DstDataType>& dst_matrix_spec, const GemmKernelSpec& interleave_matmul_kernel_spec) {
	if constexpr(is_complex_v<DstDataType>) {
		return split_complex_format<InterleaveObject> {
			std::forward<InterleaveObject>( object ),
			interleave_matmul_kernel_spec.cntg_unroll,
			dst_matrix_spec.strd_unroll
		};
	}
	else  {
		return interleave_format<InterleaveObject> {
			std::forward<InterleaveObject>( object ),
			interleave_matmul_kernel_spec.cntg_unroll,
			dst_matrix_spec.strd_unroll
		};
	}
}

template<which_matrix WhichMatrixVal, typename GemmKernelSpec>
struct convert_object_tag {
	const GemmKernelSpec& interleave_matmul_kernel_spec;

	PERFLIBS_LINALG_INLINE
	convert_object_tag(which_matrix_constant<WhichMatrixVal>, const GemmKernelSpec& interleave_matmul_kernel_spec)
	:	interleave_matmul_kernel_spec { interleave_matmul_kernel_spec }
	{	}

	constexpr auto which_matrix() {
		return which_matrix_constant<WhichMatrixVal> { };
	}
};

template<auto WhichMatrixVal, typename GemmKernelSpec, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(convert_object_tag<WhichMatrixVal, GemmKernelSpec> tag, const ProblemContext& pctx, System system) {
	constexpr auto which_matrix = tag.which_matrix();
	const auto& src             = get_matrix(which_matrix, pctx);

	const auto dst_matrix_spec  = [&] {
		if constexpr(is_triangular_matrix_v<typename ProblemContext::a_matrix_type>) {
			if(! is_left(pctx)) {
				return WhichMatrixVal == which_matrix::a
				     ? tag.interleave_matmul_kernel_spec.b_interleave_spec
				     : tag.interleave_matmul_kernel_spec.a_interleave_spec;
			}
		}
		return WhichMatrixVal == which_matrix::a
			 ? tag.interleave_matmul_kernel_spec.a_interleave_spec
			 : tag.interleave_matmul_kernel_spec.b_interleave_spec;
	}();

	return convert {
		get_format_object(
			get_interleave_kernel(src, dst_matrix_spec, pctx, system),
			dst_matrix_spec,
			tag.interleave_matmul_kernel_spec
		)
	};
}

} //namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_ARCHITECTURES_NEON_GET_KERNEL_SPEC_HPP
