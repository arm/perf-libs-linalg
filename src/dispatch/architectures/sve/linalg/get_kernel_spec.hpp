/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_ARCHITECTURES_SVE_GET_KERNEL_SPEC_HPP
#define PERFLIBS_LINALG_ARCHITECTURES_SVE_GET_KERNEL_SPEC_HPP

#include "spec/routine_specs.hpp"
#include "spec/problem_context.hpp"

#include "kernel_specs/matmul3_kernels.hpp"
#include "kernel_specs/interleave_kernels.hpp"

#include "packages/batch_interleave/matmul/kernels/trmm_kernels.hpp"
#include "packages/batch_interleave/solve/kernels/trsm_kernels.hpp"

#include "packages/misc/problem_context_bases.hpp"
#include "packages/matmul/strategies/matmul3_interleaved_strategy_tag.hpp"

#include "sve/linalg/machine_spec.hpp"

#include "atomic_kernels.hpp"
#include "detect/cpu_info.hpp" //features.neon_bf16

namespace perflibs::linalg::spec {

template<machine::system System>
constexpr bool is_v2_v = System == machine::system::v2_axion
                      || System == machine::system::v2_c8g
                      || System == machine::system::v2_grace;

template<machine::system System>
constexpr bool is_v2(const system_t<System>&) { return is_v2_v<System>; }

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, sve_architecture_spec>>, r16> {

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r16, r16, r16>, sve_architecture_spec> [ hgemm_sve_big_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System system)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, sve_architecture_spec>>, r32> {

	if constexpr(std::is_same_v<bf16, std::remove_cv_t<typename ProblemContextBase::a_matrix_type::value_type>>) {

		if(machine::cpu_info::get_cpu_features().neon_bf16) {
			return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, r32>, sve_architecture_spec> [ a64_interleaved_nomerge_bf16fp32_mmla_12x8_idx ];
		}

		return interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, r32>, sve_architecture_spec> [ sbgemm_fp32_fmla_acle_idx ];
	}
	else {
		if constexpr (is_v2(system)) {
			return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r32, r32, r32>, sve_architecture_spec> [ sgemm_vanilla_big_idx ];
		}

		return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r32, r32, r32>, sve_architecture_spec> [ sgemm_sve_big_idx ];
	}
}


template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System system)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, sve_architecture_spec>>, r64> {

	if constexpr (is_v2(system)) {
		return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r64, r64, r64>, sve_architecture_spec> [ dgemm_vanilla_big_idx ];
	}

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<r64, r64, r64>, sve_architecture_spec> [ dgemm_sve_big_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System system)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, sve_architecture_spec>>, c32> {

	if constexpr (is_v2(system)) {
		return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<c32, c32, c32>, sve_architecture_spec> [ cgemm_large_kernel_TN_nmk_idx ];
	}

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<c32, c32, c32>, sve_architecture_spec> [ cgemm_sve_big_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(gemm_kernel_tag<gemm_kernel_types::interleaved, StrategyTag>, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System system)
    requires std::same_as<compute_precision_t<problem_context<ProblemContextBase, sve_architecture_spec>>, c64> {

	if constexpr (is_v2(system)) {
		return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<c64, c64, c64>, sve_architecture_spec> [ zgemm_vanilla_big_TN_idx ];
	}

	return interleave_matmul_kernel_specs< interleave_matmul_kernel_spec<c64, c64, c64>, sve_architecture_spec> [ zgemm_sve_big_idx ];
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_spec_system(
	batch_interleave_gemm_kernel_tag<StrategyTag>,
	const problem_context<ProblemContextBase, sve_architecture_spec>& pctx,
	System)
	requires std::same_as<typename ProblemContextBase::scalar_type, r64> {
	if (pctx.alpha == 0.0 && pctx.beta_zero_mode == zero_mode::scale) {
		return &batch_interleave::matmul::scal_gemm_shim<r64, sve_architecture_spec>;
	}

	if (pctx.a.matrix().cntg() == 1 && pctx.alpha != 0.0 && pctx.beta == 1.0) {
		return &batch_interleave::matmul::ger_gemm_shim<batch_interleave::matmul::ger_sve>;
	}

	return &batch_interleave::matmul::bi_dgemm_sve;
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_batch_interleave_trmm_kernel_system(
	StrategyTag,
	const problem_context<ProblemContextBase, sve_architecture_spec>&,
	System)
	requires std::same_as<typename ProblemContextBase::scalar_type, r64> {
	return &batch_interleave::matmul::dtrmm_sve_kernel;
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_batch_interleave_trsm_kernel_system(
	StrategyTag,
	const problem_context<ProblemContextBase, sve_architecture_spec>&,
	System)
	requires std::same_as<typename ProblemContextBase::scalar_type, r64> {
	return &batch_interleave::solve::dtrsm_sve_kernel;
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_ger_qr_kernel_system(
	StrategyTag, const problem_context<ProblemContextBase, sve_architecture_spec>&, System)
	requires std::same_as<std::remove_cv_t<typename ProblemContextBase::c_matrix_type::data_type>, r64> {
	return &::perflibs::interleave_batch::ger_qr_sve;
}

template<typename StrategyTag, typename ProblemContextBase, machine::system System>
PERFLIBS_LINALG_INLINE
auto get_dot_kernel_system(StrategyTag, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, system_t<System>) {
	if(const auto kernel = get_sve_dot_kernel(pctx); kernel != nullptr) {
		return kernel;
	}

	return get_neon_dot_kernel(pctx);
}

template<typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_waxpby_kernel_system (StrategyTag strat_tag, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System) {
	return get_sve_waxpby_kernel(strat_tag, pctx);
}

template<typename ProblemContextBase>
PERFLIBS_LINALG_INLINE
constexpr auto get_atomic_dispatch_kernel(const problem_context<ProblemContextBase, sve_architecture_spec>&) {
	using data_type = typename ProblemContextBase::scalar_type;

	return &dispatch_atomic_sve<data_type>;
}

template<typename StrategyTag, typename VectorType, typename ScalarType, typename System>
PERFLIBS_LINALG_INLINE
auto get_rot_kernel_system(
	StrategyTag,
	const problem_context<misc::rot<VectorType, ScalarType>, sve_architecture_spec>& pctx,
	System) {
	if constexpr (is_complex_v<ScalarType>) { // crot, zrot
		if (pctx.incx == 1 && pctx.incy == 1)
			return &rot_kernel_sve_complex<VectorType>;
	}
	return &rot_kernel_fallback<VectorType, ScalarType>; // srot, drot, csrot, zdrot
}

template <typename StrategyTag, typename ProblemContextBase, typename System>
PERFLIBS_LINALG_INLINE
auto get_l2_norm_kernel_system(StrategyTag strat_tag, const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, System) {
	using data_type = typename ProblemContextBase::scalar_type;

	if (pctx.x.cntg_step() == 1) {
		return &l2_norm_sve_kernel<data_type>;
	}

	return &l2_norm_fallback<data_type>;
}

template <typename StrategyTag, typename ProblemContextBase>
PERFLIBS_LINALG_INLINE
auto get_gemv_kernel(StrategyTag strat_tag,
	const problem_context<ProblemContextBase, sve_architecture_spec>& pctx, const kernel_inttype max_threads) {

	return get_gemv_sve_kernel(pctx, max_threads);
}

template<typename SpecType, typename ProblemContextBase>
PERFLIBS_LINALG_INLINE
bool matches_vector_size(const SpecType& spec, const problem_context<ProblemContextBase, sve_architecture_spec>&) {
	const auto vec_size_bits = static_cast<kernel_inttype>(vector_length_bytes_sve() * 8zu);
	return spec.vector_size_bits == 0 || spec.vector_size_bits == vec_size_bits;
}

} //namespace perflibs::linalg::spec

#endif //PERFLIBS_LINALG_ARCHITECTURES_SVE_GET_KERNEL_SPEC_HPP
