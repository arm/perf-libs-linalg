/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_INTERLEAVE_KERNELS_PRE_HPP
#define PERFLIBS_LINALG_FRAMEWORK_INTERLEAVE_KERNELS_PRE_HPP

#include "framework/interleave_fallback_kernel_fwd.hpp"
#include "framework/interleave_split_complex_kernels.hpp"
#include "framework/tile_interleave.hpp"
#include "framework/linalg_util.hpp"
#include "framework/vector_length.hpp"
#include "framework/which.hpp"

#include "perflibs_assert.hpp"

#include <array>
#include <iterator>
#include <algorithm>

namespace perflibs::linalg {

template<typename SrcDataType, typename DstDataType>
using interleave_kernel_legacy_f = void (
	std::size_t src_cntg, std::size_t src_strd, const SrcDataType *src, std::size_t src_strd_step,
	std::size_t dst_cntg, std::size_t dst_strd,       DstDataType *dst, std::size_t dst_strd_step);

template<typename SrcDataType, typename DstDataType>
using interleave_kernel_f = void (
	std::size_t src_cntg, std::size_t src_strd, const SrcDataType *src, std::size_t src_cntg_step, std::size_t src_strd_step,
	std::size_t dst_cntg, std::size_t dst_strd,       DstDataType *dst, std::size_t dst_cntg_step,
	kernel_inttype src_submat_cntg, kernel_inttype src_submat_strd);


} // namespace perflibs::linalg

extern "C" {
using perflibs::r16;
using perflibs::r32;
using perflibs::r64;
using perflibs::c32;
using perflibs::c64;

perflibs::linalg::interleave_kernel_legacy_f<r32, r32> n_interleave_kernel_s2;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> n_interleave_kernel_s4;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> n_interleave_kernel_s6;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> n_interleave_kernel_s8;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> n_interleave_kernel_s20;

perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_s2;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_s4;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_s6;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_s8;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_s20;

perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_zero_pad_s2;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_zero_pad_s4;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_zero_pad_s6;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_zero_pad_s8;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> t_interleave_kernel_zero_pad_s20;

perflibs::linalg::interleave_kernel_legacy_f<r64, r64> n_interleave_kernel_d2;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> n_interleave_kernel_d4;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> n_interleave_kernel_d6;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> n_interleave_kernel_d8;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> n_interleave_kernel_d20;

perflibs::linalg::interleave_kernel_legacy_f<r64, r64> t_interleave_kernel_d2;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> t_interleave_kernel_d4;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> t_interleave_kernel_d6;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> t_interleave_kernel_d8;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> t_interleave_kernel_d20;

perflibs::linalg::interleave_kernel_legacy_f<c32, c32> n_interleave_kernel_c2;
perflibs::linalg::interleave_kernel_legacy_f<c32, c32> n_interleave_kernel_c4;
perflibs::linalg::interleave_kernel_legacy_f<c32, c32> n_interleave_kernel_c6;
perflibs::linalg::interleave_kernel_legacy_f<c32, c32> n_interleave_kernel_c8;

perflibs::linalg::interleave_kernel_legacy_f<c64, c64> n_interleave_kernel_z2;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64> n_interleave_kernel_z4;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64> n_interleave_kernel_z6;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64> n_interleave_kernel_z8;


//VL SVE Kernels
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> tran_interleave_2vl_sve_kernel_s;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> tran_interleave_2vl_sve_kernel_d;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> tran_interleave_3vl_sve_kernel_d;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64> tran_interleave_4vl_sve_kernel_z;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32>      interleave_2vl_sve_kernel_s;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64>      interleave_2vl_sve_kernel_d;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64>      interleave_3vl_sve_kernel_d;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64>      interleave_4vl_sve_kernel_z;

//static SVE Kernels
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> tran_interleave_9s_sve_512_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> tran_interleave_12s_sve_512_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> tran_interleave_8d_sve_512_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> tran_interleave_9d_sve_512_kernel;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64> tran_interleave_5z_sve_512_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32> tran_interleave_12s_sve_256_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64> tran_interleave_8d_sve_256_kernel;
perflibs::linalg::interleave_kernel_legacy_f<c64, c64> tran_interleave_5z_sve_256_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r32, r32>      interleave_9s_sve_512_kernel;
perflibs::linalg::interleave_kernel_legacy_f<r64, r64>      interleave_9d_sve_512_kernel;
} //extern "C"

template<auto InterleaveFunc>
PERFLIBS_LINALG_INLINE
void r64_as_c32(
	std::size_t src_cntg, std::size_t src_strd, const c32 *src, std::size_t src_step,
	std::size_t dst_cntg, std::size_t dst_strd,       c32 *dst, std::size_t dst_step) {

	InterleaveFunc(
		src_cntg, src_strd, reinterpret_cast<const r64*>(src), src_step,
		dst_cntg, dst_strd, reinterpret_cast<      r64*>(dst), dst_step);
}

constexpr auto      interleave_2vl_sve_kernel_c32 = r64_as_c32<     &interleave_2vl_sve_kernel_d>;
constexpr auto      interleave_3vl_sve_kernel_c32 = r64_as_c32<     interleave_3vl_sve_kernel_d>;
constexpr auto tran_interleave_2vl_sve_kernel_c32 = r64_as_c32<tran_interleave_2vl_sve_kernel_d>;
constexpr auto tran_interleave_3vl_sve_kernel_c32 = r64_as_c32<tran_interleave_3vl_sve_kernel_d>;

namespace perflibs::linalg {

template<typename FuncType>
struct decode_interleave_kernel_f { };

template<typename SrcType, typename DstType>
struct decode_interleave_kernel_f<interleave_kernel_legacy_f<SrcType, DstType>*> {
	using src_type = SrcType;
	using dst_type = DstType;
};

template<typename FuncType>
using interleave_kernel_f_src_t = typename decode_interleave_kernel_f<FuncType>::src_type;

template<typename FuncType>
using interleave_kernel_f_dst_t = typename decode_interleave_kernel_f<FuncType>::dst_type;

template<auto NInterleaveFunc>
PERFLIBS_LINALG_INLINE
void n_interleave_shim(
		std::size_t src_cntg, std::size_t src_strd, const interleave_kernel_f_src_t<decltype(NInterleaveFunc)> *src, std::size_t src_cntg_step, std::size_t src_strd_step,
		std::size_t dst_cntg, std::size_t dst_strd,       interleave_kernel_f_dst_t<decltype(NInterleaveFunc)> *dst,                            std::size_t dst_strd_step,
		kernel_inttype src_submat_cntg, kernel_inttype src_submat_strd) {

	PERFLIBS_ASSERT(src_cntg_step == 1, "strd_step must be 1")

	return NInterleaveFunc(
		src_cntg, src_strd, src, src_strd_step,
		dst_cntg, dst_strd, dst, dst_strd_step);
}

template<auto TInterleaveFunc>
PERFLIBS_LINALG_INLINE
void t_interleave_shim(
		std::size_t src_cntg, std::size_t src_strd, const interleave_kernel_f_src_t<decltype(TInterleaveFunc)> *src, std::size_t src_cntg_step, std::size_t src_strd_step,
		std::size_t dst_cntg, std::size_t dst_strd,       interleave_kernel_f_dst_t<decltype(TInterleaveFunc)> *dst,                            std::size_t dst_strd_step,
		kernel_inttype src_submat_cntg, kernel_inttype src_submat_strd) {

	PERFLIBS_ASSERT(src_strd_step == 1, "strd_step must be 1")

	return TInterleaveFunc(
		src_strd, src_cntg, src, src_cntg_step,
		dst_cntg, dst_strd, dst, dst_strd_step);
}

template<auto InterleaveExpr, auto NInterleaveFunc>
PERFLIBS_LINALG_INLINE
void n_interleave_physical_packed_dst_shim(
		std::size_t src_cntg, std::size_t src_strd, const interleave_kernel_f_src_t<decltype(NInterleaveFunc)> *src, std::size_t src_cntg_step, std::size_t src_strd_step,
		std::size_t dst_cntg, std::size_t dst_strd,       interleave_kernel_f_dst_t<decltype(NInterleaveFunc)> *dst,                            std::size_t dst_strd_step,
		kernel_inttype src_submat_cntg, kernel_inttype src_submat_strd) {

	PERFLIBS_ASSERT(src_cntg_step == 1, "strd_step must be 1")
	const auto interleave = InterleaveExpr();

	return NInterleaveFunc(
		src_cntg, src_strd, src, src_strd_step,
		dst_cntg * interleave, iround_div(dst_strd, interleave), dst, dst_strd_step);
}

template<auto InterleaveExpr, auto TInterleaveFunc>
PERFLIBS_LINALG_INLINE
void t_interleave_physical_packed_dst_shim(
		std::size_t src_cntg, std::size_t src_strd, const interleave_kernel_f_src_t<decltype(TInterleaveFunc)> *src, std::size_t src_cntg_step, std::size_t src_strd_step,
		std::size_t dst_cntg, std::size_t dst_strd,       interleave_kernel_f_dst_t<decltype(TInterleaveFunc)> *dst,                            std::size_t dst_strd_step,
		kernel_inttype src_submat_cntg, kernel_inttype src_submat_strd) {

	PERFLIBS_ASSERT(src_strd_step == 1, "strd_step must be 1")
	const auto interleave = InterleaveExpr();

	return TInterleaveFunc(
		src_strd, src_cntg, src, src_cntg_step,
		dst_cntg * interleave, iround_div(dst_strd, interleave), dst, dst_strd_step);
}

template<typename InterleaveKernel, typename SrcMatrixType, typename DstMatrixType>
auto invoke(const InterleaveKernel kernel, const SrcMatrixType& src, const DstMatrixType& dst) {
	return kernel(
		src.cntg(), src.strd(), src.data(), src.cntg_step(), src.strd_step(),
		dst.cntg(), dst.strd(), dst.data(),                  dst.strd_tile_step(),
		src.absolute_cntg(), src.absolute_strd());
}

template<typename SrcDataType, typename DstDataType>
struct interleave_kernel_spec {
	using src_data_type = SrcDataType;
	using dst_data_type = DstDataType;

	vector_length_value                            cntg_interleave = 1_ki;
	vector_length_value                            strd_interleave;             // renamed from 'interleave'
	kernel_inttype                                 cntg_interleave_step = 0;
	kernel_inttype                                 strd_interleave_step = 1;
	kernel_inttype                                 split_factor;

	/*
	 * This is a strong requirement, ie 512 mean it can only run on a 512 machine
	 * neon or VL kernels should set this to 0 so it can be pick up by any machine
	 */
	kernel_inttype                                 vector_size_bits;
	matrix_requirement                             matrix_req;
	interleave_kernel_f<SrcDataType, DstDataType> *kernel;
};

template<kernel_inttype Flags, typename SrcDataType, typename DstDataType>
struct interleave_kernel_specs_tag { };

template<kernel_inttype Flags, typename SrcDataType, typename DstDataType, typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs = std::array {
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<2, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<4, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 5_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<5, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<6, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<8, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<9, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<12, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<16, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<2, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<4, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 5_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<5, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<6, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<8, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<9, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<12, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<16, Flags, SrcDataType, DstDataType, ArchitectureSpec> },
};

template<kernel_inttype Flags, typename ProblemContext, typename System, typename... Types>
PERFLIBS_LINALG_INLINE
auto get_specs(interleave_kernel_specs_tag<Flags, Types...>, const ProblemContext&, System) {
	return interleave_kernel_specs<Flags, Types..., typename ProblemContext::architecture_spec_type>;
}

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_matrix(a_matrix_t, const ProblemContext& pctx) { return pctx.a; }

template<typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto get_matrix(b_matrix_t, const ProblemContext& pctx) { return pctx.b; }

template<typename SpecType, typename ProblemContext>
PERFLIBS_LINALG_INLINE
bool matches_vector_size(const SpecType&, const ProblemContext&) {
	return true;
}

template<typename InterleaveKernelSpecsType, typename SrcMatrixType, typename DstDataType, typename ProblemContext>
PERFLIBS_LINALG_INLINE
auto find_interleave_spec(const InterleaveKernelSpecsType& specs, const SrcMatrixType& src, const matrix_interleave_spec<DstDataType>& dst_mat_spec, const ProblemContext& pctx) {
	using spec_type = typename InterleaveKernelSpecsType::value_type;
	/*
	 * There is probably better logic to do this.
	 * It is easy to figure out if a matrix needs transpose or not if both dimensions have size
	 * but when it is a vector then it is possible for both strides to be 1 and we need to make sure
	 * we handle that as a transposing case
	 *
	 * Also this logic poorly handles the case where the input is vector and could possibly be packed with the "wrong" kernels
	 */
	const auto matrix_req = [&] {
		if(src.cntg_step() == 1 && src.strd_step() >  1 && src.cntg() > 1) return matrix_requirement::cntg_one;
		if(src.strd_step() == 1 && src.cntg_step() >= 1                  ) return matrix_requirement::strd_one;

		return matrix_requirement::any;
	}();

	const auto it = std::find_if(std::begin(specs), std::end(specs), [&](const auto& spec) {
			return spec.cntg_interleave() == dst_mat_spec.cntg_interleave()
			   &&  spec.strd_interleave() == dst_mat_spec.strd_interleave()
			   && ( spec.cntg_interleave() == 1 || spec.cntg_interleave_step == dst_mat_spec.cntg_interleave_step )
			   && ( spec.strd_interleave() == 1 || spec.strd_interleave_step == dst_mat_spec.strd_interleave_step )
			   &&  spec.split_factor    == dst_mat_spec.split_factor
			   &&  ( spec.matrix_req == matrix_requirement::any || spec.matrix_req == matrix_req )
			   &&  matches_vector_size(spec, pctx);
		}
	);

	return it != std::end(specs) ? *it : spec_type {
		dst_mat_spec.cntg_interleave(),
		dst_mat_spec.strd_interleave(),
		dst_mat_spec.cntg_interleave_step,
		dst_mat_spec.strd_interleave_step,
		dst_mat_spec.split_factor,
		0_ki,
		matrix_requirement::any,
		nullptr //will force driver to use fallback
	};
}

template<typename SrcMatrixType, typename DstDataType, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_general_interleave_spec(const SrcMatrixType& src, const matrix_interleave_spec<DstDataType>& dst_matrix_spec, const ProblemContext& pctx, System system) {
	using src_data_type = std::remove_cv_t< typename SrcMatrixType::value_type>;

	if constexpr(is_complex_v<src_data_type>) {
		if(src.is_conj()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::IsConj), src_data_type, DstDataType> { }, pctx, system), src, dst_matrix_spec, pctx);
	}
	return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::None), src_data_type, DstDataType> { }, pctx, system), src, dst_matrix_spec, pctx);
}

template<typename SrcMatrixType, typename DstDataType, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_physical_interleave_spec(const SrcMatrixType& src, const matrix_interleave_spec<DstDataType>& dst_matrix_spec, const ProblemContext& pctx, System system) {
	using src_data_type = std::remove_cv_t<typename SrcMatrixType::value_type>;

	if constexpr (is_general_matrix_v<SrcMatrixType>) {
		return get_general_interleave_spec(src, dst_matrix_spec, pctx, system);
	}
	else if constexpr (is_triangular_matrix_v<SrcMatrixType>) {
		if(src.is_unit()) {
			if(src.is_lower()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::DiagVirt | interleave_flags::DiagUnit | interleave_flags::UpperVirt | interleave_flags::VirtZero), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
			else               return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::DiagVirt | interleave_flags::DiagUnit | interleave_flags::LowerVirt | interleave_flags::VirtZero), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
		}
		else {
			if(src.is_lower()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::UpperVirt | interleave_flags::VirtZero), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
			else               return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::LowerVirt | interleave_flags::VirtZero), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
		}
	}
	else if constexpr (is_symmetric_matrix_v<SrcMatrixType>) {
		if(src.is_lower()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::UpperVirt | interleave_flags::VirtZero), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
		else               return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::LowerVirt | interleave_flags::VirtZero), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
	}
	else if constexpr (is_hermitian_matrix_v<SrcMatrixType>) {
		if(src.is_lower()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::UpperVirt | interleave_flags::DiagVirt | interleave_flags::DiagReal), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
		else               return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::LowerVirt | interleave_flags::DiagVirt | interleave_flags::DiagReal), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
	}
}

template<typename SrcMatrixType, typename DstDataType, typename ProblemContext, typename System>
PERFLIBS_LINALG_INLINE
auto get_virtual_interleave_spec(const SrcMatrixType& src, const matrix_interleave_spec<DstDataType>& dst_matrix_spec, const ProblemContext& pctx, System system) {
	using src_data_type = std::remove_cv_t<typename SrcMatrixType::value_type>;

	if constexpr (is_symmetric_matrix_v<SrcMatrixType>) {
		if(src.is_lower()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::UpperVirt | interleave_flags::DiagVirt), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
		else               return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::LowerVirt | interleave_flags::DiagVirt), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
	}
	else if constexpr (is_hermitian_matrix_v<SrcMatrixType>) {
		if(src.is_lower()) return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::UpperVirt | interleave_flags::DiagVirt | interleave_flags::IsConj), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
		else               return find_interleave_spec(get_specs(interleave_kernel_specs_tag<kernel_inttype(interleave_flags::LowerVirt | interleave_flags::DiagVirt | interleave_flags::IsConj), src_data_type, DstDataType> {}, pctx, system), src, dst_matrix_spec, pctx);
	}
}

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_INTERLEAVE_KERNELS_PRE_HPP
