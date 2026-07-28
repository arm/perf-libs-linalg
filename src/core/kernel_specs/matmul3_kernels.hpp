/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */


#ifndef PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_HPP
#define PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_HPP

#include "matmul3_kernels_pre.hpp"

#include <array>

extern "C" {
	using perflibs::bf16;
	using perflibs::r16;
	using perflibs::r32;
	using perflibs::r64;
	using perflibs::c32;
	using perflibs::c64;

	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, bf16> a64_interleaved_nomerge_bf16fp32bf16_mmla_4x12;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, bf16> a64_interleaved_nomerge_bf16fp32bf16_mmla_4x8;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, bf16> a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, bf16> bgemm_fp32_fmla_acle_bf16exten;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, bf16> bgemm_fp32_fmla_acle_nobf16exten;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, r32> a64_interleaved_nomerge_bf16fp32_mmla_4x12;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, r32> a64_interleaved_nomerge_bf16fp32_mmla_4x8;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, r32> a64_interleaved_nomerge_bf16fp32_mmla_6x8;
	perflibs::linalg::interleave_matmul_kernel<bf16, bf16, r32> sbgemm_fp32_fmla_acle;
	perflibs::linalg::interleave_matmul_kernel<c32, c32, c32> cgemm_large_kernel_TN_nmk;
	perflibs::linalg::interleave_matmul_kernel<c32, c32, c32> cgemm_sve_big;
	perflibs::linalg::interleave_matmul_kernel<c64, c64, c64> zgemm_sve_big;
	perflibs::linalg::interleave_matmul_kernel<c64, c64, c64> zgemm_vanilla_big_TN;
	perflibs::linalg::interleave_matmul_kernel<r16, r16, r16> hgemm_8_2_kernel;
	perflibs::linalg::interleave_matmul_kernel<r16, r16, r16> hgemm_sve_big;
	perflibs::linalg::interleave_matmul_kernel<r32, r32, r32> sgemm_sve_big;
	perflibs::linalg::interleave_matmul_kernel<r32, r32, r32> sgemm_vanilla_big;
	perflibs::linalg::interleave_matmul_kernel<r32, r32, r32> sgemm_vanilla_big_acle;
	perflibs::linalg::interleave_matmul_kernel<r32, r32, r32> sgemm_vanilla_big_no_prefetch;
	perflibs::linalg::interleave_matmul_kernel<r32, r32, r32> sme2_interleaved_nomerge_fp32_mopa_2VLx2VL;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> dgemm_sve_big;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> dgemm_tx2_big;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> dgemm_vanilla_big;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> dgemm_vanilla_big_no_prefetch;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> sme2_interleaved_nomerge_fp64_mopa_2VLx2VL;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> sme2_interleaved_nomerge_fp64_mopa_2VLx4VL;
	perflibs::linalg::interleave_matmul_kernel<r64, r64, r64> sme2_interleaved_nomerge_fp64_mopa_4VLx2VL;
} // extern "C"

namespace perflibs::linalg {

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, r32>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<bf16, bf16, r32> { &a64_interleaved_nomerge_bf16fp32_mmla_4x8, 8_ki, matrix_interleave_spec<bf16> { intval<4>, intval<8>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<4>, intval<4>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, r32> { &a64_interleaved_nomerge_bf16fp32_mmla_6x8, 8_ki, matrix_interleave_spec<bf16> { intval<4>, intval<8>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<4>, intval<6>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, r32> { &a64_interleaved_nomerge_bf16fp32_mmla_4x12, 8_ki, matrix_interleave_spec<bf16> { intval<4>, intval<12>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<4>, intval<4>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, r32> { &sbgemm_fp32_fmla_acle, 1_ki, matrix_interleave_spec<bf16> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon }
};
constexpr std::size_t a64_interleaved_nomerge_bf16fp32_mmla_4x8_idx = 0zu, a64_interleaved_nomerge_bf16fp32_mmla_6x8_idx = 1zu, a64_interleaved_nomerge_bf16fp32_mmla_4x12_idx = 2zu, sbgemm_fp32_fmla_acle_idx = 3zu;

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<bf16, bf16, bf16>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<bf16, bf16, bf16> { &a64_interleaved_nomerge_bf16fp32bf16_mmla_4x8, 8_ki, matrix_interleave_spec<bf16> { intval<4>, intval<8>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<4>, intval<4>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, bf16> { &a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8, 8_ki, matrix_interleave_spec<bf16> { intval<4>, intval<8>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<4>, intval<6>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, bf16> { &a64_interleaved_nomerge_bf16fp32bf16_mmla_4x12, 8_ki, matrix_interleave_spec<bf16> { intval<4>, intval<12>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<4>, intval<4>, 1_ki, 4_ki, matrix_requirement::cntg_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, bf16> { &bgemm_fp32_fmla_acle_nobf16exten, 1_ki, matrix_interleave_spec<bf16> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<bf16, bf16, bf16> { &bgemm_fp32_fmla_acle_bf16exten, 1_ki, matrix_interleave_spec<bf16> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<bf16> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon }
};
constexpr std::size_t a64_interleaved_nomerge_bf16fp32bf16_mmla_4x8_idx = 0zu, a64_interleaved_nomerge_bf16fp32bf16_mmla_6x8_idx = 1zu, a64_interleaved_nomerge_bf16fp32bf16_mmla_4x12_idx = 2zu, bgemm_fp32_fmla_acle_nobf16exten_idx = 3zu, bgemm_fp32_fmla_acle_bf16exten_idx = 4zu;

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<r16, r16, r16>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<r16, r16, r16> { &hgemm_8_2_kernel, 4_ki, matrix_interleave_spec<r16> { intval<1>, intval<24>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r16> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::not_zero, extensions::neon },
	interleave_matmul_kernel_spec<r16, r16, r16> { &hgemm_sve_big, 1_ki, matrix_interleave_spec<r16> { intval<1>, vl<extensions::sve, r16, 1>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r16> { intval<1>, intval<24>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sve }
};
constexpr std::size_t hgemm_8_2_kernel_idx = 0zu, hgemm_sve_big_idx = 1zu;

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<r32, r32, r32>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<r32, r32, r32> { &sme2_interleaved_nomerge_fp32_mopa_2VLx2VL, 1_ki, matrix_interleave_spec<r32> { intval<1>, vl<extensions::sme2, r32, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r32> { intval<1>, vl<extensions::sme2, r32, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sme2 },
	interleave_matmul_kernel_spec<r32, r32, r32> { &sgemm_vanilla_big, 1_ki, matrix_interleave_spec<r32> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r32> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<r32, r32, r32> { &sgemm_vanilla_big_acle, 1_ki, matrix_interleave_spec<r32> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r32> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<r32, r32, r32> { &sgemm_vanilla_big_no_prefetch, 1_ki, matrix_interleave_spec<r32> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r32> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<r32, r32, r32> { &sgemm_sve_big, 1_ki, matrix_interleave_spec<r32> { intval<1>, vl<extensions::sve, r32, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r32> { intval<1>, intval<12>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sve }
};
constexpr std::size_t sme2_interleaved_nomerge_fp32_mopa_2VLx2VL_idx = 0zu, sgemm_vanilla_big_idx = 1zu, sgemm_vanilla_big_acle_idx = 2zu, sgemm_vanilla_big_no_prefetch_idx = 3zu, sgemm_sve_big_idx = 4zu;

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<r64, r64, r64>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<r64, r64, r64> { &sme2_interleaved_nomerge_fp64_mopa_2VLx2VL, 1_ki, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sme2, r64, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sme2, r64, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sme2 },
	interleave_matmul_kernel_spec<r64, r64, r64> { &sme2_interleaved_nomerge_fp64_mopa_2VLx4VL, 1_ki, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sme2, r64, 4>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sme2, r64, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sme2 },
	interleave_matmul_kernel_spec<r64, r64, r64> { &sme2_interleaved_nomerge_fp64_mopa_4VLx2VL, 1_ki, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sme2, r64, 2>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sme2, r64, 4>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sme2 },
	interleave_matmul_kernel_spec<r64, r64, r64> { &dgemm_vanilla_big, 1_ki, matrix_interleave_spec<r64> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, intval<6>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<r64, r64, r64> { &dgemm_tx2_big, 1_ki, matrix_interleave_spec<r64> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, intval<6>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<r64, r64, r64> { &dgemm_vanilla_big_no_prefetch, 1_ki, matrix_interleave_spec<r64> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, intval<6>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::neon },
	interleave_matmul_kernel_spec<r64, r64, r64> { &dgemm_sve_big, 1_ki, matrix_interleave_spec<r64> { intval<1>, vl<extensions::sve, r64, 3>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, matrix_interleave_spec<r64> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 0_ki, 1_ki }, false, value_support::all, extensions::sve }
};
constexpr std::size_t sme2_interleaved_nomerge_fp64_mopa_2VLx2VL_idx = 0zu, sme2_interleaved_nomerge_fp64_mopa_2VLx4VL_idx = 1zu, sme2_interleaved_nomerge_fp64_mopa_4VLx2VL_idx = 2zu, dgemm_vanilla_big_idx = 3zu, dgemm_tx2_big_idx = 4zu, dgemm_vanilla_big_no_prefetch_idx = 5zu, dgemm_sve_big_idx = 6zu;

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<c32, c32, c32>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<c32, c32, c32> { &cgemm_large_kernel_TN_nmk, 1_ki, matrix_interleave_spec<c32> { intval<1>, intval<4>, 1_ki, 1_ki, matrix_requirement::strd_one, 4_ki, 1_ki }, matrix_interleave_spec<c32> { intval<1>, intval<10>, 1_ki, 1_ki, matrix_requirement::strd_one, 1_ki, 1_ki }, false, value_support::one, extensions::neon },
	interleave_matmul_kernel_spec<c32, c32, c32> { &cgemm_sve_big, 1_ki, matrix_interleave_spec<c32> { intval<1>, vl<extensions::sve, c32, 3>, 1_ki, 1_ki, matrix_requirement::strd_one, 1_ki, 1_ki }, matrix_interleave_spec<c32> { intval<1>, intval<8>, 1_ki, 1_ki, matrix_requirement::strd_one, 1_ki, 1_ki }, false, value_support::all, extensions::sve }
};
constexpr std::size_t cgemm_large_kernel_TN_nmk_idx = 0zu, cgemm_sve_big_idx = 1zu;

template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<c64, c64, c64>, ArchitectureSpec> = std::array {
	interleave_matmul_kernel_spec<c64, c64, c64> { &zgemm_vanilla_big_TN, 1_ki, matrix_interleave_spec<c64> { intval<1>, intval<4>, 1_ki, 1_ki, matrix_requirement::strd_one, 4_ki, 1_ki }, matrix_interleave_spec<c64> { intval<1>, intval<4>, 1_ki, 1_ki, matrix_requirement::strd_one, 1_ki, 1_ki }, false, value_support::one, extensions::neon },
	interleave_matmul_kernel_spec<c64, c64, c64> { &zgemm_sve_big, 1_ki, matrix_interleave_spec<c64> { intval<1>, vl<extensions::sve, c64, 4>, 1_ki, 1_ki, matrix_requirement::strd_one, 1_ki, 1_ki }, matrix_interleave_spec<c64> { intval<1>, intval<5>, 1_ki, 1_ki, matrix_requirement::strd_one, 1_ki, 1_ki }, false, value_support::all, extensions::sve }
};
constexpr std::size_t zgemm_vanilla_big_TN_idx = 0zu, zgemm_sve_big_idx = 1zu;

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_HPP

