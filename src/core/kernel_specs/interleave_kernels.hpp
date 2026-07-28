/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */


#ifndef PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP
#define PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP

#include "interleave_kernels_pre.hpp"

namespace perflibs::linalg {

template<kernel_inttype Flags, typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs<Flags,r16,r16,ArchitectureSpec> = std::array<interleave_kernel_spec<r16,r16>, 0> { };

template<typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs<kernel_inttype(interleave_flags::None),bf16,bf16,ArchitectureSpec> = std::array {
    interleave_kernel_spec { 4_ki,  4_ki, 1_ki, 4_ki, 0_ki, 0_ki, matrix_requirement::any, &tile_interleave<4_ki,  4_ki, bf16, bf16> },
    interleave_kernel_spec { 4_ki,  8_ki, 1_ki, 4_ki, 0_ki, 0_ki, matrix_requirement::any, &tile_interleave<4_ki,  8_ki, bf16, bf16> },
    interleave_kernel_spec { 4_ki,  6_ki, 1_ki, 4_ki, 0_ki, 0_ki, matrix_requirement::any, &tile_interleave<4_ki,  6_ki, bf16, bf16> },
    interleave_kernel_spec { 4_ki, 12_ki, 1_ki, 4_ki, 0_ki, 0_ki, matrix_requirement::any, &tile_interleave<4_ki, 12_ki, bf16, bf16> },

    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<2, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<4, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 5_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<5, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<6, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<8, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<9, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 10_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<10, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<12, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<16, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 20_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<20, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 24_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<24, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 32_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<32, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<2, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<4, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 5_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<5, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<6, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<8, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<9, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 10_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<10, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<12, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<16, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 20_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<20, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 24_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<24, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 32_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<32, kernel_inttype(interleave_flags::None), bf16, bf16, ArchitectureSpec> },
};

template<typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs<kernel_inttype(interleave_flags::None),r32,r32,ArchitectureSpec> = std::array {
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_s2> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_s4> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_s6> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_s8> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<9, kernel_inttype(interleave_flags::None), r32, r32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<12, kernel_inttype(interleave_flags::None), r32, r32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<16, kernel_inttype(interleave_flags::None), r32, r32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_s2> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_s4> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_s6> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_s8> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<9, kernel_inttype(interleave_flags::None), r32, r32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<12, kernel_inttype(interleave_flags::None), r32, r32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<16, kernel_inttype(interleave_flags::None), r32, r32, ArchitectureSpec> },
};

template<typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs<kernel_inttype(interleave_flags::None),r64,r64,ArchitectureSpec> = std::array {
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_d2> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_d4> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_d6> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_d8> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<9, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<16, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 20_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<20, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 32_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<32, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 64_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<64, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_d2> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_d4> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_d6> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_d8> },
    interleave_kernel_spec { 1_ki, 9_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<9, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 16_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<16, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 20_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_kernel_d20> },
    interleave_kernel_spec { 1_ki, 32_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<32, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 64_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<64, kernel_inttype(interleave_flags::None), r64, r64, ArchitectureSpec> },
};

template<typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs<kernel_inttype(interleave_flags::None),c32,c32,ArchitectureSpec> = std::array {
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_c2> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_c4> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_c6> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_c8> },
    interleave_kernel_spec { 1_ki, 10_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<10, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<12, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<2, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<4, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<6, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<8, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 10_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<10, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 12_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<12, kernel_inttype(interleave_flags::None), c32, c32, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 4_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_split_complex_ir4<r32>> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 4_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_split_complex_ir4<r32>> },
};

template<typename ArchitectureSpec>
inline
constexpr auto interleave_kernel_specs<kernel_inttype(interleave_flags::None),c64,c64,ArchitectureSpec> = std::array {
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_z2> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_z4> },
    interleave_kernel_spec { 1_ki, 5_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_cpp_interleave<5, kernel_inttype(interleave_flags::None), c64, c64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_z6> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_kernel_z8> },
    interleave_kernel_spec { 1_ki, 2_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<2, kernel_inttype(interleave_flags::None), c64, c64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<4, kernel_inttype(interleave_flags::None), c64, c64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 5_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<5, kernel_inttype(interleave_flags::None), c64, c64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 6_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<6, kernel_inttype(interleave_flags::None), c64, c64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 8_ki, 0_ki, 1_ki, 0_ki, 0_ki, matrix_requirement::strd_one, &t_cpp_interleave<8, kernel_inttype(interleave_flags::None), c64, c64, ArchitectureSpec> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 4_ki, 0_ki, matrix_requirement::cntg_one, &n_interleave_shim<&n_interleave_split_complex_ir4<r64>> },
    interleave_kernel_spec { 1_ki, 4_ki, 0_ki, 1_ki, 4_ki, 0_ki, matrix_requirement::strd_one, &t_interleave_shim<&t_interleave_split_complex_ir4<r64>> },
};


} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP
