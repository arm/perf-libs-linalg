/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_PRE_HPP
#define PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_PRE_HPP

#include "framework/vector_length.hpp"
#include "framework/linalg_util.hpp"

#include <array>

namespace perflibs::linalg {

enum class gemm_kernel_types {
	interleaved
}; // enum class gemm_kernel_types


template<gemm_kernel_types WhichMatrix>
using gemm_kernel_types_constant = std::integral_constant<gemm_kernel_types, WhichMatrix>;

using interleaved_t = gemm_kernel_types_constant<gemm_kernel_types::interleaved>;

constexpr interleaved_t interleaved;


template<gemm_kernel_types KernelType, typename StrategyTag>
struct gemm_kernel_tag {
	extensions extension;

	gemm_kernel_tag(gemm_kernel_types_constant<KernelType>, const StrategyTag&, extensions extension=extensions::neon)
	:	extension { extension }
	{	}
}; //gemm_kernel_tag

template
<
	typename AType,
	typename BType = AType,
	typename CType = AType,
	typename ScalarType = perflibs::linalg::promote_t<AType, BType, CType>
>
using interleave_matmul_kernel = void(
	const AType *a,
	const BType *b,
	      CType *c,
	kernel_inttype k,
	kernel_inttype m,
	kernel_inttype n,
	kernel_inttype ldc,
	ScalarType alpha,
	ScalarType beta);

template<typename T>
struct matrix_interleave_spec {
	using data_type=T;
	decltype(&intval<0>) cntg_interleave;
	decltype(&intval<0>) strd_interleave;
	kernel_inttype       cntg_interleave_step;
	kernel_inttype       strd_interleave_step;
	matrix_requirement   matrix_req;
	kernel_inttype       split_factor;
	kernel_inttype       strd_unroll;
};

template <typename AType, typename BType = AType, typename CType = AType>
struct interleave_matmul_kernel_spec {
	/*
	 * TODO: remove value_type alias below
	 *
	 * We want to keep AType/BType/CType separate. Current implementations
	 * rely heavily on a single kernel value type, so we are keeping this
	 * alias for compatibility. Once existing implementations have been
	 * updated to support mixed precision, this alias should be removed.
	 */
	using a_data_type                            = AType;
	using b_data_type                            = BType;
	using c_data_type                            = CType;
	using value_type                             = promote_t<AType, BType, CType>;

	interleave_matmul_kernel<AType, BType, CType> *kernel;
	kernel_inttype                                 cntg_unroll;
	matrix_interleave_spec<AType>                  a_interleave_spec;
	matrix_interleave_spec<BType>                  b_interleave_spec;
	bool                                           row_major;
	value_support                                  apply_beta;
	extensions                                     extension;

	auto a_interleaved_rows() const { return a_interleave_spec.strd_interleave; }
	auto b_interleaved_rows() const { return b_interleave_spec.strd_interleave; }

}; // struct interleave_matmul_kernel_spec

template<typename GemmKernelSpecType, typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs = std::array<GemmKernelSpecType, 0> { };

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_PRE_HPP
