/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef GEMM_SMALL_FRAMEWORK_H
#define GEMM_SMALL_FRAMEWORK_H

#include "framework/linalg_util.hpp"
#include "perflibs_util.hpp"
#include "perflibs_assert.hpp"

#include <algorithm>

#define MAX_MAP_ENTRIES 100

namespace perflibs::gemm {

template<typename AType, typename BType = AType, typename CType = AType,
         typename ScalarType = perflibs::linalg::promote_t<AType, BType, CType>>
using gemm_small_f = void (*)(const kernel_inttype max_threads,
                              perflibs::linalg::perflibs_trans transa, perflibs::linalg::perflibs_trans transb, kernel_inttype m, kernel_inttype n,
                              kernel_inttype k, ScalarType alpha, const AType *a, kernel_inttype lda,
                              const BType *b, kernel_inttype ldb, ScalarType beta, CType *c,
                              kernel_inttype ldc);


// Function declaration for kernels
template<typename FloatType>
using kfunc_t = void (kernel_inttype, const FloatType *, kernel_inttype, const FloatType *, kernel_inttype, FloatType *, kernel_inttype, FloatType, FloatType);

// A simple map type to use in place of std::unordered_map, which requires libstdc++ linkage
template<typename FloatType>
struct simple_map {

	struct entry{
		int key;
		kfunc_t<FloatType> *value;
	} e[MAX_MAP_ENTRIES];

	kfunc_t<FloatType> * operator[] (int required_key) const {

		for (int i = 0; i < MAX_MAP_ENTRIES; i++) {
			if (e[i].key == required_key)
				return e[i].value;
		}

		// Abort if not found; this should never happen because
		// all required kernels are expected in the lookup table.
		PERFLIBS_ALWAYS_ASSERT(false, "kernel not found in table");

		// Escape to satisfy compiler
		return 0;
	}
};

// Capture different loop orderings. The values are used in kernel lookup.
typedef enum {
	nmk = 0, mnk = 1, nkm = 2*32768, knm = 2*32768 + 1, mkn = 4*32768, kmn = 4*32768 + 1
} l_order_t;

// Capture different transpose options. The values are used in kernel lookup.
typedef enum {
	NN = 0, NT = 18*32768, TN = 36*32768, TT = 54*32768
} trans_t;

// Accessor function for matrix C
kernel_inttype getCij(l_order_t order, kernel_inttype ldc, kernel_inttype dim1, kernel_inttype dim2, kernel_inttype dim3);

// This function should be implemented for each target, returning the kernels to be used
template<typename FloatType>
kfunc_t<FloatType> *get_kernel(trans_t trans_opt, l_order_t loop_o, kernel_inttype rblock_m, kernel_inttype rblock_n, kernel_inttype rblock_k, FloatType alpha, FloatType beta);

template<typename FloatType>
int get_kernel_key(trans_t trans_opt, l_order_t loop_o, uint64_t rblock_m, uint64_t rblock_n, uint64_t rblock_k, FloatType alpha, FloatType beta);

// Forward declare our struct since it is passed as an arg in the driver function
template<typename FloatType>
struct gemm_small_options;

// Driver function
template<typename FloatType>
using dfunc_t = void (int_type nota, int_type notb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
                      FloatType alpha, const FloatType *a, kernel_inttype lda,
                      const FloatType *b, kernel_inttype ldb, FloatType beta,
                      FloatType *c, kernel_inttype ldc, const struct gemm_small_options<FloatType> &options);

// Intrinsics function
template<typename FloatType>
using ifunc_t = void (const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const FloatType alpha,
                      const FloatType * restrict A, const kernel_inttype lda,
                      const FloatType * restrict B, const kernel_inttype ldb, const FloatType beta,
                      FloatType * restrict C, const kernel_inttype ldc);

// A group of functions and variables that vary depending on transpose options
template<typename FloatType>
struct gemm_small_options {
	// Accessor functions for matrices A and B
	decltype(&getCij) get_Aij, get_Bij;

	// The transpose options expressed in a nice way for kernel lookup
	trans_t trans_opt;

	// The loop ordering expressed in a nice way for kernel lookup
	l_order_t loop_o;

	// The gemm_unrolled (C++ intrinsics) function to fall back on
	ifunc_t<FloatType> *gemm_unrolled;

	// The gemm_small_target function - called recursively in the parallel case
	dfunc_t<FloatType> *gemm_small_recursive;

	// The register blocking factors for the kernel
	kernel_inttype mfact, nfact, kfact;

	// A function to return the cache blocking size for the 2nd loop
	kernel_inttype (*get_cblock_2)();

	// A function to return the cache blocking size for the 3rd (inner) loop
	kernel_inttype (*get_cblock_3)();

	// Functions that define the tile size used for the parallel decomposition
	kernel_inttype (*get_tile_m)(kernel_inttype, kernel_inttype);
	kernel_inttype (*get_tile_n)(kernel_inttype, kernel_inttype);
	kernel_inttype (*get_tile_k)(kernel_inttype, kernel_inttype);

	// The maximum number of threads to use
	int max_threads;

};

template<typename FloatType>
void gemm_small_driver(int_type nota, int_type notb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
	FloatType alpha, const FloatType *a, kernel_inttype lda,
	const FloatType *b, kernel_inttype ldb, FloatType beta,
	FloatType *c, kernel_inttype ldc, const struct gemm_small_options<FloatType> &options);

void sgemm_small(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, float alpha, float *A, kernel_inttype lda, float *B, kernel_inttype ldb, float *C, kernel_inttype ldc, float beta);

} // end namespaces

#endif
