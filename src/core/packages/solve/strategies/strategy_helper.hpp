/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_HELPER_HPP
#define PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_HELPER_HPP

#include "framework/trsm_kernels.hpp"

namespace perflibs::linalg::solve {

template<typename ScalarType>
class trsm_kernel_exec {
	trsm_kernel_t<ScalarType> *kernel_;

public:
	trsm_kernel_exec(trsm_kernel_t<ScalarType> *kernel)
	:	kernel_{ kernel }
	{	}

	template<typename AMatrixType, typename BMatrixType>
	PERFLIBS_LINALG_INLINE
	void operator() (const AMatrixType& a, BMatrixType& b) {
		kernel_(a.data(), a.cntg_step(), a.strd_step(), b.data(), b.cntg_step(), b.strd_step(), b.cntg(),
		        b.strd());
	}
}; // class trsm_kernel_exec

template<typename ScalarType>
class trsv_kernel_exec {
	trsv_kernel_t<ScalarType>  *kernel_;
	axpby_kernel_t<ScalarType> *axpby_kernel_;
	dot_kernel_t<ScalarType>   *dot_kernel_;

public:
	trsv_kernel_exec(trsv_kernel_t<ScalarType>  *kernel,
	                 axpby_kernel_t<ScalarType> *axpby_kernel,
	                 dot_kernel_t<ScalarType>   *dot_kernel)
	:	kernel_       { kernel       }
 	,	axpby_kernel_ { axpby_kernel }
	,	dot_kernel_   { dot_kernel   }
	{	}

	template <typename AMatrixType, typename BMatrixType>
	PERFLIBS_LINALG_INLINE
	void operator() (const AMatrixType& a, BMatrixType& b) {
		kernel_(a.data(), a.cntg_step(), a.strd_step(),
		        b.data(), b.cntg(),
		        axpby_kernel_, dot_kernel_);
	}
}; // class trsv_kernel_exec
} // namespace perflibs::linalg::solve

#endif // PERFLIBS_LINALG_SOLVE_STRATEGIES_TRIANGULAR_SOLVE_MATRIX_HELPER_HPP
