/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_KERNEL_EXEC_HPP
#define PERFLIBS_LINALG_OPERATORS_KERNEL_EXEC_HPP

#include "matrix/matrix.hpp"

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

namespace perflibs::linalg {

namespace {

/**
 * A stack terminal which takes expects all of the parameters for GEMM
 * of which A & B must be in the packed data format expected by the provided kernel
 *
 * Also handles whether or not the provided kernel is row or col major
 */
template<typename Kernel>
class kernel_exec {
	Kernel kernel_;
	value_support apply_beta_;
public:
	kernel_exec(Kernel kernel, value_support apply_beta)
	:	kernel_    { kernel     }
	,	apply_beta_{ apply_beta }
	{	};

	template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename Scalar>
	inline
	void operator()(const MatrixTypeA& a, const MatrixTypeB& b, MatrixTypeC& c, compute_position compute_position, Scalar alpha, Scalar beta) {
		PERFLIBS_ASSERT(is_cntg_contig(c), "c must be contiguous in the cntg dimension");
		PERFLIBS_ASSERT(c.is_physical(), "C must be physical");

		if (compute_position.cntg != 0 || compute_position.iteration != 0) {
			beta = one<Scalar>;
		}
		/*
		 * We need to geset C if beta == 0.0 to ensure that any NaN's in the input are removed
		 */
		else if(apply_beta_!=value_support::all) {
			// Some kernels need external support to handle beta=0 cases
			if (beta == zero<Scalar>) {
				set(beta, c);
				beta = one<Scalar>;
			}
			/*
			* If the kernel does not apply Beta then we should do that for it!
			* This option is here to allow for prototyping kernels to be much easier
			*/
			else if (apply_beta_==value_support::one) {
				scale(beta, c);
			}
		}

		kernel_(a.data(), b.data(), c.data(), max(a.cntg(), b.cntg()), c.cntg(), c.strd(), c.strd_step(), alpha, beta);
	}
}; //class kernel_exec



}
} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_OPERATORS_KERNEL_EXEC_HPP
