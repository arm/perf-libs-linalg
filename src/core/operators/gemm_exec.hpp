/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_GEMM_EXEC_HPP
#define PERFLIBS_LINALG_OPERATORS_GEMM_EXEC_HPP

#include "packages/matmul/problem_context_bases.hpp"

#include "matrix/matrix.hpp"

#include "framework/compute_position.hpp"

namespace perflibs::linalg {

template<typename ArchitectureSpec>
class gemm_exec {
public:
	gemm_exec(ArchitectureSpec=ArchitectureSpec { }) { }

	template <typename AMatrixType, typename BMatrixType, typename CMatrixType, typename ScalarType>
	PERFLIBS_LINALG_INLINE
	void operator()(const AMatrixType& a, const BMatrixType& b, const CMatrixType& c, compute_position pos, ScalarType alpha, ScalarType beta) {
		if (empty(c)) return;

		if (pos.cntg != 0 || pos.iteration != 0)
			beta = one<ScalarType>;

		/*
		 * This code is copied from the now removed op_gemv and should be moved into its own operator at some point.
		 * additionally, the code should not predicate on which dimension is contiguous as this is error prone as tightly
		 * couples the operator to a specific netlib interface.
		 */
		const bool a_is_conj = is_cntg_contig(a) &&
			(
				( is_hermitian_matrix_v<AMatrixType>) ||
				( is_triangular_matrix_v<AMatrixType> && a.is_conj() ) ||
				( is_general_matrix_v<AMatrixType> && a.is_conj() )
			);

		spec::problem_context pctx {
			matmul::matmul3 {
				to_const( general_matrix { a.get_matrix_base(), a_is_conj } ),
				to_const(to_general_matrix(b)),
				         to_general_matrix(c),
				alpha, beta
			},
			ArchitectureSpec { }
		};

		compute(pctx);
	}
}; // class gemm_exec

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_OPERATORS_GEMM_EXEC_HPP
