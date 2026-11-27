/*
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_SPFFRT_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_SPFFRT_HPP

#include "packages/factorization/strategies.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include "spec/strategy_tag.hpp"
#include "framework/compute.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/packed_matrix_base.hpp"

namespace perflibs::linalg {

// Partial LDL^T without pivoting of the first ncolm columns of a symmetric (complex symmetric)
// matrix in lower packed storage, leaving the Schur complement in the trailing block; also
// implemented in oneMKL and AOCL. An exactly zero pivot is given the reciprocal zero.
// work and work2 are not referenced.

template<typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void spffrt_compute(AType *ap, const IntType n, const IntType ncolm,
                    const factorization::ldl_packed_output output) {
	if (n <= 0 || ncolm <= 0) return;

	auto pctx = spec::problem_context{
		factorization::ldl_factorization_no_pivot_packed{
			packed_matrix_base<AType>{ ap, kernel_inttype(n), kernel_inttype(n), PERFLIBS_LOWER },
			kernel_inttype(min(ncolm, n)), output },
		ArchitectureSpec{ machine::get_system_unsafe() }
	};
	factorization::compute(pctx);
}

template<typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void spffrt2(AType *ap, const IntType *n, const IntType *ncolm, AType *work, AType *work2) {
	(void)work; (void)work2;
	spffrt_compute<IntType, AType, ArchitectureSpec>(ap, *n, *ncolm, factorization::ldl_packed_output::inverse_diagonal);
}

template<typename IntType, typename AType, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void spffrtx(AType *ap, const IntType *n, const IntType *ncolm, AType *work, AType *work2) {
	(void)work; (void)work2;
	spffrt_compute<IntType, AType, ArchitectureSpec>(ap, *n, *ncolm, factorization::ldl_packed_output::diagonal);
}
} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_SPFFRT_HPP
