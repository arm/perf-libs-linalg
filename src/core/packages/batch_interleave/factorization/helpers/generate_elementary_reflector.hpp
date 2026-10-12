/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_GENERATE_ELEMENTARY_REFLECTOR_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_GENERATE_ELEMENTARY_REFLECTOR_HPP

#include "framework/linalg_util.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "matrix/matrix.hpp"
#include "packages/factorization/fwd.hpp"
#include "spec/problem_context.hpp"

namespace perflibs::linalg::batch_interleave::factorization {

/** Generate an elementary reflector independently in each interleaved lane. */
template<typename T, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void generate_elementary_reflector(
	const general_interleave_batch_matrix<T>& reflector,
	const general_interleave_batch_matrix<T>& tau,
	const ArchitectureSpec& architecture_spec) {
	PERFLIBS_ASSERT(
		reflector.strd() == 1 && tau.cntg() == 1 && tau.strd() == 1,
		"elementary-reflector matrices have incompatible dimensions");
	PERFLIBS_ASSERT(
		reflector.interleave_count() == tau.interleave_count(),
		"elementary-reflector matrices have incompatible interleave counts");
	PERFLIBS_ASSERT(
		reflector.cntg_step() >= reflector.interleave_count(),
		"elementary-reflector matrices have incompatible strides");

	const auto ninter = reflector.interleave_count();

	if (ninter == 0) {
		return;
	}

	const auto n = reflector.cntg();

	if (n <= 1) {
		tau(0, 0, write) = scalar { zero<T> };
		return;
	}

	const auto x_batch = reflector.sub_matrix(1, n - 1, 0, 1);

	for (kernel_inttype lane = 0; lane < ninter; lane++) {
		const auto x = general_matrix { x_batch.extract(lane) };
		const auto reflector_pctx = spec::problem_context {
			linalg::factorization::generate_reflector {
				reflector(0, 0, write)(lane),
				x,
				tau(0, 0, write)(lane)
			},
			architecture_spec
		};
		linalg::factorization::compute(reflector_pctx);
	}
}

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_HELPERS_GENERATE_ELEMENTARY_REFLECTOR_HPP
