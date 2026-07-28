/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARFB_HPP
#define PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARFB_HPP

#include "packages/factorization/problem_context_bases.hpp"
#include "packages/factorization/strategies.hpp"

#include "framework/linalg_util.hpp"
#include "framework/compute.hpp"
#include "matrix/matrix.hpp"

namespace perflibs::linalg {

template<bool Paramcheck, typename IntType, typename Atype, typename ArchitectureSpec>
PERFLIBS_LINALG_INLINE
void larfb(const char* side, const char* trans, const char* direct, const char* storev,
           const IntType* m, const IntType* n, const IntType* k, const Atype* v,
           const IntType* ldv, const Atype* t, const IntType* ldt, Atype* c,
           const IntType* ldc, Atype* work, const IntType* ldwork) {

	// Netlib LARFB does not perform parameter checking.
	(void)Paramcheck;

	if (*m <= 0 || *n <= 0 || *k <= 0) {
		return;
	}

	const auto aside   = c_to_side(*side);
	const auto atrans  = c_to_trans(*trans);
	const auto adirect = c_to_direct(*direct);
	const auto astorev = c_to_storev(*storev);

	const bool is_lside = is_left(aside);
	const auto nq       = kernel_inttype(is_lside ? *m : *n);

	auto raw_v = astorev == PERFLIBS_COLUMNWISE
	           ? general_matrix { matrix_base { v, nq, kernel_inttype(*k), 1, kernel_inttype(*ldv) } }
	           : general_matrix { matrix_base { v, kernel_inttype(*k), nq, 1, kernel_inttype(*ldv) } };

	// Rowwise storage uses V^H in H = I - V T V^H, so take the adjoint view here.
	auto v_matrix = astorev == PERFLIBS_COLUMNWISE ? raw_v : adjoint(raw_v);

	auto t_matrix = triangular_matrix {
		adirect == PERFLIBS_FORWARD ? PERFLIBS_UPPER : PERFLIBS_LOWER,
		PERFLIBS_NOUNIT,
		matrix_base { t, kernel_inttype(*k), kernel_inttype(*k), 1, kernel_inttype(*ldt) }
	};

	auto c_matrix    = general_matrix { matrix_base { c, kernel_inttype(*m), kernel_inttype(*n), 1, kernel_inttype(*ldc) } };
	auto work_matrix = general_matrix { matrix_base { work, kernel_inttype(*ldwork), kernel_inttype(*k), 1, kernel_inttype(*ldwork) } };

	const auto layout = adirect == PERFLIBS_FORWARD ? factorization::block_reflector_layout::leading_unit_lower
	                                             : factorization::block_reflector_layout::trailing_unit_upper;

	auto reflector = factorization::block_reflector {
		to_const(v_matrix),
		is_trans(atrans) ? to_const(adjoint(t_matrix)) : to_const(t_matrix),
		layout
	};

	if (is_lside) {
		auto pctx = spec::problem_context {
			factorization::apply_block_reflector {
				reflector,
				c_matrix,
				work_matrix
			},
			ArchitectureSpec { machine::get_system_unsafe() }
		};

		factorization::compute(pctx);
	}
	else {
		auto pctx = spec::problem_context {
			factorization::apply_block_reflector {
				c_matrix,
				reflector,
				work_matrix
			},
			ArchitectureSpec { machine::get_system_unsafe() }
		};

		factorization::compute(pctx);
	}
}

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_FACTORIZATION_INTERFACES_LARFB_HPP
