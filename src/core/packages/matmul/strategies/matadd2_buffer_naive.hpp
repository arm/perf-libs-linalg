/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD2_BUFFER_NAIVE_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD2_BUFFER_NAIVE_HPP

#include "framework/alloc.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"
#include "matrix/operations.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class matadd2_buffer_naive {
public:
	static constexpr std::string_view name() { return "matadd2_buffer_naive"; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	bool operator() (const spec::problem_context<matmul::matadd2<Ts...>, ArchitectureSpec>& pctx) const {
		if (!this->can_compute(pctx)) return false;

		using value_type = typename spec::problem_context<matmul::matadd2<Ts...>, ArchitectureSpec>::c_matrix_type::value_type;

		auto *buffer = get_memory<value_type>(pctx.c.cntg() * pctx.c.strd());

		const bool strd_output   =   pctx.c.strd_step() == 1
		                         && (pctx.a.strd_step() == 1 || pctx.b.strd_step() == 1);

		const auto tmp_cntg_step = strd_output ? pctx.c.strd() : 1;
		const auto tmp_strd_step = strd_output ? 1 : pctx.c.cntg();

		auto tmp = general_matrix {
			matrix_base { buffer, pctx.c.cntg(), pctx.c.strd(), tmp_cntg_step, tmp_strd_step }
		};

		spec::problem_context pctx3_compute {
			matmul::matadd3 { pctx.a, pctx.b, tmp, pctx.alpha, pctx.beta },
			pctx.architecture_spec
		};

		matmul::compute(pctx3_compute);

		auto tmp_in = to_const(tmp);

		spec::problem_context pctx3_copy {
			matmul::matadd3 { tmp_in, tmp_in, pctx.c, one<value_type>, zero<value_type> },
			pctx.architecture_spec
		};

		matmul::compute(pctx3_copy);

		return_memory<value_type>(buffer);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename... Ts, typename ArchitectureSpec>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const spec::problem_context<matmul::matadd2<Ts...>, ArchitectureSpec>&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
};

} // namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_MATADD2_BUFFER_NAIVE_HPP
