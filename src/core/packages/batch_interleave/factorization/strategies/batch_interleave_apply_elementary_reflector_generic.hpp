/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_APPLY_ELEMENTARY_REFLECTOR_GENERIC_HPP
#define PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_APPLY_ELEMENTARY_REFLECTOR_GENERIC_HPP

#include "framework/linalg_util.hpp"
#include "matrix/adaptors.hpp"
#include "matrix/interleave_batch_matrix.hpp"
#include "pod_vector.hpp"
#include "spec/problem_context.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>
#include <type_traits>

namespace perflibs::linalg::batch_interleave::factorization {

/**
 * Apply H = I - tau * v * v**T to an interleaved batch of matrices.
 *
 * This strategy is specialized for the reflectors produced during real QR:
 * each interleaved lane has its own tau and the first element of v is
 * implicitly one.
 */
class batch_interleave_apply_elementary_reflector_generic {
public:
	static constexpr std::string_view name() { return "batch_interleave_apply_elementary_reflector_generic"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<batch_interleave_apply_elementary_reflector_generic>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator()(const ProblemContext& pctx) const {
		if (!can_compute(pctx)) return false;
		using value_type = std::remove_cv_t<typename decltype(pctx.c)::data_type>;

		const auto ninter = pctx.c.interleave_count();
		const auto m      = pctx.c.cntg();
		const auto n      = pctx.c.strd();

		PERFLIBS_ASSERT(
			ninter >= 0 && ninter <= max_interleave_size,
			"unsupported elementary-reflector interleave count");

		if (ninter == 0 || m == 0 || n == 0) return true;

		pod_vector<value_type> work_storage(ninter * n);
		const auto work = general_matrix { interleave_batch_matrix { work_storage.data(), ninter, n, 1, ninter, 0 } };

		const auto strategy_tag = spec::strategy_tag<batch_interleave_apply_elementary_reflector_generic> {};
		const auto spec = get_spec(strategy_tag, pctx);

		spec.kernel_gemv_qr(
			ninter, n, m,
			pctx.c.data(), pctx.c.cntg_step(), pctx.c.strd_step(),
			pctx.v.data(), pctx.v.cntg_step(),
			work.data(), work.cntg_step());

		spec.kernel_ger_qr(
			ninter, m, n,
			pctx.tau.data(), pctx.v.data(), pctx.v.cntg_step(),
			work.data(), work.cntg_step(),
			pctx.c.data(), pctx.c.cntg_step(), pctx.c.strd_step());

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator()(const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<batch_interleave_apply_elementary_reflector_generic>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const {
		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }

}; // class batch_interleave_apply_elementary_reflector_generic

} // namespace perflibs::linalg::batch_interleave::factorization

#endif // PERFLIBS_LINALG_BATCH_INTERLEAVE_FACTORIZATION_STRATEGIES_APPLY_ELEMENTARY_REFLECTOR_GENERIC_HPP
