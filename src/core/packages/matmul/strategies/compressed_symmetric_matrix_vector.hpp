/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_SYMMETRIC_MATRIX_VECTOR_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_SYMMETRIC_MATRIX_VECTOR_HPP

#include "blas/gbmv_driver.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class compressed_symmetric_matrix_vector {
public:
	static constexpr std::string_view name() { return "compressed_symmetric_matrix_vector"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_symmetric_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		if ( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<compressed_symmetric_matrix_vector>{}, pctx);

		compute_position pos{ 0, 0, 0 };
		auto driver = gbmv_driver{};
		driver(spec, pctx.a, pctx.b, pctx.c, pos, pctx.alpha, pctx.beta);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_symmetric_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class compressed_symmetric_matrix_vector

} //namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_SYMMETRIC_MATRIX_VECTOR_HPP
