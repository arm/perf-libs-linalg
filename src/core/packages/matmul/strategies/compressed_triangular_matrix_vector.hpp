/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_TRIANGULAR_MATRIX_VECTOR_HPP
#define PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_TRIANGULAR_MATRIX_VECTOR_HPP

#include "blas/gbmv_driver.hpp"
#include "matrix/matrix.hpp"
#include "spec/strategy_tag.hpp"

#include <string_view>

namespace perflibs::linalg::matmul {

class compressed_triangular_matrix_vector {
public:
	static constexpr std::string_view name() { return "compressed_triangular_matrix_vector"; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_triangular_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	bool operator() (const ProblemContext& pctx) const {
		using value_type = typename ProblemContext::b_matrix_type::value_type;

		if ( !this->can_compute(pctx) ) return false;

		const auto spec = get_spec(spec::strategy_tag<compressed_triangular_matrix_vector>{}, pctx);

		compute_position pos{ 0, 0, 0 };
		auto driver = gbmv_driver{};

		auto b_buffer = get_memory<value_type>(pctx.b.cntg());

		general_matrix unpacked_b { matrix_base { b_buffer, pctx.b.cntg(), 1, 1, 0 } };

		// duplicate the contents of `pctx.b` into `unpacked_b`
		copy(pctx.b, unpacked_b);

		if(pctx.a.diag() == PERFLIBS_UNIT){
			driver(spec, pctx.a, unpacked_b, pctx.b, pos, /* alpha */ one<value_type>, /* beta */ one <value_type>);
		}
		else {
			driver(spec, pctx.a, unpacked_b, pctx.b, pos, /* alpha */ one<value_type>, /* beta */ zero<value_type>);
		}

		return_memory(b_buffer);

		return true;
	}

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool operator() (const ProblemContext&) const { return false; }

	template<typename ProblemContext>
	requires spec::has_get_spec<spec::strategy_tag<compressed_triangular_matrix_vector>, ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return true; }

	template<typename ProblemContext>
	PERFLIBS_LINALG_INLINE
	constexpr bool can_compute(const ProblemContext&) const { return false; }
}; //class compressed_triangular_matrix_vector

} //namespace perflibs::linalg::matmul

#endif // PERFLIBS_LINALG_MATMUL_STRATEGIES_COMPRESSED_TRIANGULAR_MATRIX_VECTOR_HPP
