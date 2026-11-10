/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_MATRIX_INPUT_HPP
#define PERFLIBS_PCTX_MATRIX_INPUT_HPP

/*
 * Create matrices for the benchmark. Read their dimensions and layout from
 * JSON, allocate their data, and fill the data with random values.
 *
 * To add a matrix form:
 * - Read its JSON fields and construct its view in parse_matrix_as() below.
 * - Add its JSON name to matrix_signature_for() in context_dispatch.hpp.
 * - Add contexts that use it to problem_contexts in scripts/supported_compute.py.
 * For a new element type, also check random-value generation in make_matrix_data().
 */

#include "json_optional.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"
#include "perflibs_util.hpp"
#include "framework/linalg_blas_types.hpp"
#include "matrix/matrix.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace perflibs::linalg {

struct matrix_dimensions {
	kernel_inttype cntg;
	kernel_inttype strd;
	kernel_inttype cntg_step;
	kernel_inttype strd_step;
};

/*
 * cntg and strd are the required row and column counts.
 * cntg_step and strd_step give the distance, in elements, to the next row or
 * column. Element (i, j) is at offset i * cntg_step + j * strd_step.
 *
 * Use the supplied steps only if both JSON values are integers. If either is
 * missing or is not an integer, use cntg_step = 1 and strd_step = cntg.
 * These defaults store each column in consecutive elements, with no gaps
 * between columns.
 */
inline matrix_dimensions parse_matrix_dimensions(const nlohmann::json& j) {
	const auto cntg      = j["cntg"].get<kernel_inttype>();
	const auto strd      = j["strd"].get<kernel_inttype>();

	const auto [cntg_step, strd_step] = [&] {
		const auto cntg_step_opt = perflibs::from_json<kernel_inttype>(j, "cntg_step");
		const auto strd_step_opt = perflibs::from_json<kernel_inttype>(j, "strd_step");

		if(cntg_step_opt && strd_step_opt) {
			return std::pair { *cntg_step_opt, *strd_step_opt };
		}

		return std::pair { 1_ki, cntg };
	}();

	return { cntg, strd, cntg_step, strd_step };
}

template<typename DataType>
auto make_matrix_data(const matrix_dimensions& dimensions) {
	const auto [cntg, strd, cntg_step, strd_step] = dimensions;
	using real_data_type = remove_complex_t<DataType>;
	// std::uniform_real_distribution cannot generate bf16 values directly.
	// Generate fp32 values for bf16 matrices, then convert them below.
	using distribution_type = at_least_fp32_t<real_data_type>;

	auto data_ptr = std::make_unique<std::vector<DataType>>( cntg * cntg_step + strd * strd_step );

	std::mt19937 gen { std::random_device{}() };
	std::uniform_real_distribution<distribution_type> dist {1.0, 2.0};

	std::generate(begin(*data_ptr), end(*data_ptr), [&] {
		if constexpr(is_complex_v<DataType>) {
			return DataType { dist(gen), dist(gen) };
		}
		else {
			return static_cast<DataType>(dist(gen));
		}
	});

	return data_ptr;
}

template<typename Matrix, typename Dispatch>
auto parse_matrix_as(const nlohmann::json& j, Dispatch dispatch) {
	const auto dimensions = parse_matrix_dimensions(j);
	const auto [cntg, strd, cntg_step, strd_step] = dimensions;
	using data_type = typename Matrix::value_type;
	using mutable_data_type = std::remove_cv_t<data_type>;
	auto data_ptr = make_matrix_data<mutable_data_type>(dimensions);
	auto data_raw_ptr = static_cast<data_type*>(data_ptr->data());
	matrix_base base { data_raw_ptr, cntg, strd, cntg_step, strd_step };

	if constexpr(is_general_matrix_v<Matrix>) {
		Matrix matrix { base };
		return dispatch(std::move(matrix), std::move(data_ptr));
	}
	else {
		const auto uplo_str = j["uplo"].get<std::string>();
		if(uplo_str.size() == 0) {
			throw std::invalid_argument("uplo not specified for matrix with triangular form");
		}

		const auto uplo = c_to_uplo(uplo_str.front());
		if(uplo == static_cast<perflibs_uplo>(0)) {
			throw std::invalid_argument("can not parse uplo format");
		}

		static_assert(!is_hermitian_matrix_v<Matrix> || is_complex_v<mutable_data_type>);
		if constexpr(is_symmetric_matrix_v<Matrix> || is_hermitian_matrix_v<Matrix>) {
			Matrix matrix { uplo, base };
			return dispatch(std::move(matrix), std::move(data_ptr));
		}
		else {
			static_assert(is_triangular_matrix_v<Matrix>);
			const auto diag_str = j["diag"].get<std::string>();
			if(diag_str.size() == 0) {
				throw std::invalid_argument("diag not specified for matrix with triangular form");
			}

			const auto diag = c_to_diag(diag_str.front());
			if(diag == static_cast<perflibs_diag>(0)) {
				throw std::invalid_argument("can not parse diag format");
			}

			Matrix matrix { uplo, diag, base };
			return dispatch(std::move(matrix), std::move(data_ptr));
		}
	}
}

} // namespace perflibs::linalg

#endif
