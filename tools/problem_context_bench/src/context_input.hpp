/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_CONTEXT_INPUT_HPP
#define PERFLIBS_PCTX_CONTEXT_INPUT_HPP

/*
 * Build problem contexts from JSON matrix descriptions, scalars, and options.
 *
 * Contexts refer to matrix data, outputs, and work buffers without owning them.
 * The parsing functions keep this storage alive until dispatch(pctx) returns.
 * The callback must finish using the context and its data before it returns.
 */

#include "json_complex.hpp"
#include "matrix_input.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "pl_linalg_int.h"
#include "framework/linalg_blas_types.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace perflibs::linalg {

template<typename ScalarType>
auto parse_scalar(const nlohmann::json& j) {
	// Complex JSON values use [real, imaginary]. Read bf16 values as fp32,
	// then convert them to bf16.
	using parse_type = at_least_fp32_t<ScalarType>;
	return static_cast<ScalarType>(j.get<parse_type>());
}

template<typename Context, typename Dispatch>
auto parse_matmul2(const nlohmann::json& j, Dispatch dispatch) {
	return parse_matrix_as<typename Context::a_matrix_type>(j["a"], [&](auto a, auto a_data) {
		return parse_matrix_as<typename Context::b_matrix_type>(j["b"], [&](auto b, auto b_data) {
			using scalar_type = typename Context::scalar_type;
			const auto alpha = parse_scalar<scalar_type>(j["alpha"]);
			const auto beta = zero<scalar_type>;
			Context pctx { a, b, alpha, beta };
			return dispatch(pctx);
		});
	});
}

// matmul3, matadd3, and rank_update_2k all take A, B, C, alpha, and beta.
// They share this parser because their context fields have the same order.
template<typename Context, typename Dispatch>
auto parse_matmul3(const nlohmann::json& j, Dispatch dispatch) {
	return parse_matrix_as<typename Context::a_matrix_type>(j["a"], [&](auto a, auto a_data) {
		return parse_matrix_as<typename Context::b_matrix_type>(j["b"], [&](auto b, auto b_data) {
			return parse_matrix_as<typename Context::c_matrix_type>(j["c"], [&](auto c, auto c_data) {
				using scalar_type = typename Context::scalar_type;
				const auto alpha = parse_scalar<scalar_type>(j["alpha"]);
				const auto beta = parse_scalar<scalar_type>(j["beta"]);
				Context pctx { a, b, c, alpha, beta };
				return dispatch(pctx);
			});
		});
	});
}

template<typename Context, typename Dispatch>
auto parse_solve(const nlohmann::json& j, Dispatch dispatch) {
	return parse_matrix_as<typename Context::a_matrix_type>(j["a"], [&](auto a, auto a_data) {
		return parse_matrix_as<typename Context::b_matrix_type>(j["b"], [&](auto b, auto b_data) {
			using scalar_type = typename Context::scalar_type;
			const auto side = c_to_side(j["side"].get<std::string>()[0]);
			const auto transa = c_to_trans(j["transa"].get<std::string>()[0]);
			const auto alpha = parse_scalar<scalar_type>(j["alpha"]);
			Context pctx { side, transa, a, b, alpha };
			return dispatch(pctx);
		});
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_lu_factorization_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	const pl_linalg_int_t m = a_matrix.cntg();
	const pl_linalg_int_t n = a_matrix.strd();

	const auto min_mn = perflibs::min(m, n);
	auto ipiv_data_ptr = std::make_unique<std::vector<pl_linalg_int_t>>(min_mn);
	auto ipiv = ipiv_data_ptr->data();

	auto info_data_ptr = std::make_unique<pl_linalg_int_t>(0);

	factorization::lu_factorization pctx {
		a_matrix,
		std::span<pl_linalg_int_t>(ipiv, min_mn),
		*info_data_ptr
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_lu_factorization_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_lu_factorization_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_cholesky_factorization_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	auto info_data_ptr = std::make_unique<pl_linalg_int_t>(0);

	factorization::cholesky_factorization pctx { a_matrix, *info_data_ptr };

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_cholesky_factorization_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_cholesky_factorization_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_qr_factorization_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;

	const pl_linalg_int_t m = a_matrix.cntg();
	const pl_linalg_int_t n = a_matrix.strd();

	const auto min_mn = perflibs::min(m, n);
	auto tau_data_ptr = std::make_unique<std::vector<value_type>>(min_mn);
	auto tau = tau_data_ptr->data();

	const auto lwork = n;
	auto work_data_ptr = std::make_unique<std::vector<value_type>>(n);
	auto work = work_data_ptr->data();

	factorization::qr_factorization pctx {
		a_matrix,
		std::span<value_type>(tau, min_mn),
		std::span<value_type>(work, lwork)
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_qr_factorization_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_qr_factorization_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_bidiagonalization_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;
	using real_type  = remove_complex_t<value_type>;

	const pl_linalg_int_t m      = a_matrix.cntg();
	const pl_linalg_int_t n      = a_matrix.strd();
	const auto        min_mn = perflibs::min(m, n);

	// Allocate for diagonal elements
	auto d_data_ptr = std::make_unique<std::vector<real_type>>(min_mn);

	// Allocate for off-diagonal elements (size is min_mn-1)
	auto e_data_ptr = std::make_unique<std::vector<real_type>>(min_mn > 0 ? min_mn - 1 : 0);

	// Allocate for reflectors
	auto tauq_data_ptr = std::make_unique<std::vector<value_type>>(min_mn);
	auto taup_data_ptr = std::make_unique<std::vector<value_type>>(min_mn);

	// Optimal workspace size
	const auto lwork   = perflibs::max(m, n);
	auto work_data_ptr = std::make_unique<std::vector<value_type>>(lwork);

	const kernel_inttype d_cntg    = min_mn;
	const kernel_inttype e_cntg    = min_mn > 0 ? min_mn - 1 : 0;
	const kernel_inttype work_cntg = lwork;

	factorization::bidiagonalization pctx {
		a_matrix,
		general_matrix<matrix_base<real_type>>  { matrix_base<real_type>  { d_data_ptr->data(),    d_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<real_type>>  { matrix_base<real_type>  { e_data_ptr->data(),    e_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { tauq_data_ptr->data(), d_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { taup_data_ptr->data(), d_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { work_data_ptr->data(), work_cntg, 1_ki, 1_ki, 1_ki } }
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_bidiagonalization_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_bidiagonalization_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_bidiagonalization_block_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;
	using real_type  = remove_complex_t<value_type>;

	const pl_linalg_int_t m      = a_matrix.cntg();
	const pl_linalg_int_t n      = a_matrix.strd();
	const pl_linalg_int_t nb     = j["nb"].get<pl_linalg_int_t>();
	const auto        min_mn = perflibs::min(m, n);

	auto d_data_ptr    = std::make_unique<std::vector<real_type>>(nb);
	auto e_data_ptr    = std::make_unique<std::vector<real_type>>(nb);
	auto tauq_data_ptr = std::make_unique<std::vector<value_type>>(nb);
	auto taup_data_ptr = std::make_unique<std::vector<value_type>>(nb);
	auto x_data_ptr    = std::make_unique<std::vector<value_type>>(m * nb);
	auto y_data_ptr    = std::make_unique<std::vector<value_type>>(n * nb);

	const kernel_inttype d_cntg    = nb;
	const kernel_inttype e_cntg    = nb;
	const kernel_inttype tauq_cntg = nb;
	const kernel_inttype taup_cntg = nb;
	const kernel_inttype x_cntg    = m;
	const kernel_inttype x_strd    = nb;
	const kernel_inttype y_cntg    = n;
	const kernel_inttype y_strd    = nb;

	factorization::bidiagonalization_block pctx {
		a_matrix,
		general_matrix<matrix_base<real_type>>  { matrix_base<real_type>  { d_data_ptr->data(),    d_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<real_type>>  { matrix_base<real_type>  { e_data_ptr->data(),    e_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { tauq_data_ptr->data(), tauq_cntg, 1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { taup_data_ptr->data(), taup_cntg, 1_ki, 1_ki, 1_ki } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { x_data_ptr->data(),    x_cntg,    x_strd, 1_ki, x_cntg } },
		general_matrix<matrix_base<value_type>> { matrix_base<value_type> { y_data_ptr->data(),    y_cntg,    y_strd, 1_ki, y_cntg } }
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_bidiagonalization_block_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_bidiagonalization_block_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_tridiagonalization_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;
	using real_type  = remove_complex_t<value_type>;

	const pl_linalg_int_t n = a_matrix.strd();

	auto d_data_ptr   = std::make_unique<std::vector<real_type>>(n);
	auto e_data_ptr   = std::make_unique<std::vector<real_type>>(n > 0 ? n - 1 : 0);
	auto tau_data_ptr = std::make_unique<std::vector<value_type>>(n > 0 ? n - 1 : 0);

	const auto lwork   = n;
	auto work_data_ptr = std::make_unique<std::vector<value_type>>(lwork);

	const kernel_inttype d_cntg    = n;
	const kernel_inttype e_cntg    = n > 0 ? n - 1 : 0;
	const kernel_inttype tau_cntg  = n > 0 ? n - 1 : 0;
	const kernel_inttype work_cntg = lwork;

	factorization::tridiagonalization pctx {
		a_matrix,
		general_matrix { matrix_base { d_data_ptr->data(),    d_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix { matrix_base { e_data_ptr->data(),    e_cntg,    1_ki, 1_ki, 1_ki } },
		general_matrix { matrix_base { tau_data_ptr->data(),  tau_cntg,  1_ki, 1_ki, 1_ki } },
		general_matrix { matrix_base { work_data_ptr->data(), work_cntg, 1_ki, 1_ki, 1_ki } }
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_tridiagonalization_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_tridiagonalization_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_apply_q_from_qr_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;

	const pl_linalg_int_t m = a_matrix.cntg();
	const pl_linalg_int_t n = a_matrix.strd();
	const pl_linalg_int_t k = std::min(m, n);

	// Parse side and trans from JSON
	const auto side = c_to_side(j["side"].get<std::string>()[0]);
	const auto trans = c_to_trans(j["trans"].get<std::string>()[0]);

	if (side == PERFLIBS_LEFT && k > m) {
		throw std::invalid_argument("Error: Invalid dimensions for left multiplication: M >= K required. Got M="
			+ std::to_string(m) + ", K=" + std::to_string(k));
	}
	if (side == PERFLIBS_RIGHT && k > n) {
		throw std::invalid_argument("Error: Invalid dimensions for right multiplication: N >= K required. Got N="
			+ std::to_string(n) + ", K=" + std::to_string(k));
	}

	// Allocate tau vector
	auto tau_data_ptr = std::make_unique<std::vector<value_type>>(k);

	// Allocate C matrix
	auto c_data_ptr = std::make_unique<std::vector<value_type>>(m * n);

	// Allocate work array
	const auto lwork = std::max(m, n);
	auto work_data_ptr = std::make_unique<std::vector<value_type>>(lwork);

	// Create problem context (data will be properly initialized in run_apply_q_from_qr)
	factorization::apply_q_from_qr pctx {
		side,
		trans,
		a_matrix,
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{tau_data_ptr->data(), k, 1_ki, 1_ki, 1_ki}},
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{c_data_ptr->data(), m, n, 1_ki, m}},
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{work_data_ptr->data(), lwork, 1_ki, 1_ki, 1_ki}}
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_apply_q_from_qr_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_apply_q_from_qr_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_generate_q_from_qr_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;

	const pl_linalg_int_t m = a_matrix.cntg();
	const pl_linalg_int_t n = a_matrix.strd();
	const pl_linalg_int_t k = std::min(m, n);

	// Input validation
	if (m < n || k < 0) {
		throw std::invalid_argument("Error: Invalid dimensions for generate_q_from_qr: M >= N >= 0 and K >= 0 required. Got M="
			+ std::to_string(m) + ", N=" + std::to_string(n) + ", K=" + std::to_string(k));
	}

	// Allocate tau vector
	auto tau_data_ptr = std::make_unique<std::vector<value_type>>(k);

	// Allocate work array
	const auto lwork = std::max(m, n);
	auto work_data_ptr     = std::make_unique<std::vector<value_type>>(lwork);

	// Create problem context
	factorization::generate_q_from_qr pctx {
		a_matrix,
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{tau_data_ptr->data(), k, 1_ki, 1_ki, 1_ki}},
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{work_data_ptr->data(), lwork, 1_ki, 1_ki, 1_ki}}
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_generate_q_from_qr_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_generate_q_from_qr_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_apply_q_from_lq_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;

	const pl_linalg_int_t m = a_matrix.cntg();
	const pl_linalg_int_t n = a_matrix.strd();
	const pl_linalg_int_t k = std::min(m, n);

	// Parse side and trans from JSON
	const auto side = c_to_side(j["side"].get<std::string>()[0]);
	const auto trans = c_to_trans(j["trans"].get<std::string>()[0]);

	// Validation logic (same as UNMQR)
	if (side == PERFLIBS_LEFT && k > m) {
		throw std::invalid_argument("Error: Invalid dimensions for left multiplication: M >= K required. Got M="
			+ std::to_string(m) + ", K=" + std::to_string(k));
	}
	if (side == PERFLIBS_RIGHT && k > n) {
		throw std::invalid_argument("Error: Invalid dimensions for right multiplication: N >= K required. Got N="
			+ std::to_string(n) + ", K=" + std::to_string(k));
	}

	// Allocate tau vector
	auto tau_data_ptr = std::make_unique<std::vector<value_type>>(k);

	// Allocate C matrix
	auto c_data_ptr = std::make_unique<std::vector<value_type>>(m * n);

	// Allocate work array
	const auto lwork = std::max(m, n);
	auto work_data_ptr = std::make_unique<std::vector<value_type>>(lwork);

	// Create problem context
	factorization::apply_q_from_lq pctx {
		side,
		trans,
		a_matrix,
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{tau_data_ptr->data(), k, 1_ki, 1_ki, 1_ki}},
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{c_data_ptr->data(), m, n, 1_ki, m}},
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{work_data_ptr->data(), lwork, 1_ki, 1_ki, 1_ki}}
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_apply_q_from_lq_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_apply_q_from_lq_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

template<
	typename Dispatch,
	typename AMatrixType, typename ADataStore
>
auto parse_generate_q_from_lq_pctx(const nlohmann::json& j,
	Dispatch dispatch,
	AMatrixType a_matrix, ADataStore a_data) {

	using value_type = typename AMatrixType::value_type;

	const pl_linalg_int_t m = a_matrix.cntg();
	const pl_linalg_int_t n = a_matrix.strd();
	const pl_linalg_int_t k = std::min(m, n);

	// Input validation - for LQ: N >= M >= 0 and K >= 0
	if (n < m || m < 0 || k < 0) {
		throw std::invalid_argument("Error: Invalid dimensions for generate_q_from_lq: N >= M >= 0 and K >= 0 required. Got M="
			+ std::to_string(m) + ", N=" + std::to_string(n) + ", K=" + std::to_string(k));
	}

	// Allocate tau vector
	auto tau_data_ptr = std::make_unique<std::vector<value_type>>(k);

	// Allocate work array
	const auto lwork = std::max(m, n);
	auto work_data_ptr = std::make_unique<std::vector<value_type>>(lwork);

	// Create problem context
	factorization::generate_q_from_lq pctx {
		a_matrix,
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{tau_data_ptr->data(), k, 1_ki, 1_ki, 1_ki}},
		general_matrix<matrix_base<value_type>>{matrix_base<value_type>{work_data_ptr->data(), lwork, 1_ki, 1_ki, 1_ki}}
	};

	return dispatch(pctx);
}

template<typename Context, typename... Args1>
auto parse_generate_q_from_lq_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_generate_q_from_lq_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}

} // namespace perflibs::linalg

#endif
