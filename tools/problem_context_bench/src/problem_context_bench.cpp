/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "benchmark_io.hpp"
#include "context_flops.hpp"
#include "context_dispatch.hpp"
#include "matrix_input.hpp"
#include "measurement.hpp"
#include "supported_compute.hpp"

#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"
#include "pl_linalg_int.h"
#include "perflibs_numeric_utils.hpp"
#include "perflibs_util.hpp"
#include "framework/linalg_blas_types.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include <nlohmann/json.hpp>
#include "json_complex.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace perflibs::linalg {

template<typename ScalarType>
auto parse_scalar(const nlohmann::json& j) {
	// bf16 scalar values are read via fp32 before conversion.
	using parse_type = at_least_fp32_t<ScalarType>;
	return static_cast<ScalarType>(j.get<parse_type>());
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

	return dispatch(pctx, std::move(a_data), std::move(ipiv_data_ptr), std::move(info_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(info_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(tau_data_ptr), std::move(work_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(d_data_ptr), std::move(e_data_ptr),
	                std::move(tauq_data_ptr), std::move(taup_data_ptr), std::move(work_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(d_data_ptr), std::move(e_data_ptr),
	                std::move(tauq_data_ptr), std::move(taup_data_ptr), std::move(x_data_ptr),
	                std::move(y_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(d_data_ptr), std::move(e_data_ptr),
	                std::move(tau_data_ptr), std::move(work_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(tau_data_ptr), std::move(c_data_ptr), std::move(work_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(tau_data_ptr), std::move(work_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(tau_data_ptr), std::move(c_data_ptr), std::move(work_data_ptr));
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

	return dispatch(pctx, std::move(a_data), std::move(tau_data_ptr), std::move(work_data_ptr));
}

template<typename Context, typename... Args1>
auto parse_generate_q_from_lq_a(const nlohmann::json& j, Args1... args1) {
	return parse_matrix_as<typename Context::matrix_type>(j["a"], [&](auto... args2) {
		return parse_generate_q_from_lq_pctx(j, std::move(args1)..., std::move(args2)...);
	});
}


template<typename Context, typename Dispatch>
auto parse_matmul2(const nlohmann::json& j, Dispatch dispatch) {
	return parse_matrix_as<typename Context::a_matrix_type>(j["a"], [&](auto a, auto a_data) {
		return parse_matrix_as<typename Context::b_matrix_type>(j["b"], [&](auto b, auto b_data) {
			using scalar_type = typename Context::scalar_type;
			const auto alpha = parse_scalar<scalar_type>(j["alpha"]);
			const auto beta = zero<scalar_type>;
			Context pctx { a, b, alpha, beta };
			return dispatch(pctx, std::move(a_data), std::move(b_data));
		});
	});
}

template<typename Context, typename Dispatch>
auto parse_matmul3(const nlohmann::json& j, Dispatch dispatch) {
	return parse_matrix_as<typename Context::a_matrix_type>(j["a"], [&](auto a, auto a_data) {
		return parse_matrix_as<typename Context::b_matrix_type>(j["b"], [&](auto b, auto b_data) {
			return parse_matrix_as<typename Context::c_matrix_type>(j["c"], [&](auto c, auto c_data) {
				using scalar_type = typename Context::scalar_type;
				const auto alpha = parse_scalar<scalar_type>(j["alpha"]);
				const auto beta = parse_scalar<scalar_type>(j["beta"]);
				Context pctx { a, b, c, alpha, beta };
				return dispatch(pctx, std::move(a_data), std::move(b_data), std::move(c_data));
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
			return dispatch(pctx, std::move(a_data), std::move(b_data));
		});
	});
}

struct null_prepare {
	template<typename ProblemContextBase>
	inline
	void operator()(const ProblemContextBase&) const { }
};

template<typename ProblemContext, typename Prepare>
void bench(const nlohmann::json& j, const ProblemContext& pctx_base, const Prepare& prepare=null_prepare { } ) {
	auto settings = parse_bench_settings(j, get_n_strategies(pctx_base));
	auto output = make_bench_output(j);

	for(auto si = settings.first_strategy; si != settings.last_strategy; ++si) {
		if (settings.ignore[si]) {
			continue;
		}

		const auto nflops = get_dnflops(pctx_base);
		const auto measurement = measure(settings.measurement,
			[&] { prepare(pctx_base); },
			[&] { return compute_index(pctx_base, si); });
		const auto result = make_bench_result(measurement, nflops);

		append_bench_result(output, si, get_strategy_name(pctx_base, si), result);
		if (result.out_of_time) {
			settings.ignore[si] = true;
		}
	}

	print_bench_output(output, settings.ignore);
}

template<typename ProblemContextBase>
void run_cholesky(const nlohmann::json& j, const ProblemContextBase& pctx_base) {

	bench(j, pctx_base, [](auto& pctx) {
		// make input matrix positive definite
		auto a = to_general_matrix(pctx.a);

		if (pctx.a.is_lower()) {
			for (auto j = 0; j < a.strd(); j++) {
				a(j, j, write) = 10.0 * a.cntg();

				for (auto i = j + 1; i < a.cntg(); i++) {
					a(i, j, write) = 2.0;
				}
			}
		}
		else {
			for (auto j = 0; j < a.strd(); j++) {
				for (auto i = 0; i < j; i++) {
					a(i, j, write) = 2.0;
				}
				a(j, j, write) = 10 * a.cntg();
			}
		}
	});
}

template<typename ProblemContextBase>
void run_with_reset_matrix(
	const nlohmann::json& j,
	const ProblemContextBase& pctx_base) {

	using matrix_type       = typename ProblemContextBase::matrix_type;
	using value_type        = typename matrix_type::value_type;
	using mutable_data_type = std::remove_cv_t<value_type>;
	using real_data_type    = remove_complex_t<mutable_data_type>;

	std::vector<mutable_data_type> a_original_data(pctx_base.a.cntg() * pctx_base.a.strd());

	// Generate random data
	std::mt19937 gen{std::random_device{}()};
	std::uniform_real_distribution<real_data_type> dist{1.0, 2.0};

	std::generate(begin(a_original_data), end(a_original_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	if constexpr (is_symmetric_matrix_v<matrix_type>) {
		auto a_original = symmetric_matrix{
			pctx_base.a.uplo(),
			matrix_base{
				a_original_data.data(),
				pctx_base.a.cntg(), pctx_base.a.strd(),
				1, pctx_base.a.cntg()
			}
		};

		bench(j, pctx_base, [&](auto& pctx) {
			copy(a_original, pctx.a);
		});
	}
	else if constexpr (is_hermitian_matrix_v<matrix_type>) {
		auto a_original = hermitian_matrix{
			pctx_base.a.uplo(),
			matrix_base{
				a_original_data.data(),
				pctx_base.a.cntg(), pctx_base.a.strd(),
				1, pctx_base.a.cntg()
			}
		};

		bench(j, pctx_base, [&](auto& pctx) {
			copy(a_original, pctx.a);
		});
	}
	else {
		auto a_original = general_matrix{
			matrix_base{
				a_original_data.data(),
				pctx_base.a.cntg(), pctx_base.a.strd(),
				1, pctx_base.a.cntg()
			}
		};
		bench(j, pctx_base, [&](auto& pctx) {
			copy(a_original, pctx.a);
		});
	}
}


template<typename DataType>
void run_apply_q_from_qr(
	const nlohmann::json& j,
	const factorization::apply_q_from_qr<general_matrix<matrix_base<DataType>>>& pctx_base) {

	using matrix_type       = general_matrix<matrix_base<DataType>>;
	using value_type        = typename matrix_type::value_type;
	using mutable_data_type = std::remove_cv_t<value_type>;
	using real_data_type    = remove_complex_t<mutable_data_type>;

	const pl_linalg_int_t m = pctx_base.a.cntg();
	const pl_linalg_int_t n = pctx_base.a.strd();
	const pl_linalg_int_t k = pctx_base.tau.cntg();


	std::vector<mutable_data_type> a_qr_data(m * n);
	std::vector<mutable_data_type> tau_qr_data(k);
	std::vector<mutable_data_type> c_original_data(pctx_base.c.cntg() * pctx_base.c.strd());

	// Generate random data
	unsigned int seed = 42;
	std::mt19937 gen(seed);
	std::uniform_real_distribution<real_data_type> dist{1.0, 2.0};

	// Initialize A with random entries
	std::generate(begin(a_qr_data), end(a_qr_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});
	auto a_original = general_matrix{ matrix_base{ a_qr_data.data(), m, n, pctx_base.a.cntg_step(), pctx_base.a.strd_step()} };

	// Initialize random C matrix
	std::generate(begin(c_original_data), end(c_original_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	// Initialize random tau matrix
	std::generate(begin(tau_qr_data), end(tau_qr_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	auto tau_original = general_matrix{ matrix_base{ tau_qr_data.data(), k, 1, 1, 1 } };

	auto c_original = general_matrix{
		matrix_base{
			c_original_data.data(),
			pctx_base.c.cntg(), pctx_base.c.strd(),
			pctx_base.c.cntg_step(), pctx_base.c.strd_step()
		}
	};

	// Run benchmark with A, tau, and C reset before each iteration
	bench(j, pctx_base, [&](auto& pctx) {
		copy(a_original, pctx.a);
		copy(tau_original, pctx.tau);
		copy(c_original, pctx.c);
	});
}

template<typename DataType>
void run_generate_q_from_qr(
	const nlohmann::json& j,
	const factorization::generate_q_from_qr<general_matrix<matrix_base<DataType>>>& pctx_base) {

	using matrix_type       = general_matrix<matrix_base<DataType>>;
	using value_type        = typename matrix_type::value_type;
	using mutable_data_type = std::remove_cv_t<value_type>;
	using real_data_type    = remove_complex_t<mutable_data_type>;

	const auto m = pctx_base.a.cntg();
	const auto n = pctx_base.a.strd();
	const auto k = pctx_base.tau.cntg();

	std::vector<mutable_data_type> a_qr_data(m * n);
	std::vector<mutable_data_type> tau_qr_data(k);

	// Generate random data
	unsigned int seed = 42;
	std::mt19937 gen(seed);
	std::uniform_real_distribution<real_data_type> dist{1.0, 2.0};

	// Initialize A with random entries
	std::generate(begin(a_qr_data), end(a_qr_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	// Create temporary matrix for QR factorization
	auto a_original = general_matrix{ matrix_base{ a_qr_data.data(), m, n, pctx_base.a.cntg_step(), pctx_base.a.strd_step()} };

	// Initialize random tau vector
	std::generate(begin(tau_qr_data), end(tau_qr_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	auto tau_original = general_matrix{ matrix_base{ tau_qr_data.data(), k, 1, 1, 1 } };

	// Run benchmark with A and tau reset before each iteration
	// A matrix is overwritten by generate_q_from_qr, so we need fresh copy each time
	bench(j, pctx_base, [&](auto& pctx) {
		copy(a_original, pctx.a);
		copy(tau_original, pctx.tau);
	});
}

template<typename DataType>
void run_apply_q_from_lq(
	const nlohmann::json& j,
	const factorization::apply_q_from_lq<general_matrix<matrix_base<DataType>>>& pctx_base) {

	using matrix_type       = general_matrix<matrix_base<DataType>>;
	using value_type        = typename matrix_type::value_type;
	using mutable_data_type = std::remove_cv_t<value_type>;
	using real_data_type    = remove_complex_t<mutable_data_type>;

	const pl_linalg_int_t m = pctx_base.a.cntg();
	const pl_linalg_int_t n = pctx_base.a.strd();
	const pl_linalg_int_t k = pctx_base.tau.cntg();

	std::vector<mutable_data_type> a_lq_data(m * n);
	std::vector<mutable_data_type> tau_lq_data(k);
	std::vector<mutable_data_type> c_original_data(pctx_base.c.cntg() * pctx_base.c.strd());

	// Generate random data
	unsigned int seed = 42;
	std::mt19937 gen(seed);
	std::uniform_real_distribution<real_data_type> dist{1.0, 2.0};

	// Initialize A with random entries
	std::generate(begin(a_lq_data), end(a_lq_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});
	auto a_original = general_matrix{ matrix_base{ a_lq_data.data(), m, n, pctx_base.a.cntg_step(), pctx_base.a.strd_step()} };

	// Initialize random C matrix
	std::generate(begin(c_original_data), end(c_original_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	// Initialize random tau matrix
	std::generate(begin(tau_lq_data), end(tau_lq_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	auto tau_original = general_matrix{ matrix_base{ tau_lq_data.data(), k, 1, 1, 1 } };

	auto c_original = general_matrix{
		matrix_base{
			c_original_data.data(),
			pctx_base.c.cntg(), pctx_base.c.strd(),
			pctx_base.c.cntg_step(), pctx_base.c.strd_step()
		}
	};

	// Run benchmark with A, tau, and C reset before each iteration
	bench(j, pctx_base, [&](auto& pctx) {
		copy(a_original, pctx.a);
		copy(tau_original, pctx.tau);
		copy(c_original, pctx.c);
	});
}

template<typename DataType>
void run_generate_q_from_lq(
	const nlohmann::json& j,
	const factorization::generate_q_from_lq<general_matrix<matrix_base<DataType>>>& pctx_base) {

	using matrix_type       = general_matrix<matrix_base<DataType>>;
	using value_type        = typename matrix_type::value_type;
	using mutable_data_type = std::remove_cv_t<value_type>;
	using real_data_type    = remove_complex_t<mutable_data_type>;

	const auto m = pctx_base.a.cntg();
	const auto n = pctx_base.a.strd();
	const auto k = pctx_base.tau.cntg();

	std::vector<mutable_data_type> a_lq_data(m * n);
	std::vector<mutable_data_type> tau_lq_data(k);

	// Generate random data
	unsigned int seed = 42;
	std::mt19937 gen(seed);
	std::uniform_real_distribution<real_data_type> dist{1.0, 2.0};

	// Initialize A with random entries
	std::generate(begin(a_lq_data), end(a_lq_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	// Create temporary matrix for LQ factorization
	auto a_original = general_matrix{ matrix_base{ a_lq_data.data(), m, n, pctx_base.a.cntg_step(), pctx_base.a.strd_step()} };

	// Initialize random tau vector
	std::generate(begin(tau_lq_data), end(tau_lq_data), [&] {
		if constexpr (is_complex_v<mutable_data_type>) {
			return mutable_data_type{ dist(gen), dist(gen) };
		}
		else {
			return dist(gen);
		}
	});

	auto tau_original = general_matrix{ matrix_base{ tau_lq_data.data(), k, 1, 1, 1 } };

	// Run benchmark with A and tau reset before each iteration
	// A matrix is overwritten by generate_q_from_lq, so we need fresh copy each time
	bench(j, pctx_base, [&](auto& pctx) {
		copy(a_original, pctx.a);
		copy(tau_original, pctx.tau);
	});
}


matrix_signature parse_matrix_signature(const nlohmann::json& j) {
	return { j["data_type"].get_ref<const std::string&>(), j["matrix_type"].get_ref<const std::string&>() };
}

auto parse_two_matrix_signature(const nlohmann::json& j) {
	return std::array { parse_matrix_signature(j["a"]), parse_matrix_signature(j["b"]) };
}

auto parse_three_matrix_signature(const nlohmann::json& j) {
	return std::array {
		parse_matrix_signature(j["a"]), parse_matrix_signature(j["b"]), parse_matrix_signature(j["c"])
	};
}

void parse_problem_context(const nlohmann::json& j) {
	const auto& operation = j.at("problem_context_base");
	bool matched = true;

	if(operation == "matmul2") {
		matched = dispatch_contexts(matmul2_contexts {}, parse_two_matrix_signature(j), two_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul2<Context>(j, [&j](auto pctx_base, auto a_data, auto b_data) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "matmul3") {
		matched = dispatch_contexts(matmul3_contexts {}, parse_three_matrix_signature(j), three_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul3<Context>(j, [&j](auto pctx_base, auto a_data, auto b_data, auto c_data) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "matadd3") {
		matched = dispatch_contexts(matadd3_contexts {}, parse_three_matrix_signature(j), three_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul3<Context>(j, [&j](auto pctx_base, auto a_data, auto b_data, auto c_data) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "rank_update_2k") {
		matched = dispatch_contexts(rank_update_2k_contexts {}, parse_three_matrix_signature(j), three_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul3<Context>(j, [&j](auto pctx_base, auto a_data, auto b_data, auto c_data) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "solve") {
		matched = dispatch_contexts(solve_contexts {}, parse_two_matrix_signature(j), two_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_solve<Context>(j, [&j](auto pctx_base, auto a_data, auto b_data) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "lu_factorization") {
		matched = dispatch_contexts(lu_factorization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_lu_factorization_a<Context>(j, [&j](auto pctx_base, auto a_data, auto ipiv_data, auto info_data) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "cholesky_factorization") {
		matched = dispatch_contexts(cholesky_factorization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_cholesky_factorization_a<Context>(j, [&j](auto pctx_base, auto a_data, auto info_data) {
					run_cholesky(j, pctx_base);
				});
			});
	}
	else if(operation == "qr_factorization") {
		matched = dispatch_contexts(qr_factorization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_qr_factorization_a<Context>(j, [&j](auto pctx_base, auto a_data, auto tau_data, auto work_data) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "bidiagonalization") {
		matched = dispatch_contexts(bidiagonalization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_bidiagonalization_a<Context>(j, [&j](auto pctx_base, auto a_data, auto d_data, auto e_data, auto tauq_data, auto taup_data, auto work_data) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "bidiagonalization_block") {
		matched = dispatch_contexts(bidiagonalization_block_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_bidiagonalization_block_a<Context>(j, [&j](auto pctx_base, auto a_data, auto d_data, auto e_data, auto tauq_data, auto taup_data, auto x_data, auto y_data) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "tridiagonalization") {
		matched = dispatch_contexts(tridiagonalization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_tridiagonalization_a<Context>(j, [&j](auto pctx_base, auto a_data, auto d_data, auto e_data, auto tau_data, auto work_data) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "apply_q_from_qr") {
		matched = dispatch_contexts(apply_q_from_qr_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_apply_q_from_qr_a<Context>(j, [&j](auto pctx_base, auto a_data, auto tau_data, auto c_data, auto work_data) {
					run_apply_q_from_qr(j, pctx_base);
				});
			});
	}
	else if(operation == "generate_q_from_qr") {
		matched = dispatch_contexts(generate_q_from_qr_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_generate_q_from_qr_a<Context>(j, [&j](auto pctx_base, auto a_data, auto tau_data, auto work_data) {
					run_generate_q_from_qr(j, pctx_base);
				});
			});
	}
	else if(operation == "apply_q_from_lq") {
		matched = dispatch_contexts(apply_q_from_lq_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_apply_q_from_lq_a<Context>(j, [&j](auto pctx_base, auto a_data, auto tau_data, auto c_data, auto work_data) {
					run_apply_q_from_lq(j, pctx_base);
				});
			});
	}
	else if(operation == "generate_q_from_lq") {
		matched = dispatch_contexts(generate_q_from_lq_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_generate_q_from_lq_a<Context>(j, [&j](auto pctx_base, auto a_data, auto tau_data, auto work_data) {
					run_generate_q_from_lq(j, pctx_base);
				});
			});
	}
	else {
		throw std::invalid_argument("Unknown problem context " + operation.dump());
	}

	if(!matched) {
		throw std::invalid_argument("Unsupported matrix types for " + operation.dump());
	}
}

bool parse_problem_context(std::istream& is) {
	try {
		parse_problem_context( nlohmann::json::parse(is) );
		return true;
	}
	catch(const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return false;
	}
}

} // namespace perflibs:linalg

int main(int argc, char **argv) {
	const auto usage = [&] {
		std::cerr << "USAGE: " << argv[0] << " STDIN | FILE <file1> ... <fileN>\n";
	};

	if(argc < 2) {
		usage();
		return EXIT_FAILURE;
	}

	const std::string_view input_type { argv[1] };

	if(input_type == "STDIN" && argc == 2) {
		return perflibs::linalg::parse_problem_context(std::cin) ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	else if(input_type == "FILE" && argc > 2) {
		bool success = true;
		for(const std::filesystem::path file_path : std::span ( argv + 2, argc - 2 )) {
			std::ifstream file { file_path };
			if(!file) {
				std::cerr << "Cannot open " << file_path << '\n';
				success = false;
				continue;
			}
			if(!perflibs::linalg::parse_problem_context(file)) {
				success = false;
			}
		}
		return success ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	else {
		usage();
		return EXIT_FAILURE;
	}
}
