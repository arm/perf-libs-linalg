/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_FACTORIZATION_RUNNER_HPP
#define PERFLIBS_PCTX_FACTORIZATION_RUNNER_HPP

/*
 * Prepare inputs for factorization and QR/LQ benchmarks. These operations
 * overwrite input data, so restore the required inputs before each call.
 */

#include "benchmark_runner.hpp"
#include "perflibs_complex.hpp"
#include "pl_linalg_int.h"
#include "matrix/matrix.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iterator>
#include <random>
#include <type_traits>
#include <vector>

namespace perflibs::linalg {

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

} // namespace perflibs::linalg

#endif
