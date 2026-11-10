/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "benchmark_io.hpp"
#include "context_flops.hpp"
#include "context_dispatch.hpp"
#include "context_input.hpp"
#include "measurement.hpp"
#include "supported_compute.hpp"

#include "perflibs_complex.hpp"
#include "pl_linalg_int.h"
#include "perflibs_util.hpp"
#include "framework/linalg_blas_types.hpp"
#include "framework/linalg_util.hpp"
#include "matrix/matrix.hpp"
#include "packages/factorization/problem_context_bases.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace perflibs::linalg {

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
				parse_matmul2<Context>(j, [&j](const auto& pctx_base) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "matmul3") {
		matched = dispatch_contexts(matmul3_contexts {}, parse_three_matrix_signature(j), three_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul3<Context>(j, [&j](const auto& pctx_base) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "matadd3") {
		matched = dispatch_contexts(matadd3_contexts {}, parse_three_matrix_signature(j), three_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul3<Context>(j, [&j](const auto& pctx_base) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "rank_update_2k") {
		matched = dispatch_contexts(rank_update_2k_contexts {}, parse_three_matrix_signature(j), three_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_matmul3<Context>(j, [&j](const auto& pctx_base) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "solve") {
		matched = dispatch_contexts(solve_contexts {}, parse_two_matrix_signature(j), two_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_solve<Context>(j, [&j](const auto& pctx_base) {
					bench(j, pctx_base, null_prepare {});
				});
			});
	}
	else if(operation == "lu_factorization") {
		matched = dispatch_contexts(lu_factorization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_lu_factorization_a<Context>(j, [&j](const auto& pctx_base) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "cholesky_factorization") {
		matched = dispatch_contexts(cholesky_factorization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_cholesky_factorization_a<Context>(j, [&j](const auto& pctx_base) {
					run_cholesky(j, pctx_base);
				});
			});
	}
	else if(operation == "qr_factorization") {
		matched = dispatch_contexts(qr_factorization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_qr_factorization_a<Context>(j, [&j](const auto& pctx_base) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "bidiagonalization") {
		matched = dispatch_contexts(bidiagonalization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_bidiagonalization_a<Context>(j, [&j](const auto& pctx_base) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "bidiagonalization_block") {
		matched = dispatch_contexts(bidiagonalization_block_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_bidiagonalization_block_a<Context>(j, [&j](const auto& pctx_base) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "tridiagonalization") {
		matched = dispatch_contexts(tridiagonalization_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_tridiagonalization_a<Context>(j, [&j](const auto& pctx_base) {
					run_with_reset_matrix(j, pctx_base);
				});
			});
	}
	else if(operation == "apply_q_from_qr") {
		matched = dispatch_contexts(apply_q_from_qr_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_apply_q_from_qr_a<Context>(j, [&j](const auto& pctx_base) {
					run_apply_q_from_qr(j, pctx_base);
				});
			});
	}
	else if(operation == "generate_q_from_qr") {
		matched = dispatch_contexts(generate_q_from_qr_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_generate_q_from_qr_a<Context>(j, [&j](const auto& pctx_base) {
					run_generate_q_from_qr(j, pctx_base);
				});
			});
	}
	else if(operation == "apply_q_from_lq") {
		matched = dispatch_contexts(apply_q_from_lq_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_apply_q_from_lq_a<Context>(j, [&j](const auto& pctx_base) {
					run_apply_q_from_lq(j, pctx_base);
				});
			});
	}
	else if(operation == "generate_q_from_lq") {
		matched = dispatch_contexts(generate_q_from_lq_contexts {}, parse_matrix_signature(j["a"]), one_matrix_signature {},
			[&]<typename Context>(std::type_identity<Context>) {
				parse_generate_q_from_lq_a<Context>(j, [&j](const auto& pctx_base) {
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
