/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

/*
 * Choose the C++ context type from the JSON operation name and each matrix's
 * data_type and matrix_type fields. Then build the context and run the benchmark.
 */

#include "context_runner.hpp"
#include "benchmark_runner.hpp"
#include "context_dispatch.hpp"
#include "context_input.hpp"
#include "factorization_runner.hpp"
#include "supported_compute.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace perflibs::linalg {
namespace {

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

} // namespace

void run_context_benchmark(const nlohmann::json& j) {
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

bool run_context_benchmark(std::istream& is) {
	try {
		run_context_benchmark( nlohmann::json::parse(is) );
		return true;
	}
	catch(const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return false;
	}
}

} // namespace perflibs::linalg
