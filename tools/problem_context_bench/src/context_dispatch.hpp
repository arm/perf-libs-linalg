/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_CONTEXT_DISPATCH_HPP
#define PERFLIBS_PCTX_CONTEXT_DISPATCH_HPP

#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"
#include "matrix/matrix.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace perflibs::linalg {

constexpr std::string_view data_type_name(std::type_identity<r16>) { return "r16"; }
constexpr std::string_view data_type_name(std::type_identity<bf16>) { return "bf16"; }
constexpr std::string_view data_type_name(std::type_identity<r32>) { return "r32"; }
constexpr std::string_view data_type_name(std::type_identity<r64>) { return "r64"; }
constexpr std::string_view data_type_name(std::type_identity<c32>) { return "c32"; }
constexpr std::string_view data_type_name(std::type_identity<c64>) { return "c64"; }

struct matrix_signature {
	std::string_view data_type;
	std::string_view matrix_type;

	bool operator==(const matrix_signature&) const = default;
};

template<typename Matrix>
constexpr matrix_signature matrix_signature_for() {
	using value_type = std::remove_cv_t<typename Matrix::value_type>;
	const auto type = data_type_name(std::type_identity<value_type> {});
	if constexpr(is_general_matrix_v<Matrix>) {
		return { type, "general_matrix" };
	}
	else if constexpr(is_symmetric_matrix_v<Matrix>) {
		return { type, "symmetric_matrix" };
	}
	else if constexpr(is_hermitian_matrix_v<Matrix>) {
		return { type, "hermitian_matrix" };
	}
	else {
		static_assert(is_triangular_matrix_v<Matrix>);
		return { type, "triangular_matrix" };
	}
}

struct one_matrix_signature {
	template<typename Context>
	constexpr auto operator()(std::type_identity<Context>) const {
		return matrix_signature_for<typename Context::matrix_type>();
	}
};

struct two_matrix_signature {
	template<typename Context>
	constexpr auto operator()(std::type_identity<Context>) const {
		return std::array {
			matrix_signature_for<typename Context::a_matrix_type>(),
			matrix_signature_for<typename Context::b_matrix_type>()
		};
	}
};

struct three_matrix_signature {
	template<typename Context>
	constexpr auto operator()(std::type_identity<Context>) const {
		return std::array {
			matrix_signature_for<typename Context::a_matrix_type>(),
			matrix_signature_for<typename Context::b_matrix_type>(),
			matrix_signature_for<typename Context::c_matrix_type>()
		};
	}
};

template<typename Key, std::size_t N>
constexpr bool unique_context_keys(const std::array<Key, N>& keys) {
	for(std::size_t i = 0; i < N; ++i) {
		for(std::size_t j = i + 1; j < N; ++j) {
			if(keys[i] == keys[j]) {
				return false;
			}
		}
	}
	return true;
}

template<typename... Contexts, typename Key, typename KeyFor, typename Run>
bool dispatch_contexts(std::type_identity<std::tuple<Contexts...>>, const Key& key, KeyFor, Run&& run) {
	constexpr std::array keys { KeyFor {}(std::type_identity<Contexts> {})... };
	static_assert(unique_context_keys(keys), "Context types have ambiguous input signatures");
	return ((key == KeyFor {}(std::type_identity<Contexts> {})
		&& (run(std::type_identity<Contexts> {}), true)) || ...);
}

} // namespace perflibs::linalg

#endif
