/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_EXTRA_JSON_OPTIONAL_HPP
#define PERFLIBS_EXTRA_JSON_OPTIONAL_HPP

#include "perflibs_float.hpp"
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace perflibs {
	template<typename T, typename = void>
	struct json_impl;

	template<typename T>
	static inline std::optional<T> from_json(const nlohmann::json &j) {
		return json_impl<T>::from_json(j);
	}

	template<typename T>
	static inline std::optional<T> from_json(const nlohmann::json &j, const char *member) {
		if (!j.is_object()) {
			return std::nullopt;
		}
		const auto it = j.find(member);
		if (it == j.end()) {
			return std::nullopt;
		}
		return perflibs::from_json<T>(*it);
	}

	template<typename T>
	static inline T from_json_or(const nlohmann::json &j, const char *member, T v0) {
		if(const auto v1 = perflibs::from_json<T>(j, member); v1) {
			return *v1;
		}
		return v0;
	}

	template<typename T>
	static inline nlohmann::json to_json(const T &v) {
		return json_impl<T>::to_json(v);
	}

	template<typename T, template<typename...> typename vec_t>
	struct json_impl<vec_t<T>> {
		/**
		 * Deserialistion of a vector type from JSON.
		 * @return Optional with vector populated if reading an array from
		 *         the json object succeeds. Else returns a \c nullopt
		 */
		static inline std::optional<vec_t<T>> from_json(const nlohmann::json &j) {
			if (!j.is_array()) {
				return std::nullopt;
			}
			vec_t<T> ret;
			for (const auto &e : j) {
				auto e2 = json_impl<T>::from_json(e);
				if (!e2) {
					return std::nullopt;
				}
				ret.emplace_back(std::move(*e2));
			}
			return ret;
		}

		/**
		 * Serialization of a vector type to JSON.
		 * @return JSON representation of the vector
		 */
		static inline nlohmann::json to_json(const vec_t<T> &v) {
			auto j = nlohmann::json::array_t{};
			for (const auto &e : v) {
				j.emplace_back(json_impl<T>::to_json(e));
			}
			return j;
		}
	};

	template<>
	struct json_impl<std::string> {
		static inline std::optional<std::string> from_json(const nlohmann::json &j) {
			if (!j.is_string()) {
				return std::nullopt;
			}
			return j.get<std::string>();
		}

		static inline nlohmann::json to_json(const std::string &v) {
			return (nlohmann::json) v;
		}
	};

	/** Convert all integral datatypes to/from JSON.
	 */
	template<typename T>
	struct json_impl<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> {
		static inline std::optional<T> from_json(const nlohmann::json &j) {
			if (!j.is_number_integer()) {
				return std::nullopt;
			}
			return j.get<T>();
		}

		static inline nlohmann::json to_json(const T &v) {
			return (nlohmann::json) v;
		}
	};

	/** Convert boolean values to/from JSON.
	 */
	template<>
	struct json_impl<bool> {
		static inline std::optional<bool> from_json(const nlohmann::json &j) {
			if (!j.is_boolean()) {
				return std::nullopt;
			}
			return j.get<bool>();
		}

		static inline nlohmann::json to_json(const bool &v) {
			return (nlohmann::json) v;
		}
	};

	/** Convert all floating-point values to/from JSON.
	 *  Handles NaN by serializing to/from null.
	 *  Does not handle inf, or NaNs of different bit-patterns.
	 */
	template<typename T>
	struct json_impl<T, std::enable_if_t<perflibs::is_floating_point_v<T>>> {
		static inline std::optional<T> from_json(const nlohmann::json &j) {
			if (j.is_null()) {
				return perflibs::nan<T>;
			}
			else if (j.is_number_float()) {
				return static_cast<T>(j.get<at_least_fp32_t<T>>());
			}
			return std::nullopt;
		}

		static inline nlohmann::json to_json(const T &v) {
			return (nlohmann::json) perflibs::at_least_fp32(v);
		}
	};
}

#endif
