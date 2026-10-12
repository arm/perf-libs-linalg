/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_EXTRA_JSON_OPTIONAL_HPP
#define PERFLIBS_EXTRA_JSON_OPTIONAL_HPP

#include "perflibs_float.hpp"
#include <nlohmann/json.hpp>

#include <optional>
#include <complex>

namespace perflibs {
	template<typename T, typename = void>
	struct json_impl;

	template<typename T>
	static inline std::optional<T> from_json(const nlohmann::json &j) {
		return json_impl<T>::from_json(j);
	}

	template<typename T>
	static inline T from_json_or(const nlohmann::json &j, T v0) {
		if(const auto v1 = perflibs::from_json<T>(j); v1) {
			return *v1;
		}
		return v0;
	}

	template<typename T>
	static inline std::optional<T> from_json(const nlohmann::json &j, const char *member) {
		if (!j.is_object()) {
			return std::nullopt;
		}
		auto it = j.find(member);
		if (it == j.end()) {
			return std::nullopt;
		}
		return perflibs::from_json<T>(*it);
	}

	template<typename T>
	static inline std::optional<T> from_json(const nlohmann::json &j, const std::string &member) {
		return perflibs::from_json<T>(j, member.c_str());
	}

	template<typename T>
	static inline T from_json_or(const nlohmann::json &j, const char *member, T v0) {
		if(const auto v1 = perflibs::from_json<T>(j, member); v1) {
			return *v1;
		}
		return v0;
	}

	template<typename T>
	static inline std::optional<T> from_json_or(const nlohmann::json &j, const std::string& member, T v1) {
		return from_json_or(j, member.c_str(), v1);
	}

	template<typename T>
	static inline nlohmann::json to_json(const T &v) {
		return json_impl<T>::to_json(v);
	}

	template<typename K, typename V>
	struct json_impl<std::map<K,V>> {

		static inline std::optional<std::map<K,V>> from_json(const nlohmann::json &j) {
			if (!j.is_object()) {
				return {};
			}

			std::map<K, V> ret;

			for(auto it = begin(j); it != end(j); ++it) {
				auto kt = json_impl<K>::from_json(it.key());
				auto vt = json_impl<V>::from_json(it.value());

				if(! ( kt && vt ))
					return {};

				ret.emplace(std::move(*kt), std::move(*vt));
			}
			return ret;
		}

		static inline nlohmann::json to_json(const std::map<K, V> &map) {
			auto j = nlohmann::json::object_t{};
			for (const auto &[k, v] : map) {
				j.emplace(
					json_impl<K>::to_json(k),
					json_impl<V>::to_json(v));
			}
			return j;
		}
	};

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

	template<>
	struct json_impl<nlohmann::json> {
		static inline std::optional<nlohmann::json> from_json(const nlohmann::json &j) {
			return std::optional<nlohmann::json>(j);
		}

		static inline nlohmann::json to_json(const nlohmann::json &v) {
			return v;
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

	/** Convert enums to/from JSON by converting to ordinal values.
	 */
	template<typename T>
	struct json_impl<T, std::enable_if_t<std::is_enum_v<T>>> {
		typedef std::underlying_type_t<T> U;

		static inline std::optional<T> from_json(const nlohmann::json &j) {
			auto ret = json_impl<U>::from_json(j);
			if (!ret) {
				return std::nullopt;
			}
			return (T) *ret;
		}

		static inline nlohmann::json to_json(const T &v) {
			return json_impl<U>::to_json((U) v);
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

	/** Convert complex numbers to/from JSON.
	 *  Implemented by storing as a JSON array of two elements.
	 */
	template<typename T>
	struct json_impl<std::complex<T>> {
		static inline std::optional<std::complex<T>> from_json(const nlohmann::json &j) {
			if (!j.is_array() || j.size() != 2) {
				return std::nullopt;
			}
			auto r = perflibs::from_json<T>(j.at(0));
			auto i = perflibs::from_json<T>(j.at(1));
			if (!r || !i) {
				return std::nullopt;
			}
			return {{ *r, *i }};
		}

		static inline nlohmann::json to_json(const std::complex<T> &v) {
			auto r = perflibs::to_json(v.real());
			auto i = perflibs::to_json(v.imag());
			return nlohmann::json::array_t{ r, i };
		}
	};
}

#endif
