/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_COMMON_FLOAT_HPP
#define PERFLIBS_COMMON_FLOAT_HPP

#include <arm_bf16.h>

#include <complex>
#include <vector>

#include <cmath>
#include <cstdint>
#include <cstring>

typedef __fp16 half;

namespace perflibs {

using bf16=__bf16;
using r16=half;
using r32=float;
using r64=double;

template<typename T>
struct is_floating_point {
	constexpr static const bool value = false;
};

template<>
struct is_floating_point<bf16> {
	constexpr static const bool value = true;
};

template<>
struct is_floating_point<half> {
	constexpr static const bool value = true;
};

template<>
struct is_floating_point<float> {
	constexpr static const bool value = true;
};

template<>
struct is_floating_point<double> {
	constexpr static const bool value = true;
};

template<typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template<typename T>
struct float_uint;

template<>
struct float_uint<bf16> {
	typedef uint16_t type;
};

template<>
struct float_uint<half> {
	typedef uint16_t type;
};

template<>
struct float_uint<float> {
	typedef uint32_t type;
};

template<>
struct float_uint<double> {
	typedef uint64_t type;
};

/** Get an appropriately-sized unsigned integer type
 *  for the floating-point type specified.
 */
template<typename T>
using float_uint_t = typename float_uint<T>::type;

template<typename T>
struct at_least_fp32_type {
	typedef T type;
};
template<>
struct at_least_fp32_type<std::complex<half>> {
	typedef std::complex<float> type;
};
template<>
struct at_least_fp32_type<half> {
	typedef float type;
};

template<>
struct at_least_fp32_type<bf16> {
	typedef float type;
};

/** Returns a floating-point type at least 32-bits wide.
 *  i.e. bf16   -> float
 *       half   -> float
 *       float  -> float
 *       double -> double
 */
template <typename T>
using at_least_fp32_t = typename at_least_fp32_type<T>::type;

template<typename T>
struct at_least_fp32_value {
	static auto convert(T value) {
		return at_least_fp32_t<T> { value };
	}
};

template<>
struct at_least_fp32_value<bf16> {
	static float convert(bf16 value) {
		uint16_t in;
		std::memcpy(&in, &value, sizeof(in));

		uint32_t out = static_cast<uint32_t>(in) << 16;
		float result;
		std::memcpy(&result, &out, sizeof(result));
		return result;
	}
};

template<typename T>
struct at_least_fp32_value<std::complex<T>> {
	static auto convert(std::complex<T> value) {
		using out_type = at_least_fp32_t<T>;
		return std::complex {
			out_type { value.real() },
			out_type { value.imag() }
		};
	}
};

template<typename T>
struct at_least_fp32_value<std::vector<T>> {
	static std::vector<at_least_fp32_t<T>> convert(std::vector<T> values) {
		std::vector<at_least_fp32_t<T>> res;
		res.reserve(values.size());
		for (const T &value : values) {
			res.push_back(at_least_fp32_value<T>::convert(value));
		}
		return res;
	}
};

/** Returns a floating-point or complex value with an element size
 *  at least 32-bits wide.
 */
template<typename T>
auto at_least_fp32(T value) {
	return at_least_fp32_value<T>::convert(std::move(value));
}

template<typename T, typename = void>
struct nan_value;

template<typename T>
struct nan_value<std::complex<T>> {
	static std::complex<T> value() {
		return { nan_value<T>::value(), nan_value<T>::value() };
	}
};

template<typename T>
struct nan_value<T, std::enable_if_t<is_floating_point_v<T>>> {
	constexpr static T value() {
		float_uint_t<T> ival = 0;
		ival = ~ival;
		ival >>= 1;

		//in an ideal world we'd use: std::bit_cast<T>( ival );
		//but MSVC stdlib on windows w/ llvm doesn't support it
		T out { };
		std::memcpy(&out, &ival, sizeof(out));
		return out;
	}
};

inline bool isnan(auto val) {
	return std::isnan(val);
}

inline bool isnan(bf16 val) {
	std::uint16_t bits;
    std::memcpy(&bits, &val, sizeof bits);

    bits &= std::uint16_t{0x7fff};

    return bits > std::uint16_t{0x7f80};
}

/// Get a floating-point or complex NaN value.
template<typename T>
inline T nan = nan_value<T>::value();

template<typename T>
inline T get_nan() {
	return nan_value<T>::value();
}

inline auto abs(auto v) {
	return std::abs(v);
}

inline bf16 abs(bf16 val) {
	uint16_t bits;
	std::memcpy(&bits, &val, sizeof bits);

	bits &= std::uint16_t{0x7fff};

	std::memcpy(&val, &bits, sizeof val);
	return val;
}

} //namespace perflibs

inline std::complex<half> operator/(std::complex<half> a, float b) {
	return { (half) (a.real()/b), (half) (a.imag()/b) };
}

inline std::complex<half> operator/(std::complex<half> a, double b) {
	return { (half) (a.real()/b), (half) (a.imag()/b) };
}

inline std::complex<half> operator*(std::complex<half> a, float b) {
	return { (half) (a.real()*b), (half) (a.imag()*b) };
}

inline std::complex<half> operator*(std::complex<half> a, double b) {
	return { (half) (a.real()*b), (half) (a.imag()*b) };
}

#endif
