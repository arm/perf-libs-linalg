/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */


#ifndef PERFLIBS_COMMON_COMPLEX_H
#define PERFLIBS_COMMON_COMPLEX_H

#include <complex>

#include "perflibs_float.hpp"

typedef std::complex<half> complex_half;
typedef std::complex<float> complex_float;
typedef std::complex<double> complex_double;

namespace perflibs {

using c16=complex_half;
using c32=complex_float;
using c64=complex_double;

template<typename T>
inline auto real(const T& x) {
	return std::real(x);
}

inline half real(half x) {
	return x;
}

inline bf16 real(bf16 x) {
	return x;
}

template<typename T>
inline auto imag(const T& x) {
	return std::imag(x);
}

inline half imag(half) {
	return static_cast<half>( 0.0f );
}

inline bf16 imag(bf16) {
	return static_cast<bf16>( 0.0f );
}

template <typename T>
struct is_complex : public std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex<const std::complex<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
struct remove_complex {
	typedef T type;
};
template <typename T>
struct remove_complex<std::complex<T>> {
	typedef T type;
};

template <typename T>
using remove_complex_t = typename remove_complex<T>::type;

template <typename T>
perflibs::remove_complex_t<T> sum_abs(const T z){
	return perflibs::abs(perflibs::real(z)) + perflibs::abs(perflibs::imag(z));
}

template <typename T>
using add_complex_t = std::complex<remove_complex_t<T>>;

template <bool b, typename T>
struct add_complex_if {
	using type = add_complex_t<T>;
};

template <typename T>
struct add_complex_if<false, T> {
	using type = T;
};

template <bool b, typename T>
using add_complex_if_t = typename add_complex_if<b, T>::type;

template <typename T1, typename T2>
inline constexpr bool is_r2c_v = !is_complex_v<T1> && is_complex_v<T2>;

template <typename T1, typename T2>
inline constexpr bool is_c2c_v = is_complex_v<T1> && is_complex_v<T2>;

template <typename T1, typename T2>
inline constexpr bool is_c2r_v = is_complex_v<T1> && !is_complex_v<T2>;

template<typename T1, typename T2>
inline constexpr bool is_r2r_v = !is_complex_v<T1> && !is_complex_v<T2>;

template <typename T>
static inline bool is_real_nan(T x) {
	return perflibs::isnan(perflibs::real(x));
}

template <typename T>
static inline bool is_imag_nan(T x) {
	return perflibs::isnan(perflibs::imag(x));
}

static inline half conj(half x) { return x; }
static inline float conj(float x) { return x; }
static inline double conj(double x) { return x; }
static inline std::complex<half> conj(std::complex<half> x) { return std::conj(x); }
static inline std::complex<float> conj(std::complex<float> x) { return std::conj(x); }
static inline std::complex<double> conj(std::complex<double> x) { return std::conj(x); }

namespace literals {
struct cmplx_cnvrt {
	double real, imag;
	inline constexpr operator std::complex<double>() const { return {real, imag}; }
	inline constexpr operator std::complex<float>() const {
		return {static_cast<float>(real), static_cast<float>(imag)};
	}
}; // struct cmplx_cnvrt

static inline constexpr cmplx_cnvrt operator+(double in, const cmplx_cnvrt &cc) {
	return {in + cc.real, cc.imag};
}
static inline constexpr cmplx_cnvrt operator-(double in, const cmplx_cnvrt &cc) {
	return {in - cc.real, -cc.imag};
}
static inline constexpr cmplx_cnvrt operator-(const cmplx_cnvrt &cc) { return {-cc.real, -cc.imag}; }
static inline constexpr cmplx_cnvrt operator""_i(long double in) { return {0.0, static_cast<double>(in)}; }
} // namespace literals
} // namespace perflibs

#endif
