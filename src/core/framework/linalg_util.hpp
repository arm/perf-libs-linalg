/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_FRAMEWORK_LINALG_UTIL_HPP
#define PERFLIBS_FRAMEWORK_LINALG_UTIL_HPP

#include "perflibs_util.hpp"
#include "perflibs_numeric_utils.hpp"

#include "perflibs_complex.hpp" //remove_complex_t
#include "perflibs_assert.hpp"
#include "framework/linalg_blas_types.hpp"

#include <array>
#include <type_traits>
#include <cstdio>

#ifdef PERFLIBS_LINALG_DEBUG
	// Turning off aggressive inline makes debugging much easier
	#define PERFLIBS_LINALG_INLINE inline
#else
	#define PERFLIBS_LINALG_INLINE inline __attribute__((always_inline))
#endif //PERFLIBS_LINALG_DEBUG

namespace perflibs::linalg {

template<int N>
struct auto_compare {
	constexpr auto_compare(int n) {
		PERFLIBS_ASSERT(n == N, "n must match N");
	}

	template<typename T>
	operator T () const {
		return T { N };
	}
}; //class auto_compare

template<int N, typename T>
inline
bool operator==(auto_compare<N>, const T& v) {
	return T{ N } == v;
}

template<int N, typename T>
inline
bool operator==(const T& v, auto_compare<N>) {
	return v == T{ N };
}

} //namespace perflibs::linalg

template<typename T = perflibs::linalg::auto_compare<0>>
constexpr static auto zero = T { 0 };

template<typename T = perflibs::linalg::auto_compare<1>>
constexpr static auto one = T { 1 };


namespace perflibs::linalg {
namespace {
/**
 * Aligns a pointer to a given byte alignment, ensuring that the pointer is not also aligned
 * to double the alignment parameter. An alignment of zero will return the original pointer.
 * @param [in,out] ptr         The pointer to align
 * @param [in]     alignment   The byte alignment to align to
 * @returns a pointer to the next highest address with the given byte alignment, which is not
 *          aligned to double the byte alignment
 */
template <typename T>
inline T* align_to(T* ptr, unsigned int alignment) {
	if (alignment == 0)
		return ptr;
	auto aligned_ptr = iround(reinterpret_cast<intptr_t>(ptr), alignment);
	if (aligned_ptr % (2 * alignment) == 0) {
		aligned_ptr += alignment;
	}
	return reinterpret_cast<T*>(aligned_ptr);
};
template<kernel_inttype Value>
struct step_val_fixed {
	constexpr PERFLIBS_LINALG_INLINE
	operator kernel_inttype() const {
		return Value;
	}
}; //struct step_val


template<typename T, int Value>
struct constexpr_float_val  {
	using type = T;

	constexpr PERFLIBS_LINALG_INLINE
	operator type() const { return static_cast<T>(Value); }
};

template<typename T> using float_one  = constexpr_float_val<T, 1>;
template<typename T> using float_zero = constexpr_float_val<T, 0>;

template<typename T>
struct compute_type {
	using type = T;
};

template<typename T, int Val>
struct compute_type< constexpr_float_val<T, Val> > {
	using type = typename constexpr_float_val<T, Val>::type;
};

template<typename T>
using compute_type_t = typename compute_type<T>::type;


template<typename T>
constexpr PERFLIBS_LINALG_INLINE
T operator*(T v, float_one<T>) {
	return v;
}

template<typename T>
constexpr PERFLIBS_LINALG_INLINE
T operator*(T v, float_zero<T>) {
	return zero<T>;
}
} //anon namespace

/**
 * type is the inferred scalar type from matrix/vector input types in accordance
 * with Next Gen BLAS inference rules
 */
template<typename... Types>
struct promote {
	using precision_t = std::common_type_t< remove_complex_t<std::remove_cv_t<Types>>...>;

	using type = std::conditional_t<
		( is_complex_v<std::remove_cv_t<Types>> || ... ), std::complex<precision_t>, precision_t>;
}; // struct promote

template<typename... Types>
using promote_t = typename promote<Types...>::type;

enum class value_support {
	one      =0,
	zero     =1,
	not_zero =2,
	real     =3,
	all      =4
};

enum class zero_mode : bool {
	set   = false,
	scale = true,
};

PERFLIBS_LINALG_INLINE
bool is_conj(perflibs_trans trans) {
	return trans == PERFLIBS_CONJ
	    || trans == PERFLIBS_CONJTRANS;
}

template <typename Type, std::size_t... Sizes>
PERFLIBS_LINALG_INLINE
auto array_concat(const std::array<Type, Sizes>&... arrays) {
	std::array<Type, (Sizes + ...)> result;
	std::size_t index{};

	((std::copy_n(arrays.begin(), Sizes, result.begin() + index), index += Sizes), ...);

	return result;
}

enum class matrix_requirement {
	any, cntg_one=1, strd_one=2
};

} //namespace perflibs::linalg

#endif //PERFLIBS_FRAMEWORK_LINALG_UTIL_HPP
