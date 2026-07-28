/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_TYPE_TRAITS_HPP
#define PERFLIBS_LINALG_MATRIX_TYPE_TRAITS_HPP

#include "matrix/types.hpp"
#include "matrix/write.hpp"

#include <complex>
#include <type_traits>
#ifndef _WIN32
#include <experimental/type_traits>
#endif

namespace perflibs::linalg {

/**
 * is value_type typedef member in Class the same as T
 */
template<typename Class, typename T>
struct value_type_is : std::is_same<typename Class::value_type, T> { };

template<typename Class, typename T>
constexpr bool value_type_is_v = value_type_is<Class, T>::value;

/**
 * is value_type typedef member in Class on of a range of types
 */
template<typename Class, typename... Args>
//constexpr bool value_type_is_one_of_v = ( ... || value_type_is_v<Class, Args> );
constexpr bool value_type_is_one_of_v = ( value_type_is_v<Class, Args> || ... );

/**
 * This trait detects whether or not a matrix type has an `is_lower` function
 * the presence of such a function is a good indicator that it has triangular form
 */
template<class T>
using is_lower_test = decltype(std::declval<T>().is_lower());

#ifdef _WIN32
/**
 * Windows builds use the MSVC standard library, which does not provide
 * std::experimental::is_detected, so define our own.
 */
namespace detail {
	/*
	 * The default implementation
	 */
	template<typename Void, template <typename...> typename Check, typename... CheckArgs>
	struct is_detected_impl : std::false_type { };

	/**
	 * Using SFINAE the compiler is going to try and piece together
	 *     Check<CheckArgs...>
	 *
	 * So for example, we might have a test which gets the type of a member function `foo()`
	 *
	 *     has_foo< struct_with_foo_t >
	 *
	 * When `struct_with_foo_t` indeed does have `foo()`, then `has_foo` will successfully
	 * be able to determine the return type of `foo()`.
	 *
	 * Because we don't really care what that type is, we then "throw it away" by
	 * dumping that successfully retrieved results into std::void_t<...> and in
	 * doing so we tell the compiler that this specialized structs interface is:
	 *
	 *     struct<void, Check, CheckArgs...>
	 *            ^
	 *       from std::void_t<Check<CheckArgs...>
	 *
	 * And the compiler will then select this version of the default version, this version
	 * derives from the base type `std::true_type`
	 *
	 * If the struct did not have the `foo()` member function, then `Check<CheckArgs...>` would
	 * fail to compile because the compiler could not find the return type of the function
	 * which did not exist.
	 *
	 * This in turn means that our template parameter to std::void_t  could not be determined.
	 * This means that this struct candidate would not be valid for the these template parameters.
	 * The compiler would then instead pick the more permissive and less general default version
	 * of this struct instead, which derives from the std::false_type base class.
	 */
	template<template <typename...> typename Check, typename... CheckArgs>
	struct is_detected_impl
	<
		/* if Check<CheckArgs...> is valid, then `std::void_t` match with `void`
		 * if it is not valid, then this specialization will not be valid for these
		 * give input parameters and the compiler will instead have to use the less specific
		 * default implementation
		 */
		std::void_t<Check<CheckArgs...>>,
		Check,
		CheckArgs...
	> : std::true_type { };
}

template<template<typename ...> typename Check, typename... CheckArgs>
using is_detected = detail::is_detected_impl<void, Check, CheckArgs...>;

template<template<typename ...> typename Check, typename... CheckArgs>
constexpr bool is_detected_v = is_detected<Check, CheckArgs...>::value;

/**
 * If we detect that a matrix has an `is_lower()` member function
 */
template<typename MatType>
constexpr auto is_triangular_form_v =
	perflibs::linalg::is_detected_v<is_lower_test, MatType>;
#else
/**
 * We can use std::experimental::is_detected
 */

/**
 * If we detect that a matrix has an `is_lower()` member function
 */
template<typename MatType>
constexpr auto is_triangular_form_v =
	std::experimental::is_detected_v<is_lower_test, MatType>;
#endif

/**
 * detect whether T is a matrix adaptor
 */
template<typename T>
constexpr bool is_matrix_adaptor_v = std::is_base_of_v<matrix_base<typename T::value_type>, T> ||
                                     std::is_base_of_v<banded_matrix_base<typename T::value_type>, T>;

/**
 * detect whether  or not T is a general_matrix
 */
template<typename T>
struct is_general_matrix : std::false_type {};
template<typename T>
struct is_general_matrix<general_matrix<T>> : std::true_type {};
template<typename T>
constexpr bool is_general_matrix_v = is_general_matrix<T>::value;

/**
 * detect whether  or not T is a symmetric_matrix
 */
template<typename T>
struct is_symmetric_matrix : std::false_type {};
template<typename T>
struct is_symmetric_matrix<symmetric_matrix<T>> : std::true_type {};
template<typename T>
constexpr bool is_symmetric_matrix_v = is_symmetric_matrix<T>::value;

/**
 * detect whether  or not T is a hermitian_matrix
 */
template<typename T>
struct is_hermitian_matrix : std::false_type {};
template<typename T>
struct is_hermitian_matrix<hermitian_matrix<T>> : std::true_type {};
template<typename T>
constexpr bool is_hermitian_matrix_v = is_hermitian_matrix<T>::value;

/**
 * detect whether  or not T is a triangular_matrix
 */
template<typename T>
struct is_triangular_matrix : std::false_type {};
template<typename T>
struct is_triangular_matrix<triangular_matrix<T>> : std::true_type {};
template<typename T>
constexpr bool is_triangular_matrix_v = is_triangular_matrix<T>::value;

/**
 * detect whether  or not T is a split_complex_matrix
 */
template<typename T>
struct is_split_complex_matrix : std::false_type { };
template<typename T>
struct is_split_complex_matrix<split_complex_matrix<T>> : std::true_type { };
template<typename T>
constexpr bool is_split_complex_matrix_v = is_split_complex_matrix<T>::value;


/**
 * Detect whether the result of calling operator() is an l-value reference
 * and therefore assignment is meaningful
 */
template<typename T>
struct is_result_of_call_reference : std::is_lvalue_reference<
	decltype(std::declval<T>()(kernel_inttype{0}, kernel_inttype{0}, write))> {};

template<typename T>
constexpr bool is_result_of_call_reference_v = is_result_of_call_reference<T>::value;

template<typename T>
constexpr bool is_result_of_callable_assignable_v = is_result_of_call_reference<T>::value;


/**
 * Get the inner most type of a nested template class
 */
template<typename T>
struct inner_most;

template<template <typename> class T0, typename T1>
struct inner_most<T0<T1>> {
	///the inner most type
    using type = T1;

	///everything else that wraps that inner most type
    template<typename T>
    using rest = T0<T>;
};


template<template <typename> class T0, typename T1>
struct inner_most<T0<std::complex<T1>>> {
	///the inner most type
    using type = std::complex<T1>;

	///everything else that wraps that inner most type
    template<typename T>
    using rest = T0<T>;
};




template<template <typename> class T0, template <typename> class T1, typename T2>
struct inner_most<T0<T1<T2>> > {
    using type  = typename inner_most<T1<T2>>::type;

    template<typename T>
    using rest = T0<typename inner_most<T1<T2>>::template rest<T>>;
};

template<typename T>
using inner_most_value_type_t = typename inner_most<T>::type;

template<typename T, typename T0>
using inner_most_rest_t = typename inner_most<T>::template rest<T0>;

//given a matrix of value_type T, produce the same matrix but of value_type `const T`
template<typename T>
using to_const_t = inner_most_rest_t<T, const inner_most_value_type_t<T>>;


template<typename T>
using remove_const_t = inner_most_rest_t<T, std::remove_cv_t<inner_most_value_type_t<T>>>;


} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_TYPE_TRAITS_HPP
