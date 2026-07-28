/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "matrix/matrix.hpp"

#include <print_matrix.hpp>
#include <ostream>

/**
 * To aid development and debugging
 *
 * print the LINALG matrix types can be used as a LINALG stack element
 */


namespace perflibs::linalg {

inline bool print_dimension = true;
inline bool print_contents = false;
inline bool print_contents_row_major = false;

template<typename MatType>
std::ostream& print_matrix_dimension(std::ostream& os, const MatType& mat) {
	return os << " cntg: "      << mat.cntg()
	          << " strd: "      << mat.strd()
	          << " is_trans: "  << mat.is_trans()
	          << " orig_cntg: " << mat.orig_cntg()
	          << " orig_strd: " << mat.orig_strd()
	          << " is_physical: " << mat.is_physical();
}


template<typename T>
std::ostream& print_matrix_contents(std::ostream& os, const general_unpacked_matrix<T>& mat) {
	auto matrix_order = print_contents_row_major ^ mat.is_trans()
	                  ? matrix_order::row_major
	                  : matrix_order::col_major;

	return ::perflibs::print_matrix(
		os,
		mat.orig_cntg(),
		mat.orig_strd(),
		mat.data(),
		mat.stride(),
		matrix_order
	);
}

template<typename MatType>
std::ostream& print_matrix(std::ostream& os, const MatType& mat) {
	if(print_dimension) print_matrix_dimension(os, mat) << "\n";
	if(print_contents) print_matrix_contents(os << "\n", mat) << "\n";
	return os;
}


/**
 * Linalg stack element, conforms to the next_(a, b, d, ...)
 * interface, prints out the details about the matrices
 */
auto make_print(std::ostream& os) {
	return [&](const auto& a, const auto& b, const auto& c, const auto&... args) {
		os << "A: ";
		print_matrix(os, a);
		os << "B: ";
		print_matrix(os, b);
		os << "C: ";
		print_matrix(os, c);
		os << "\n";
	};
}

} //namespace perflibs::linalg
