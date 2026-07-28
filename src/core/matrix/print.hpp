/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_PRINT
#define PERFLIBS_LINALG_MATRIX_PRINT

#include "framework/which.hpp"

#include <ostream>

namespace perflibs::linalg {

/**
 * Prints a matrix to the specified ostream
 */
template<typename MatType>
std::ostream& print(std::ostream& os, const MatType& m) {
	for(kernel_inttype i = 0; i!=m.cntg(); ++i) {
		for(kernel_inttype j = 0; j!=m.strd(); ++j) {
			os << m(i, j) << " ";
		}
		os << "\n";
	}
	return os;
}

/**
 * A linalg stack operator which prints out any of the matrices its operator() receives
 * which matrices are specified in the ctor
 */
template<which_matrix... WhichMatrix>
class matrix_printer {
	std::ostream& os_;
public:
	matrix_printer(std::ostream& os, which_matrix_constant<WhichMatrix>...)
	:	os_ { os }
	{	}

	template<typename AMatType, typename BMatType, typename CMatType, typename... Args>
	void operator()(const AMatType& a, const BMatType& b, const CMatType& c, Args&&... args) const {
		if constexpr (((WhichMatrix == which_matrix::a) || ...)
		            || sizeof...(WhichMatrix) == 0)
			print(os_ << "A matrix: \n\n", a) << std::endl;
		if constexpr (((WhichMatrix == which_matrix::b) || ...)
		            || sizeof...(WhichMatrix) == 0)
			print(os_ << "B matrix: \n\n", b) << std::endl;
		if constexpr (((WhichMatrix == which_matrix::c) || ...)
		            || sizeof...(WhichMatrix) == 0)
			print(os_ << "C matrix: \n\n", c) << std::endl;
	}
}; //class matrix_printer

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_PRINT
