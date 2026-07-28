/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_WHICH
#define PERFLIBS_LINALG_WHICH

#include <type_traits>

namespace perflibs {
namespace linalg {

enum class which_matrix : std::uint8_t {
	a,
	b,
	c,
}; //enum class which_matrix

enum class which_dimension : std::uint8_t {
	a_strd,
	b_strd,
	cntg,
}; //enum class which_dimension

enum class which_parallel_strat : std::uint8_t {
	/**
	 * Each thread processes a single panel across the dimension we're parallelising over.
	 */
	general,

	/**
	 * Each thread processes two panels at opposite ends of the matrix in the dimension we're parallelising
	 * over. Gives better thread balancing for triangular matrices.
	 */
	triangular,
}; // enum class which_parallel_strat

enum class which_reduction_strategy {
	serial,
	parallel,
}; // enum class reduction_strategy

template<which_matrix WhichMatrix>
using which_matrix_constant = std::integral_constant<which_matrix, WhichMatrix>;

using a_matrix_t = which_matrix_constant<which_matrix::a>;
using b_matrix_t = which_matrix_constant<which_matrix::b>;
using c_matrix_t = which_matrix_constant<which_matrix::c>;

constexpr a_matrix_t a_matrix;
constexpr b_matrix_t b_matrix;
constexpr c_matrix_t c_matrix;


template<which_dimension WhichDimension>
using which_dimension_constant = std::integral_constant<which_dimension, WhichDimension>;

using a_strd_t = which_dimension_constant<which_dimension::a_strd>;
using b_strd_t = which_dimension_constant<which_dimension::b_strd>;
using cntg_t   = which_dimension_constant<which_dimension::cntg>;

constexpr a_strd_t a_strd;
constexpr b_strd_t b_strd;
constexpr cntg_t   cntg;


template<which_parallel_strat WhichParallelStrat>
using which_parallel_strat_constant = std::integral_constant<which_parallel_strat, WhichParallelStrat>;

using general_parallel_strat_t    = which_parallel_strat_constant<which_parallel_strat::general>;
using triangular_parallel_strat_t = which_parallel_strat_constant<which_parallel_strat::triangular>;

constexpr general_parallel_strat_t    general_parallel_strat;
constexpr triangular_parallel_strat_t triangular_parallel_strat;


template<which_reduction_strategy WhichReductionStrategy>
using which_reduction_strat_constant = std::integral_constant<which_reduction_strategy, WhichReductionStrategy>;

using serial_reduction_strat_t   = which_reduction_strat_constant<which_reduction_strategy::serial>;
using parallel_reduction_strat_t = which_reduction_strat_constant<which_reduction_strategy::parallel>;

constexpr serial_reduction_strat_t   serial_reduction_strat;
constexpr parallel_reduction_strat_t parallel_reduction_strat;


template<which_matrix WhichMatrix, which_dimension WhichDimension>
struct dimension_belongs_to_matrix : std::bool_constant<
	(WhichMatrix == which_matrix::a && WhichDimension != b_strd ) ||
	(WhichMatrix == which_matrix::b && WhichDimension != a_strd ) ||
	(WhichMatrix == which_matrix::c && WhichDimension != cntg ) > {};

template<which_matrix WhichMatrix, which_dimension WhichDimension>
constexpr bool dimension_belongs_to_matrix_v =
	dimension_belongs_to_matrix<WhichMatrix, WhichDimension>::value;

/**
 * Given a matrix and a dimension from the GEMM contract,
 * assert that the dimensions belongs to that matrix
 */
template<which_matrix WhichMatrix, which_dimension WhichDimension>
struct matrix_dimension_assert {
	static_assert(dimension_belongs_to_matrix_v<WhichMatrix, WhichDimension>, "specified dimension does not belong to specified matrix");
}; // struct matrix_dimension_assert


/**
 * LINALG problems normally take the form of C += AB, where
 *
 * C.cntg() == A.strd() //a_strd
 * C.strd() == B.strd() //b_strd
 * A.cntg() == B.cntg() //cntg
 *
 * Given a matrix (WhichMatrix) and a dimension, this meta function returns the corresponding dimension, ie
 *
 * static_assert(corresponding_dimension_v<which_matrix::a, which_dimension::a_strd> == which_dimension::cntg);
 */
template<which_matrix WhichMatrix, which_dimension WhichDimension>
struct corresponding_dimension : matrix_dimension_assert<WhichMatrix, WhichDimension> {
	constexpr static which_dimension value
		= WhichMatrix == which_matrix::a ? ( WhichDimension == which_dimension::a_strd ? which_dimension::cntg   : which_dimension::a_strd )
		: WhichMatrix == which_matrix::b ? ( WhichDimension == which_dimension::b_strd ? which_dimension::cntg   : which_dimension::b_strd )
		: /* which_matrix::c */            ( WhichDimension == which_dimension::a_strd ? which_dimension::b_strd : which_dimension::a_strd );
};

template<which_matrix WhichMatrix, which_dimension WhichDimension>
constexpr which_dimension corresponding_dimension_v =
	corresponding_dimension<WhichMatrix, WhichDimension>::value;

} //namespace linalg

} //namespace perflibs

#endif //PERFLIBS_LINALG_WHICH
