/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SEQUENCE_HPP
#define PERFLIBS_LINALG_SEQUENCE_HPP

#include "framework/compute_position.hpp"
#include "matrix/type_traits.hpp"

namespace perflibs::linalg {

/**
 * Base template. Equivalent to a NO-OP.
 */
template<typename...>
class sequence {
public:
	template<typename... Args>
	PERFLIBS_LINALG_INLINE void operator()(Args&&...) const { }
};

/**
 * LINALG stack operator for sequencing a variable number of other LINALG stack operators. The stack operators
 * will be run in sequence, one after the other.
 */
template<typename Next0, typename... Next1>
class sequence<Next0, Next1...> {
	Next0              next0_;
	sequence<Next1...> next1_;

public:
	sequence(Next0 next0, Next1... next1)
	:	next0_ { std::move(next0)    }
	,	next1_ { std::move(next1)... }
	{	}

	/**
	 * Runs the first operator, waits for it to complete, then runs the next operator.
	 *
	 * @tparam AType The type of the matrix `a`.
	 * @tparam BType The type of the matrix `b`.
	 * @tparam CType The type of the matrix `c`.
	 * @param [in,out] a Passed to the next operators which may modify the referenced object.
	 * @param [in,out] b Passed to the next operators which may modify the referenced object.
	 * @param [in,out] c Passed to the next operators which may modify the referenced object.
	 * @param [in] pos The compute position.
	 */
	template<typename AType, typename BType, typename CType, typename... Args>
	PERFLIBS_LINALG_INLINE void operator()(AType &a, BType &b, CType &c, compute_position pos, Args &&... args) {
		next0_(a, b, c, pos, args...);
		next1_(a, b, c, pos, args...);
	}
}; // class sequence

//https://en.cppreference.com/w/cpp/language/class_template_argument_deduction
template<typename Next0, typename... Next1>
sequence(Next0, Next1...) -> sequence<Next0, Next1...>;

} // namespace perflibs::linalg

#endif /* ifndef PERFLIBS_LINALG_SEQUENCE_HPP */
