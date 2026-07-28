/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_NO_OP_HPP
#define PERFLIBS_LINALG_NO_OP_HPP

#include <type_traits>

namespace perflibs::linalg {

/**
 * A linalg operator which does nothing
 */
class no_op {
public:
	//unsurprisingly, does nothing
	template<typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(Args&&...) const { }
}; //class no_op

template<typename T>
constexpr bool is_no_op_v = std::is_same_v<T, no_op>;

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_NO_OP_HPP
