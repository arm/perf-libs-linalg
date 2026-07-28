/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_VECTOR_LENGTH_HPP
#define PERFLIBS_LINALG_FRAMEWORK_VECTOR_LENGTH_HPP

#include "framework/linalg_util.hpp"

#include <cstddef>

namespace perflibs::linalg {

enum class extensions {
	neon = 0, //i.e. none
	sve = 1,
	streaming_sve = 2,
	sme2 = 4
};

using vector_length_value_f = kernel_inttype (*)();

struct vector_length_value {
	vector_length_value_f expr;
	kernel_inttype value;

	constexpr vector_length_value()
	:	expr { nullptr }
	,	value { 0 }
	{	}

	constexpr vector_length_value(vector_length_value_f expr)
	:	expr { expr }
	,	value { 0 }
	{	}

	constexpr vector_length_value(kernel_inttype value)
	:	expr { nullptr }
	,	value { value }
	{	}

	PERFLIBS_LINALG_INLINE
	kernel_inttype operator()() const {
		return expr ? expr() : value;
	}
};

std::size_t vector_length_bytes_neon();
std::size_t vector_length_bytes_sve();
std::size_t vector_length_bytes_sme();

/*
 * These functions are in an anon namespace because it is critical that the
 * SVE and neon versions end up with their own copy in respective translation
 * units (.o file).
 */
namespace {

template<kernel_inttype Value>
PERFLIBS_LINALG_INLINE
static
kernel_inttype intval() { return Value; }

template<extensions Extension, typename DataType, kernel_inttype Multiplier>
PERFLIBS_LINALG_INLINE
kernel_inttype vl() {
	const auto vlb = Extension == extensions::sme2 ? vector_length_bytes_sme()
	               : Extension == extensions::sve  ? vector_length_bytes_sve()
	                                               : vector_length_bytes_neon();

	return vlb / sizeof(DataType) * Multiplier;
}

} //namespace <anon>

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_VECTOR_LENGTH_HPP
