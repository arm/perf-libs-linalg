/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MV_COMMON_HPP
#define PERFLIBS_LINALG_MV_COMMON_HPP

#include "framework/linalg_util.hpp"
#include "framework/compute_position.hpp"

#include "matrix/matrix.hpp"

namespace perflibs::linalg {

/**
 * Given two vectors adds the source vector into the destination vector using an operation with
 * the same function signature as an axpby
 */
template <typename AxpbyOp>
class vec_accumulate {
	AxpbyOp axpby_op_;
public:
	vec_accumulate(AxpbyOp axpby_op)
	: axpby_op_  { std::move(axpby_op) }
	{  }
	template <typename T, typename... Args>
	void operator()(const general_unpacked_matrix<T>& dst, const general_unpacked_matrix<T>& src, Args&&... args) {
		const auto size = dst.cntg();
		auto src_ptr = src.data();
		auto dst_ptr = dst.data();

		auto src_step = src.cntg_step();
		auto dst_step = dst.cntg_step();

		axpby_op_(size, one<T>, src_ptr, one<T>, dst_ptr, src_step, dst_step);
	}
}; // vec_accumulate

/**
 * Given two vectors adds the source vector into the destination vector using an operation with
 * the same function signature as an axpby
 */
template <typename AxpbyOp>
class vec_accumulate_scaled {
	AxpbyOp axpby_op_;
public:
	vec_accumulate_scaled(AxpbyOp axpby_op)
	: axpby_op_  { std::move(axpby_op) }
	{  }
	template <typename T>
	void operator()(const general_unpacked_matrix<T>& dst, const general_unpacked_matrix<T>& src, compute_position pos, T alpha, T beta) {
		const auto size = dst.cntg();
		auto src_ptr = src.data();
		auto dst_ptr = dst.data();

		auto src_step = src.cntg_step();
		auto dst_step = dst.cntg_step();

		axpby_op_(size, one<T>, src_ptr, beta, dst_ptr, src_step, dst_step);
	}
}; // vec_accumulate_scaled class

} //namespace perflibs::linalg

#endif // PERFLIBS_LINALG_MV_COMMON_HPP
