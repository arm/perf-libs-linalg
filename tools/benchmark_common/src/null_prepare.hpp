/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_BENCHMARK_NULL_PREPARE_HPP
#define PERFLIBS_BENCHMARK_NULL_PREPARE_HPP

// No-op preparation callback, callable with or without a context.
namespace perflibs::linalg {

struct null_prepare {
	constexpr void operator()() const noexcept { }

	template<typename Context>
	constexpr void operator()(const Context&) const noexcept { }
};

} // namespace perflibs::linalg

#endif
