/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_PCTX_CONTEXT_RUNNER_HPP
#define PERFLIBS_PCTX_CONTEXT_RUNNER_HPP

/*
 * Run one JSON benchmark request and print its result to standard output.
 * Both overloads return after the benchmark finishes.
 *
 * The JSON overload propagates input and execution exceptions. The stream
 * overload reads JSON and returns true if no exception occurs. It reports
 * exceptions to standard error and returns false.
 *
 * A strategy that returns false sets compute_success to false in the JSON result.
 * Exceeding the time budget sets out_of_time to true.
 */

#include <nlohmann/json_fwd.hpp>

#include <iosfwd>

namespace perflibs::linalg {

void run_context_benchmark(const nlohmann::json& j);
bool run_context_benchmark(std::istream& is);

} // namespace perflibs::linalg

#endif
