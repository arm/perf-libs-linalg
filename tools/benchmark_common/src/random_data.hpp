/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_BENCHMARK_RANDOM_DATA_HPP
#define PERFLIBS_BENCHMARK_RANDOM_DATA_HPP

/*
 * Shared random fills for benchmark buffers. Callers own and seed the generator
 * and select the buffer range. Consecutive fills continue the same sequence.
 * Use the two-argument overload for the fixed 24-bit [0, 1) mapping, or pass a
 * distribution to retain a caller's range and sampling precision.
 */

#include <complex>
#include <concepts>
#include <random>
#include <span>

namespace perflibs::linalg {

template<std::floating_point T>
void fill_random(const std::span<T> values, std::mt19937& generator) {
	// Use 24 bits so float and double receive the same values in [0, 1).
	// The explicit mapping also avoids library-dependent distribution algorithms.
	for(auto& value : values) value = T(generator() >> 8) * T(0x1p-24);
}

template<typename T, typename Distribution>
void fill_random(const std::span<T> values, std::mt19937& generator, Distribution& distribution) {
	for(auto& value : values) value = static_cast<T>(distribution(generator));
}

template<typename T, typename Distribution>
void fill_random(const std::span<std::complex<T>> values, std::mt19937& generator, Distribution& distribution) {
	// Draw the real component before the imaginary component for each value.
	for(auto& value : values) value = std::complex<T> { distribution(generator), distribution(generator) };
}

} // namespace perflibs::linalg

#endif
