/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_WRITE_HPP
#define PERFLIBS_LINALG_MATRIX_WRITE_HPP

namespace perflibs::linalg {

/**
 * tag used to choose overload of `operator()` in `general_matrix`.
 *
 * For an instance, `m` of `general_matrix`:
 * `m(i, j)`        Returns a value and is suitable for reading.
 *                  The value is conjugated if `m.is_conj()`
 * `m(i, j, write)` Returns a reference, and is suitable for writing.
 *                  The reference is to the underlying data, and is never conjugated.
 */
struct write_t {};
constexpr write_t write;

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_WRITE_HPP
