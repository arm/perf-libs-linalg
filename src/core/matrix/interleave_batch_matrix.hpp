/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_INTERLEAVE_BATCH_MATRIX
#define PERFLIBS_LINALG_INTERLEAVE_BATCH_MATRIX

#include "adaptors.hpp"
#include "interleave_batch_matrix_base.hpp"

namespace perflibs::linalg {

template<typename T>
using general_interleave_batch_matrix = general_matrix<interleave_batch_matrix<T>>;

template<typename T>
using triangular_interleave_batch_matrix = triangular_matrix<interleave_batch_matrix<T>>;

template<typename T>
using symmetric_interleave_batch_matrix = symmetric_matrix<interleave_batch_matrix<T>>;

template<typename T>
using hermitian_interleave_batch_matrix = hermitian_matrix<interleave_batch_matrix<T>>;

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_INTERLEAVE_BATCH_MATRIX
