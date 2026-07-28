/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_TYPES
#define PERFLIBS_LINALG_TYPES

namespace perflibs::linalg {

/**
 * forward declaration of matrix_base
 */
template<typename T>
class matrix_base;

template<typename T>
class banded_matrix_base;

template<typename T>
class interleave_batch_matrix;

/**
 * forward declarations of adaptor types
 */
template<typename MatrixBaseType>
class general_matrix;

template<typename MatrixBaseType>
class symmetric_matrix;

template<typename MatrixBaseType>
class triangular_matrix;

template<typename MatrixBaseType>
class hermitian_matrix;

template<typename MatrixBaseType>
class band_matrix;

template<typename T>
class split_complex_matrix;

template<typename T>
class packed_matrix_base;

/**
 * further adaptors
 */
template<typename MatrixBaseType>
class owning;

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_TYPES
