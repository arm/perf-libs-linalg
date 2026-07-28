/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX
#define PERFLIBS_LINALG_MATRIX

#include "owning.hpp"
#include "adaptors.hpp"
#include "operations.hpp"
#include "matrix_base.hpp"
#include "banded_matrix_base.hpp"
#include "packed_matrix_base.hpp"
#include "split_complex_matrix.hpp"

namespace perflibs::linalg {

//these variants reference existing user data
template<typename T>
using general_unpacked_matrix = general_matrix<matrix_base<T>>;

template<typename T>
using triangular_unpacked_matrix = triangular_matrix<matrix_base<T>>;

template<typename T>
using symmetric_unpacked_matrix = symmetric_matrix<matrix_base<T>>;

template<typename T>
using hermitian_unpacked_matrix = hermitian_matrix<matrix_base<T>>;

namespace {
template<typename T>
using general_band_unpacked_matrix = general_matrix<banded_matrix_base<T>>;

template<typename T>
using triangular_band_unpacked_matrix = triangular_matrix<banded_matrix_base<T>>;

template<typename T>
using symmetric_band_unpacked_matrix = symmetric_matrix<banded_matrix_base<T>>;

template<typename T>
using hermitian_banded_unpacked_matrix = hermitian_matrix<banded_matrix_base<T>>;
} // namespace anon

namespace {
template<typename T>
using triangular_packed_matrix = triangular_matrix<packed_matrix_base<T>>;

template<typename T>
using symmetric_packed_matrix = symmetric_matrix<packed_matrix_base<T>>;

template<typename T>
using hermitian_packed_matrix = hermitian_matrix<packed_matrix_base<T>>;

} // namespace anon

//these variants own their own data
template<typename T>
using owning_general_matrix = owning<general_unpacked_matrix<T>>;

template<typename T>
using owning_triangular_matrix = owning<triangular_unpacked_matrix<T>>;

template<typename T>
using owning_symmetric_matrix = owning<symmetric_unpacked_matrix<T>>;

template<typename T>
using owning_hermitian_matrix = owning<hermitian_unpacked_matrix<T>>;

template<typename T>
using owning_general_band_matrix = owning<general_band_unpacked_matrix<T>>;

template<typename T>
using owning_triangular_band_matrix = owning<triangular_band_unpacked_matrix<T>>;

template<typename T>
using owning_symmetric_band_matrix = owning<symmetric_band_unpacked_matrix<T>>;

template<typename T>
using owning_hermitian_band_matrix = owning<hermitian_banded_unpacked_matrix<T>>;

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX
