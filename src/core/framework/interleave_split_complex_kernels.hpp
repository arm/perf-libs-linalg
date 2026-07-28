/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_FRAMEWORK_INTERLEAVE_SPLIT_COMPLEX_KERNELS_HPP
#define PERFLIBS_LINALG_FRAMEWORK_INTERLEAVE_SPLIT_COMPLEX_KERNELS_HPP

#include "perflibs_numeric_utils.hpp"
#include "framework/linalg_util.hpp"
#include "perflibs_assert.hpp"

namespace perflibs::linalg {

template<typename FloatType>
PERFLIBS_LINALG_INLINE
void n_interleave_split_complex_ir4(
	std::size_t m, std::size_t n,
	const std::complex<FloatType> *a, std::size_t lda,
	std::size_t dst_m, std::size_t dst_n,
	      std::complex<FloatType> *b, std::size_t ldb) {

	// TODO: make this configurable
	const std::size_t n_interleave_rows = 4;

	//dimensions of b
	const std::size_t nb = iround_div(n, n_interleave_rows);
	const std::size_t mb = m * n_interleave_rows;

	PERFLIBS_ASSERT(m <= lda); //the lda is reasonable for the request
	PERFLIBS_ASSERT(mb <= ldb); //the ldb is reasonable for the request

	for(std::size_t i=0; i < mb; i++) { // row
		for (std::size_t j=0; j < nb; j++) { // col
			const auto ai = i/n_interleave_rows;
			const auto aj = j*n_interleave_rows;

			switch (i%n_interleave_rows) {
				case 0:
					b[j*ldb + i].real(a[aj*lda + ai].real());
					b[j*ldb + i].imag(aj+1 < n ? a[(aj+1)*lda + ai].real() : 0);
					break;
				case 1:
					b[j*ldb + i].real(aj+2 < n ? a[(aj+2)*lda + ai].real() : 0);
					b[j*ldb + i].imag(aj+3 < n ? a[(aj+3)*lda + ai].real() : 0);
					break;
				case 2:
					b[j*ldb + i].real(a[aj*lda + ai].imag());
					b[j*ldb + i].imag(aj+1 < n ? a[(aj+1)*lda + ai].imag() : 0);
					break;
				case 3:
					b[j*ldb + i].real(aj+2 < n ? a[(aj+2)*lda + ai].imag() : 0);
					b[j*ldb + i].imag(aj+3 < n ? a[(aj+3)*lda + ai].imag() : 0);
					break;
			}
		}
	}
}

template<typename FloatType>
PERFLIBS_LINALG_INLINE
void t_interleave_split_complex_ir4(
	std::size_t am, std::size_t an,
	const std::complex<FloatType> *a, std::size_t lda,
	std::size_t dstm, std::size_t dstn,
	      std::complex<FloatType> *b, std::size_t ldb) {

	// TODO: make this configurable
	const std::size_t n_interleave_rows = 4;

	//dimensions of b
	const std::size_t nb = iround_div(am, n_interleave_rows);
	const std::size_t mb = an * n_interleave_rows;

	for(std::size_t i=0; i < mb; i++) { // row
		for (std::size_t j=0; j < nb; j++) { // col
			const auto ai = j * n_interleave_rows;
			const auto aj = i / n_interleave_rows;

			switch (i%n_interleave_rows) {
				case 0:
					b[j*ldb + i].real(a[aj*lda + ai].real());
					b[j*ldb + i].imag(ai+1 < am ? a[aj*lda + ai+1].real() : 0);
					break;
				case 1:
					b[j*ldb + i].real(ai+2 < am ? a[aj*lda + ai+2].real() : 0);
					b[j*ldb + i].imag(ai+3 < am ? a[aj*lda + ai+3].real() : 0);
					break;
				case 2:
					b[j*ldb + i].real(a[aj*lda + ai].imag());
					b[j*ldb + i].imag(ai+1 < am ? a[aj*lda + ai+1].imag() : 0);
					break;
				case 3:
					b[j*ldb + i].real(ai+2 < am ? a[aj*lda + ai+2].imag() : 0);
					b[j*ldb + i].imag(ai+3 < am ? a[aj*lda + ai+3].imag() : 0);
					break;
			}
		}
	}
}

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_INTERLEAVE_SPLIT_COMPLEX_INTERLEAVE_KERNELS_HPP
