/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_SPLIT_COMPLEX_REF
#define PERFLIBS_LINALG_SPLIT_COMPLEX_REF

#include "framework/linalg_util.hpp"

#include <complex>
#include <iosfwd>

namespace perflibs::linalg {

/**
 * A referenced type designed to represent a complex number where
 * the real and imag components are not stored contiguously in memory
 * instead we store references to each and provide accessors to mutate them
 * as though they were a standard std::complex type
 */
template<typename Prec>
class split_complex_ref {
public:
	using precision_t = Prec;
	using complex_t = std::complex<precision_t>;

	PERFLIBS_LINALG_INLINE
	split_complex_ref(precision_t& real, precision_t& imag)
	:	real_ { real }
	,	imag_ { imag }
	{	}

	/**
	 * implicit conversion operator to its std::complex analogue
	 */
	PERFLIBS_LINALG_INLINE
	operator complex_t() const { return { real_, imag_ }; }

	PERFLIBS_LINALG_INLINE
	split_complex_ref& operator=(const complex_t& rhs) {
		real_ = rhs.real();
		imag_ = rhs.imag();

		return *this;
	}
private:
	precision_t& real_;
	precision_t& imag_;
}; //class split_complex_ref

template<typename T>
PERFLIBS_LINALG_INLINE
split_complex_ref<const T> to_const(split_complex_ref<T>& spr) {
	return { spr.real(), spr.imag() };
}

/**
 * Primarily for debugging and completeness
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const split_complex_ref<T>& scr) {
	return os << std::complex<T> { scr };
}

} //namespace perflibs::linalg

#endif // PERFLIBS_LINALG_SPLIT_COMPLEX_REF
