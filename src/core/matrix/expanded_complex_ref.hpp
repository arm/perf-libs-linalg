/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_EXPANDED_COMPLEX_REF
#define PERFLIBS_LINALG_EXPANDED_COMPLEX_REF

#include "framework/linalg_util.hpp"

#include <complex>

namespace perflibs::linalg {

/**
 * References the two stored columns of an expanded complex value.
 * Assigning a + bi writes {a, b} and {-b, a}
*/

template<typename Prec>
class expanded_complex_ref {
public:
	using precision_t = Prec;
	using complex_t = std::complex<precision_t>;

	PERFLIBS_LINALG_INLINE
	expanded_complex_ref(complex_t& first_column, complex_t& second_column)
	: first_column_ { first_column }
	, second_column_ { second_column }
	{}

	PERFLIBS_LINALG_INLINE
	expanded_complex_ref& operator=(const complex_t& rhs) {
		const auto real = rhs.real();
		const auto imag = rhs.imag();
		first_column_ = complex_t { real, imag };
		second_column_ = complex_t { -imag, real };
		return *this;
	}

	PERFLIBS_LINALG_INLINE
	operator complex_t() const { return first_column_; }

	PERFLIBS_LINALG_INLINE
	expanded_complex_ref& operator=(const expanded_complex_ref& rhs) {
		return *this = static_cast<complex_t>(rhs);
	}

private:
	complex_t& first_column_;
	complex_t& second_column_;

}; // class expanded_complex_ref

} // namespace perflibs::linalg

#endif // PERFLIBS_LINALG_EXPANDED_COMPLEX_REF
