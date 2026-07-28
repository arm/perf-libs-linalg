/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_BLAS_TYPES_HPP
#define PERFLIBS_BLAS_TYPES_HPP

#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"

namespace perflibs {

struct mat_offset {
	kernel_inttype cntg, strd;
};

enum perflibs_trans {
	PERFLIBS_NOTRANS = 1,
	PERFLIBS_TRANS,
	PERFLIBS_CONJTRANS,
	PERFLIBS_CONJ
};
static inline perflibs_trans c_to_trans(char c) {
	return (c == 'n' || c == 'N') ? PERFLIBS_NOTRANS :
	       (c == 't' || c == 'T') ? PERFLIBS_TRANS :
	       (c == 'c' || c == 'C') ? PERFLIBS_CONJTRANS :
	       (c == 'r' || c == 'R') ? PERFLIBS_CONJ :
	       (perflibs_trans) 0;
}

static inline perflibs_trans transpose(perflibs_trans trans) {
	return trans == PERFLIBS_NOTRANS ? PERFLIBS_TRANS : PERFLIBS_NOTRANS;
}
static inline bool is_trans(perflibs_trans trans) {
	return (trans == PERFLIBS_TRANS || trans == PERFLIBS_CONJTRANS);
}
static inline char trans_to_c(perflibs_trans trans) {
	switch(trans) {
	case PERFLIBS_NOTRANS:   return 'N';
	case PERFLIBS_TRANS:     return 'T';
	case PERFLIBS_CONJTRANS: return 'C';
	case PERFLIBS_CONJ:      return 'R';
	default: PERFLIBS_ASSERT(false);
	}
}
static inline const char *trans_to_c_ptr(perflibs_trans trans) {
	switch(trans) {
	case PERFLIBS_NOTRANS:   return "N";
	case PERFLIBS_TRANS:     return "T";
	case PERFLIBS_CONJTRANS: return "C";
	case PERFLIBS_CONJ:      return "R";
	default: PERFLIBS_ASSERT(false);
	}
}

enum perflibs_side {
	PERFLIBS_LEFT = 1,
	PERFLIBS_RIGHT
};
static inline bool is_left(perflibs_side side) {
	return side == PERFLIBS_LEFT;
}
static inline perflibs_side c_to_side(char c) {
	return (c == 'l' || c == 'L') ? PERFLIBS_LEFT :
	       (c == 'r' || c == 'R') ? PERFLIBS_RIGHT :
	       (perflibs_side) 0;
}
static inline char side_to_c(perflibs_side side) {
	return (side == PERFLIBS_LEFT) ? 'L' : 'R';
}
static inline const char *side_to_c_ptr(perflibs_side side) {
	return (side == PERFLIBS_LEFT) ? "L" : "R";
}


enum perflibs_uplo {
	PERFLIBS_LOWER = 1,
	PERFLIBS_UPPER
};
static inline perflibs_uplo c_to_uplo(char c) {
	return (c == 'l' || c == 'L') ? PERFLIBS_LOWER :
	       (c == 'u' || c == 'U') ? PERFLIBS_UPPER :
	       (perflibs_uplo) 0;
}
static inline char uplo_to_c(perflibs_uplo uplo) {
	return (uplo == PERFLIBS_LOWER) ? 'L' : 'U';
}
static inline const char *uplo_to_c_ptr(perflibs_uplo uplo) {
	return (uplo == PERFLIBS_LOWER) ? "L" : "U";
}

static inline perflibs_uplo lower_flip(perflibs_uplo v) {
	return v == PERFLIBS_LOWER ? PERFLIBS_UPPER : PERFLIBS_LOWER;

}

enum perflibs_diag {
	PERFLIBS_NOUNIT = 1,
	PERFLIBS_UNIT
};
static inline perflibs_diag c_to_diag(char c) {
	return (c == 'n' || c == 'N') ? PERFLIBS_NOUNIT :
	       (c == 'u' || c == 'U') ? PERFLIBS_UNIT :
	       (perflibs_diag) 0;
}
static inline char diag_to_c(perflibs_diag diag) {
	return (diag == PERFLIBS_NOUNIT) ? 'N' : 'U';
}

static inline const char *diag_to_c_ptr(perflibs_diag diag) {
	return (diag == PERFLIBS_NOUNIT) ? "N" : "U";
}

// These enums are required for the use
// of LARFB and LARFT, the can be deduced
// but the rules are a bit complex as these
// routines are used differently from various
// lapack routines
enum perflibs_storev {
	PERFLIBS_COLUMNWISE = 1,
	PERFLIBS_ROWWISE
};
static inline perflibs_storev c_to_storev(char c) {
	return (c == 'c' || c == 'C') ? PERFLIBS_COLUMNWISE :
	       (c == 'r' || c == 'R') ? PERFLIBS_ROWWISE :
	       (perflibs_storev) 0;
}
static inline char storev_to_c(perflibs_storev storev) {
	return (storev == PERFLIBS_COLUMNWISE) ? 'C' : 'R';
}
static inline const char *storev_to_c_ptr(perflibs_storev storev) {
	return (storev == PERFLIBS_COLUMNWISE) ? "C" : "R";
}

enum perflibs_direct {
	PERFLIBS_FORWARD = 1,
	PERFLIBS_BACKWARD
};
static inline perflibs_direct c_to_direct(char c) {
	return (c == 'f' || c == 'F') ? PERFLIBS_FORWARD :
	       (c == 'b' || c == 'B') ? PERFLIBS_BACKWARD :
	       (perflibs_direct) 0;
}
static inline char direct_to_c(perflibs_direct direct) {
	return (direct == PERFLIBS_FORWARD) ? 'F' : 'B';
}
static inline const char *direct_to_c_ptr(perflibs_direct direct) {
	return (direct == PERFLIBS_FORWARD) ? "F" : "B";
}

} // end namespace perflibs

#endif // PERFLIBS_BLAS_TYPES_HPP
