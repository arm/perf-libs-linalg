#include "pl_linalg_int.h"

/*
   Translated from Netlib lsame.f with dependencies on the Fortran runtime removed.
   Derived from Netlib LAPACK sources distributed under BSD-3-Clause-Open-MPI; see netlib/LICENSE and THIRD_PARTY_NOTICES.md.
 */

static unsigned char uppercase_lsame_char(unsigned char c) {
    const unsigned char zcode = (unsigned char)'Z';

    if (zcode == 90 || zcode == 122) {
        if (c >= 97 && c <= 122) {
            return (unsigned char)(c - 32);
        }
    } else if (zcode == 233 || zcode == 169) {
        if ((c >= 129 && c <= 137) || (c >= 145 && c <= 153) ||
            (c >= 162 && c <= 169)) {
            return (unsigned char)(c + 64);
        }
    } else if (zcode == 218 || zcode == 250) {
        if (c >= 225 && c <= 250) {
            return (unsigned char)(c - 32);
        }
    }

    return c;
}

pl_linalg_int_t lsame_(const char *ca, const char *cb, ...) {
    const unsigned char a = (unsigned char)*ca;
    const unsigned char b = (unsigned char)*cb;

    if (a == b) {
        return 1;
    }

    return uppercase_lsame_char(a) == uppercase_lsame_char(b);
}
