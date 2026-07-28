#include "pl_linalg_int.h"
#include <stddef.h>

#if (defined(__clang__) || __GNUC__ > 7) && ! defined(__APPLE__)
typedef size_t fortran_charlen_t;
#else
typedef int fortran_charlen_t;
#endif

/*
   Converted from Netlib dgemm.f using f2c with all double precision variables changed to __bf16 and all dependencies on the f2c runtime library removed.
   Derived from Netlib LAPACK sources distributed under BSD-3-Clause-Open-MPI; see netlib/LICENSE and THIRD_PARTY_NOTICES.md.
 */

/*  -- translated by f2c (version 19940927). */

/* Subroutine */ int bgemm_reference_(char *transa, char *transb, pl_linalg_int_t *m, pl_linalg_int_t *
	n, pl_linalg_int_t *k, __bf16 *alpha, __bf16 *a, pl_linalg_int_t *lda,
	__bf16 *b, pl_linalg_int_t *ldb, __bf16 *beta, __bf16 *c, pl_linalg_int_t
	*ldc)
{


    /* Local variables */
    pl_linalg_int_t info;
    int nota, notb;
    __bf16 temp;
    pl_linalg_int_t i, j, l;
    extern int lsame_(char *, char *);
    pl_linalg_int_t nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, pl_linalg_int_t *, fortran_charlen_t str_len);

#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]
#define B(I,J) b[(I)-1 + ((J)-1)* ( *ldb)]
#define C(I,J) c[(I)-1 + ((J)-1)* ( *ldc)]

    nota = lsame_(transa, "N");
    notb = lsame_(transb, "N");
    if (nota) {
	nrowa = *m;
    } else {
	nrowa = *k;
    }
    if (notb) {
	nrowb = *k;
    } else {
	nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! lsame_(transa, "C") && ! lsame_(transa, "T")) {
	info = 1;
    } else if (! notb && ! lsame_(transb, "C") && ! lsame_(transb,
	    "T")) {
	info = 2;
    } else if (*m < 0) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < (nrowa > 1 ? nrowa : 1)) {
	info = 8;
    } else if (*ldb < (nrowb > 1 ? nrowb : 1)) {
	info = 10;
    } else if (*ldc < (*m > 1 ? *m : 1)) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("BGEMM ", &info, 6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || ((*alpha == 0. || *k == 0) && *beta == 1.)) {
	return 0;
    }

/*     And if  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (*beta == 0.) {
	    for (j = 1; j <= *n; ++j) {
		for (i = 1; i <= *m; ++i) {
		    C(i,j) = 0.;
/* L10: */
		}
/* L20: */
	    }
	} else {
	    for (j = 1; j <= *n; ++j) {
		for (i = 1; i <= *m; ++i) {
		    C(i,j) = *beta * C(i,j);
/* L30: */
		}
/* L40: */
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (notb) {
	if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

	    for (j = 1; j <= *n; ++j) {
		if (*beta == 0.) {
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = 0.;
/* L50: */
		    }
		} else if (*beta != 1.) {
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = *beta * C(i,j);
/* L60: */
		    }
		}
		for (l = 1; l <= *k; ++l) {
		    temp = *alpha * B(l,j);
		    for (i = 1; i <= *m; ++i) {
			C(i,j) += temp * A(i,l);
/* L70: */
		    }
/* L80: */
		}
/* L90: */
	    }
	} else {

/*           Form  C := alpha*A'*B + beta*C */

	    for (j = 1; j <= *n; ++j) {
		for (i = 1; i <= *m; ++i) {
		    temp = 0.;
		    for (l = 1; l <= *k; ++l) {
			temp += A(l,i) * B(l,j);
/* L100: */
		    }
		    if (*beta == 0.) {
			C(i,j) = *alpha * temp;
		    } else {
			C(i,j) = *alpha * temp + *beta * C(i,j);
		    }
/* L110: */
		}
/* L120: */
	    }
	}
    } else {
	if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

	    for (j = 1; j <= *n; ++j) {
		if (*beta == 0.) {
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = 0.;
/* L130: */
		    }
		} else if (*beta != 1.) {
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = *beta * C(i,j);
/* L140: */
		    }
		}
		for (l = 1; l <= *k; ++l) {
		    temp = *alpha * B(j,l);
		    for (i = 1; i <= *m; ++i) {
			C(i,j) += temp * A(i,l);
/* L150: */
		    }
/* L160: */
		}
/* L170: */
	    }
	} else {

/*           Form  C := alpha*A'*B' + beta*C */

	    for (j = 1; j <= *n; ++j) {
		for (i = 1; i <= *m; ++i) {
		    temp = 0.;
		    for (l = 1; l <= *k; ++l) {
			temp += A(l,i) * B(j,l);
/* L180: */
		    }
		    if (*beta == 0.) {
			C(i,j) = *alpha * temp;
		    } else {
			C(i,j) = *alpha * temp + *beta * C(i,j);
		    }
/* L190: */
		}
/* L200: */
	    }
	}
    }

    return 0;

/*     End of BGEMM_REFERENCE . */

} /* bgemm_reference */
