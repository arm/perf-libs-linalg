#include "pl_linalg_int.h"
#include <stddef.h>

#if (defined(__clang__) || __GNUC__ > 7) && ! defined(__APPLE__)
typedef size_t fortran_charlen_t;
#else
typedef int fortran_charlen_t;
#endif

/*
   Converted from Netlib dgemm.f using f2c with all double precision variables changed to __fp16 and all dependencies on the f2c runtime library removed.
   Derived from Netlib LAPACK sources distributed under BSD-3-Clause-Open-MPI; see netlib/LICENSE and THIRD_PARTY_NOTICES.md.
 */

/*  -- translated by f2c (version 19940927). */

/* Subroutine */ int hgemm_reference_(char *transa, char *transb, pl_linalg_int_t *m, pl_linalg_int_t *
	n, pl_linalg_int_t *k, __fp16 *alpha, __fp16 *a, pl_linalg_int_t *lda,
	__fp16 *b, pl_linalg_int_t *ldb, __fp16 *beta, __fp16 *c, pl_linalg_int_t
	*ldc)
{


    /* Local variables */
    static pl_linalg_int_t info;
    static int nota, notb;
    static __fp16 temp;
    static pl_linalg_int_t i, j, l;
    extern int lsame_(char *, char *);
    static pl_linalg_int_t nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, pl_linalg_int_t *, fortran_charlen_t str_len);


/*  Purpose
    =======

    HGEMM  performs one of the matrix-matrix operations

       C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )

    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.


    Parameters
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in

             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op( A ) = A.

                TRANSA = 'T' or 't',  op( A ) = A'.

                TRANSA = 'C' or 'c',  op( A ) = A'.

             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in

             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op( B ) = B.

                TRANSB = 'T' or 't',  op( B ) = B'.

                TRANSB = 'C' or 'c',  op( B ) = B'.

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix

             op( A )  and of the  matrix  C.  M  must  be at least  zero.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix

             op( B ) and the number of columns of the matrix C. N must be

             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix

             op( A ) and the number of rows of the matrix op( B ). K must

             be at least  zero.
             Unchanged on exit.

    ALPHA  - HALF PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - HALF PRECISION array of DIMENSION ( LDA, ka ), where ka is

             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k

             part of the array  A  must contain the matrix  A,  otherwise

             the leading  k by m  part of the array  A  must contain  the

             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared

             in the calling (sub) program. When  TRANSA = 'N' or 'n' then

             LDA must be at least  max( 1, m ), otherwise  LDA must be at

             least  max( 1, k ).
             Unchanged on exit.

    B      - HALF PRECISION array of DIMENSION ( LDB, kb ), where kb is

             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n

             part of the array  B  must contain the matrix  B,  otherwise

             the leading  n by k  part of the array  B  must contain  the

             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared

             in the calling (sub) program. When  TRANSB = 'N' or 'n' then

             LDB must be at least  max( 1, k ), otherwise  LDB must be at

             least  max( 1, n ).
             Unchanged on exit.

    BETA   - HALF PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is

             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C      - HALF PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must

             contain the matrix  C,  except when  beta  is zero, in which

             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix

             ( alpha*op( A )*op( B ) + beta*C ).

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared

             in  the  calling  (sub)  program.   LDC  must  be  at  least

             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.



       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not

       transposed and set  NROWA, NCOLA and  NROWB  as the number of rows

       and  columns of  A  and the  number of  rows  of  B  respectively.



   Parameter adjustments
       Function Body */

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
	xerbla_("HGEMM ", &info, 5);
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

/*     End of HGEMM . */

} /* hgemm_reference */
