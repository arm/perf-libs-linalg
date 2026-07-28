*> \brief \b ISAMIN
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       INTEGER FUNCTION ISAMIN(N,SX,INCX)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       REAL SX(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    ISAMIN finds the index of the first element having minimum absolute value.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] SX
*> \verbatim
*>          SX is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of SX
*> \endverbatim
*
*  =====================================================================
      INTEGER FUNCTION ISAMIN_REFERENCE(N,SX,INCX)
*
*  -- Reference BLAS level1 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      REAL SX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      REAL SMIN
      INTEGER I,IX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS
*     ..
      ISAMIN_REFERENCE = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      ISAMIN_REFERENCE = 1
      IF (N.EQ.1) RETURN
      IF (INCX.EQ.1) THEN
*
*        code for increment equal to 1
*
         SMIN = ABS(SX(1))
         DO I = 2,N
            IF (ABS(SX(I)).LT.SMIN) THEN
               ISAMIN_REFERENCE = I
               SMIN = ABS(SX(I))
            END IF
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         IX = 1
         SMIN = ABS(SX(1))
         IX = IX + INCX
         DO I = 2,N
            IF (ABS(SX(IX)).LT.SMIN) THEN
               ISAMIN_REFERENCE = I
               SMIN = ABS(SX(IX))
            END IF
            IX = IX + INCX
         END DO
      END IF
      RETURN
*
*     End of ISAMIN
*
      END
