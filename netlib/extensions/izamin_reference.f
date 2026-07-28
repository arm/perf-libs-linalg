*> \brief \b IZAMIN
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       INTEGER FUNCTION IZAMIN(N,ZX,INCX)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       COMPLEX*16 ZX(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    IZAMIN finds the index of the first element having minimum |Re(.)| + |Im(.)|
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
*> \param[in] ZX
*> \verbatim
*>          ZX is COMPLEX*16 array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of ZX
*> \endverbatim
*
*  =====================================================================
      INTEGER FUNCTION IZAMIN_REFERENCE(N,ZX,INCX)
*
*  -- Reference BLAS level1 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      COMPLEX*16 ZX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION DMIN
      INTEGER I,IX
*     ..
*     .. External Functions ..
      DOUBLE PRECISION DCABS1
      EXTERNAL DCABS1
*     ..
      IZAMIN_REFERENCE = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      IZAMIN_REFERENCE = 1
      IF (N.EQ.1) RETURN
      IF (INCX.EQ.1) THEN
*
*        code for increment equal to 1
*
         DMIN = DCABS1(ZX(1))
         DO I = 2,N
            IF (DCABS1(ZX(I)).LT.DMIN) THEN
               IZAMIN_REFERENCE = I
               DMIN = DCABS1(ZX(I))
            END IF
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         IX = 1
         DMIN = DCABS1(ZX(1))
         IX = IX + INCX
         DO I = 2,N
            IF (DCABS1(ZX(IX)).LT.DMIN) THEN
               IZAMIN_REFERENCE = I
               DMIN = DCABS1(ZX(IX))
            END IF
            IX = IX + INCX
         END DO
      END IF
      RETURN
*
*     End of IZAMIN
*
      END
