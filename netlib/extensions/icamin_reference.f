*> \brief \b ICAMIN
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       INTEGER FUNCTION ICAMIN(N,CX,INCX)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       COMPLEX CX(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    ICAMIN finds the index of the first element having minimum |Re(.)| + |Im(.)|
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
*> \param[in] CX
*> \verbatim
*>          CX is COMPLEX array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of CX
*> \endverbatim
*
*  =====================================================================
      INTEGER FUNCTION ICAMIN_REFERENCE(N,CX,INCX)
*
*  -- Reference BLAS level1 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      COMPLEX CX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      REAL SMIN
      INTEGER I,IX
*     ..
*     .. External Functions ..
      REAL SCABS1
      EXTERNAL SCABS1
*     ..
      ICAMIN_REFERENCE = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      ICAMIN_REFERENCE = 1
      IF (N.EQ.1) RETURN
      IF (INCX.EQ.1) THEN
*
*        code for increment equal to 1
*
         SMIN = SCABS1(CX(1))
         DO I = 2,N
            IF (SCABS1(CX(I)).LT.SMIN) THEN
               ICAMIN_REFERENCE = I
               SMIN = SCABS1(CX(I))
            END IF
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         IX = 1
         SMIN = SCABS1(CX(1))
         IX = IX + INCX
         DO I = 2,N
            IF (SCABS1(CX(IX)).LT.SMIN) THEN
               ICAMIN_REFERENCE = I
               SMIN = SCABS1(CX(IX))
            END IF
            IX = IX + INCX
         END DO
      END IF
      RETURN
*
*     End of ICAMIN
*
      END
