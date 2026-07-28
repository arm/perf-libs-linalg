## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// Template for complex versions of copy-like functions (scal and axpy varieties)
// Note that there is no need to generate complex BLAS *copy themselves - see the
// we handle those in the "real" version of this template.
//
// Assumed input parameters for copy-like functions:
// x0 : size of vectors
// x1 : Pointer to vec0
// x2 : vec0 Stride
// x3 : Pointer to vec1
// x4 : vec1 Stride
// Note that some copy-like kernels differ in the arguments passed in. This
// is handled in the template by moving parameters into the expected registers.


// Process single-complex elements as double (except for compute instructions, which use cprec)
<%
  prec = "d"
  mem = "d"
  x = "x"
  ncshift = "lsl #3"
%>

%for routine in ["scal","scal_out_of_place","axpy", "axpby", "waxpby", "vecadd"]:

  %for cprec in [ "s", "d" ]:

    %for rotation, suffix in [('#90',  '_sve_kernel_fcmla'), ('#270', '_sve_conj_kernel_fcmla')]:

    <%
      ctype = "z" if cprec == "d" else "c"
      cntprec = cprec if cprec == "d" else "w"
      L = lambda name: ".L{}_{}{}{}".format(name, ctype, routine, suffix)
      func_name = ctype + routine + suffix
      shift = 3 if cprec == "d" else 2 # amount to shift a real/imag part by
      cshift = shift + 1 # amount to shift a complex number by
      is_axpb = "axpb" in routine
      is_axp = "axp" in routine
      is_vecadd = "vecadd" in routine
    %>

    ${prologue(func_name)}
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    %if routine == "scal":
      // Copy in-place scal x pointer, incx into "y, incy" regs expected in this impl
      mov x3, x1
      mov x4, x2
    %elif routine == "waxpby":
      // y: x2 -> x3 ; x13 is w
      mov x13, x3
      mov x3, x2
      // incx: x4 -> x2
      mov x2, x4
      // incy: x5 -> x4
      mov x4, x5
      // incw: x6 -> x14
      mov x14, x6
    %elif is_axp or routine == "scal_out_of_place":
      // Put the input parameters in the expected order for axpy
      mov x5, x2
      mov x2, x3
      mov x3, x5
    %endif

    %if not is_vecadd:
      // Broadcast scalar
      %if cprec == "s":
        mov v0.s[1], v1.s[0]
        mov z0.d, z0.d[0]
      %else:
        mov v0.d[1], v1.d[0]
        mov z0.q, z0.q[0]
      %endif
      %if is_axpb:
        %if cprec == "s":
          mov v2.s[1], v3.s[0]
          mov z2.d, z2.d[0]
        %else:
          mov v2.d[1], v3.d[0]
          mov z2.q, z2.q[0]
        %endif
      %endif
    %endif

    %if routine == "waxpby":
      cmp x14, #1
      bne ${L("use_noncontiguous_loop")}
    %endif

    // First check if this is a 'contiguous' copy (i.e. incx and incy are 1)
    cmp x2, #1
    // if incx is strided branch down and check if we want mixed or completely strided
    bne ${L("use_mixed_strides_incy_loop_check")}
    cmp x4, #1
    // We already know incx=1, so just go straight to mixed_stride_incy
    bne ${L("use_mixed_strides_incy_loop")}


    /*******************/
    /* CONTIGUOUS LOOP */
    /*******************/

    mov x5, #1
    lsl x5, x5, #${cshift}

    ptrue p0.b             // ignore predication
    cntb x6, all, mul #2   // 2x vec len in bytes

    lsl x0, x0, #${cshift} // work in bytes

    add x9, x1, x0         // end pointer
    sub x8, x9, x6         // unrolled loop should go no further than x8

    b ${L("contiguous_loop_cond")}

${L("contiguous_loop")}:
    ld1${mem} z4.${prec}, p0/z, [x1]
    ld1${mem} z5.${prec}, p0/z, [x1, #1, mul vl]
    add x1, x1, x6

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
      movi ${prec}7, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec},  p0/z,    [x3]
      ld1${mem} z21.${prec},  p0/z,    [x3, #1, mul vl]
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #90
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #0
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3]
      ld1${mem} z7.${prec}, p0/z, [x3, #1, mul vl]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
        fcadd z7.${cprec}, p0/m, z7.${cprec}, z5.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
        fadd z7.${cprec}, p0/m, z7.${cprec}, z5.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, #0
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} z6.${prec}, p0, [x13]
      st1${mem} z7.${prec}, p0, [x13, #1, mul vl]
      add x13, x13, x6
    %else:
      st1${mem} z6.${prec}, p0, [x3]
      st1${mem} z7.${prec}, p0, [x3, #1, mul vl]
    %endif
    add x3, x3, x6

${L("contiguous_loop_cond")}:
    cmp x1, x8
    // End of the loop, check for exits
    ble ${L("contiguous_loop")}

    cntb x6

    b ${L("contiguous_loop_tail_cond")}

${L("contiguous_loop_tail")}:

    ld1${mem} z4.${prec}, p0/z, [x1]
    add x1, x1, x6

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec},  p0/z,   [x3]
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} {z6.${prec}},  p0,    [x13]
      add x13, x13, x6
    %else:
      st1${mem} {z6.${prec}},  p0,    [x3]
    %endif

    add x3, x3, x6

${L("contiguous_loop_tail_cond")}:
    whilelt p0.b, x1, x9
    // End of the loop, check for exits
    b.first ${L("contiguous_loop_tail")}

    ldp x29, x30, [sp], #16
    ret

${L("use_mixed_strides_incy_loop_check")}:
    cmp x4, #1
    bne ${L("use_noncontiguous_loop")}
    // Otherwise fall into mixed_stride where incx strided, incy=1


    /***************************************/
    /* MIXED STRIDED LOOP INCX !=1, INCY=1 */
    /***************************************/

${L("use_mixed_strides_incx_loop")}:

    // The address offsets for the first vector
    %if cprec == "d":
      lsl x6, x2, #1
      index z16.${prec}, #0, ${x}6
      index z17.${prec}, #1, ${x}6
      zip1 z16.${prec}, z16.${prec}, z17.${prec}
    %else:
      index z16.${prec}, #0, ${x}2
    %endif

    // Multiply increments by the vector size
    cnt${mem} x5
    mul x10, x2, x5

    // From here, x5 is the loop counter, x7 is the number of elements
    // copied in the unrolled loop. We use a loop counter here because the
    // non-contiguous loop handles negative increments, so we can't
    // rely on the ble test as we did in the contig loop above.
    %if cprec == "d":
      mov x7, x5
    %else:
      lsl x7, x5, #1
    %endif
    sub x5, x0, x7

    // Skip setting up the second vectors if we're not entering the main loop, noncontig_loop
    cmp x5, #0
    blt ${L("mixed_strides_incx_loop_tail_start")}

    // The address offsets for the second vector
    %if cprec == "d":
      lsl x6, x2, #1
      index z18.${prec}, ${x}10, ${x}6
      add ${x}10, ${x}10, #1
      index z19.${prec}, ${x}10, ${x}6
      zip1 z18.${prec}, z18.${prec}, z19.${prec}
    %else:
      index z18.${prec}, ${x}10, ${x}2
    %endif

    ptrue p0.b             // ignore predication
    cntb x6, all, mul #2   // 2x vec len in bytes

    mul x10, x2, x6

    b ${L("mixed_strides_incx_loop_cond")}

${L("mixed_strides_incx_loop")}:
    // Copy the element
    ld1${mem} z4.${prec}, p0/z, [x1, z16.${prec}, ${ncshift}]
    ld1${mem} z5.${prec}, p0/z, [x1, z18.${prec}, ${ncshift}]
    add x1, x1, x10

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
      movi ${prec}7, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec},  p0/z,   [x3]
      ld1${mem} z21.${prec},  p0/z,   [x3, #1, mul vl]
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #90
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #0
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3]
      ld1${mem} z7.${prec}, p0/z, [x3, #1, mul vl]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
        fadd z7.${cprec}, p0/m, z7.${cprec}, z5.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, #0
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} z6.${prec}, p0, [x13]
      st1${mem} z7.${prec}, p0, [x13, #1, mul vl]
      add x13, x13, x6
    %else:
      st1${mem} z6.${prec}, p0, [x3]
      st1${mem} z7.${prec}, p0, [x3, #1, mul vl]
    %endif
    sub x5, x5, x7
    add x3, x3, x6

${L("mixed_strides_incx_loop_cond")}:
    cmp x5, #0
    bge ${L("mixed_strides_incx_loop")}

${L("mixed_strides_incx_loop_tail_start")}:
    // unroll by 1 vector, predicate

    // undo the last subtract from above
    add x5, x5, x7

    // double the number of elements to process, since we want the predicate to
    // be in terms of real and imag parts, separately
    lsl x5, x5, #1
    cnt${cntprec} x7 // x7 is the number of real and imag elements processed in each loop iter

    cntb x6  // vec len in bytes

    mul x10, x2, x6

    b ${L("mixed_strides_incx_loop_tail_cond")}

${L("mixed_strides_incx_loop_tail")}:
    // Copy the element
    ld1${mem} z4.${prec}, p0/z, [x1, z16.${prec}, ${ncshift}]
    add x1, x1, x10

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec}, p0/z, [x3]
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} {z6.${prec}}, p0, [x13]
      add x13, x13, x6
    %else:
      st1${mem} {z6.${prec}}, p0, [x3]
    %endif

    sub x5, x5, x7
    add x3, x3, x6

${L("mixed_strides_incx_loop_tail_cond")}:
    whilelt p0.${cprec}, xzr, x5
    b.first ${L("mixed_strides_incx_loop_tail")}

    ldp x29, x30, [sp], #16
    ret

    /*****************************************/
    /* MIXED STRIDED LOOP INCX =1, INCY != 1 */
    /*****************************************/

${L("use_mixed_strides_incy_loop")}:

    // The address offsets for the first vector
    %if cprec == "d":
      lsl x12, x4, #1
      index z17.${prec}, #0, ${x}12
      index z18.${prec}, #1, ${x}12
      zip1 z17.${prec}, z17.${prec}, z18.${prec}
    %else:
      index z17.${prec}, #0, ${x}4
    %endif

    // Multiply increments by the vector size
    cnt${mem} x5
    mul x11, x4, x5

    // From here, x5 is the loop counter, x7 is the number of elements
    // copied in the unrolled loop. We use a loop counter here because the
    // non-contiguous loop handles negative increments, so we can't
    // rely on the ble test as done in the contig loop above.
    %if cprec == "d":
      mov x7, x5
    %else:
      lsl x7, x5, #1
    %endif
    sub x5, x0, x7

    // Skip setting up the second vectors if we're not entering the main loop, noncontig_loop
    cmp x5, #0
    blt ${L("mixed_strides_incy_loop_tail_start")}

    // The address offsets for the second vector
    %if cprec == "d":
      lsl x12, x4, #1
      index z19.${prec}, ${x}11, ${x}12
      add ${x}11, ${x}11, #1
      index z20.${prec}, ${x}11, ${x}12
      zip1 z19.${prec}, z19.${prec}, z20.${prec}
    %else:
      index z19.${prec}, ${x}11, ${x}4
    %endif

    ptrue p0.b             // ignore predication
    cntb x6, all, mul #2   // 2x vec len in bytes

    mul x11, x4, x6

    b ${L("mixed_strides_incy_loop_cond")}

${L("mixed_strides_incy_loop")}:
    // Copy the element
    ld1${mem} z4.${prec}, p0/z, [x1]
    ld1${mem} z5.${prec}, p0/z, [x1, #1, mul vl]
    add x1, x1, x6

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
      movi ${prec}7, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec},  p0/z,   [x3, z17.${prec}, ${ncshift}]
      ld1${mem} z21.${prec},  p0/z,   [x3, z19.${prec}, ${ncshift}]
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #90
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #0
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3, z17.${prec}, ${ncshift}]
      ld1${mem} z7.${prec}, p0/z, [x3, z19.${prec}, ${ncshift}]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
        fadd z7.${cprec}, p0/m, z7.${cprec}, z5.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, #0
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} z6.${prec}, p0, [x13]
      st1${mem} z7.${prec}, p0, [x13, #1, mul vl]
      add x13, x13, x6
    %else:
      st1${mem} z6.${prec}, p0, [x3, z17.${prec}, ${ncshift}]
      st1${mem} z7.${prec}, p0, [x3, z19.${prec}, ${ncshift}]
    %endif
    sub x5, x5, x7
    add x3, x3, x11

${L("mixed_strides_incy_loop_cond")}:
    cmp x5, #0
    bge ${L("mixed_strides_incy_loop")}

${L("mixed_strides_incy_loop_tail_start")}:
    // unroll by 1 vector, predicate

    // undo the last subtract from above
    add x5, x5, x7

    // double the number of elements to process, since we want the predicate to
    // be in terms of real and imag parts, separately
    lsl x5, x5, #1
    cnt${cntprec} x7 // x7 is the number of real and imag elements processed in each loop iter

    cntb x6  // vec len in bytes

    mul x11, x4, x6

    b ${L("mixed_strides_incy_loop_tail_cond")}

${L("mixed_strides_incy_loop_tail")}:
    // Copy the element
    ld1${mem} z4.${prec}, p0/z, [x1]
    add x1, x1, x6

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec}, p0/z, [x3, z17.${prec}, ${ncshift}]
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3, z17.${prec}, ${ncshift}]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} {z6.${prec}}, p0, [x13]
      add x13, x13, x6
    %else:
      st1${mem} {z6.${prec}}, p0, [x3, z17.${prec}, ${ncshift}]
    %endif

    sub x5, x5, x7
    add x3, x3, x11

${L("mixed_strides_incy_loop_tail_cond")}:
    whilelt p0.${cprec}, xzr, x5
    b.first ${L("mixed_strides_incy_loop_tail")}

    ldp x29, x30, [sp], #16
    ret


${L("use_noncontiguous_loop")}:
    /***********************/
    /* NON-CONTIGUOUS LOOP */
    /***********************/

    // The address offsets for the first vector
    %if cprec == "d":
      lsl x6, x2, #1
      lsl x12, x4, #1
      index z16.${prec}, #0, ${x}6
      index z17.${prec}, #1, ${x}6
      zip1 z16.${prec}, z16.${prec}, z17.${prec}
      index z17.${prec}, #0, ${x}12
      index z18.${prec}, #1, ${x}12
      zip1 z17.${prec}, z17.${prec}, z18.${prec}
      %if routine == "waxpby":
        lsl x6, x14, #1
        index z22.${prec}, #0, ${x}6
        index z23.${prec}, #1, ${x}6
        zip1 z22.${prec}, z22.${prec}, z23.${prec}
      %endif
    %else:
      index z16.${prec}, #0, ${x}2
      index z17.${prec}, #0, ${x}4
      %if routine == "waxpby":
        index z22.${prec}, #0, ${x}14
      %endif
    %endif

    // Multiply increments by the vector size
    cnt${mem} x5
    mul x10, x2, x5
    mul x11, x4, x5
    %if routine == "waxpby":
      mul x15, x14, x5
    %endif

    // After this x5 is the loop counter, x7 is the number of elements
    // copied in the unrolled loop. We use a loop counter here because the
    // non-contiguous loop handles negative increments, so we can't
    // rely on the ble test in .Loop_cond as we do in the contig loop above.
    %if cprec == "d":
      mov x7, x5
    %else:
      lsl x7, x5, #1
    %endif
    sub x5, x0, x7

    // Skip setting up the second vectors if we're not entering the main loop, noncontig_loop
    cmp x5, #0
    blt ${L("noncontig_loop_tail_start")}

    // The address offsets for the second vector
    %if cprec == "d":
      lsl x6, x2, #1
      lsl x12, x4, #1
      index z18.${prec}, ${x}10, ${x}6
      add ${x}10, ${x}10, #1
      index z19.${prec}, ${x}10, ${x}6
      zip1 z18.${prec}, z18.${prec}, z19.${prec}
      index z19.${prec}, ${x}11, ${x}12
      add ${x}11, ${x}11, #1
      index z20.${prec}, ${x}11, ${x}12
      zip1 z19.${prec}, z19.${prec}, z20.${prec}
      %if routine == "waxpby":
        lsl x6, x14, #1
        index z23.${prec}, ${x}15, ${x}6
        add ${x}15, ${x}15, #1
        index z20.${prec}, ${x}15, ${x}6
        zip1 z23.${prec}, z23.${prec}, z20.${prec}
      %endif
    %else:
      index z18.${prec}, ${x}10, ${x}2
      index z19.${prec}, ${x}11, ${x}4
      %if routine == "waxpby":
        index z23.${prec}, ${x}15, ${x}14
      %endif
    %endif

    ptrue p0.b             // ignore predication
    cntb x6, all, mul #2   // 2x vec len in bytes

    mul x10, x2, x6
    mul x11, x4, x6
    %if routine == "waxpby":
      mul x15, x14, x6
    %endif

    b ${L("noncontig_loop_cond")}

${L("noncontig_loop")}:
    // Copy the element
    ld1${mem} z4.${prec}, p0/z, [x1, z16.${prec}, ${ncshift}]
    ld1${mem} z5.${prec}, p0/z, [x1, z18.${prec}, ${ncshift}]
    add x1, x1, x10

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
      movi ${prec}7, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec},  p0/z,   [x3, z17.${prec}, ${ncshift}]
      ld1${mem} z21.${prec},  p0/z,   [x3, z19.${prec}, ${ncshift}]
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec},  p0/m, z2.${cprec}, z20.${cprec}, #90
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #0
      fcmla z7.${cprec},  p0/m, z2.${cprec}, z21.${cprec}, #90
    %elif is_axp or is_vecadd:
      ld1${mem} z6.${prec}, p0/z, [x3, z17.${prec}, ${ncshift}]
      ld1${mem} z7.${prec}, p0/z, [x3, z19.${prec}, ${ncshift}]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
        fcadd z7.${cprec}, p0/m, z7.${cprec}, z5.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
        fadd z7.${cprec}, p0/m, z7.${cprec}, z5.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, #0
      fcmla z7.${cprec}, p0/m, z5.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} z6.${prec}, p0, [x13, z22.${prec}, ${ncshift}]
      st1${mem} z7.${prec}, p0, [x13, z23.${prec}, ${ncshift}]
      add x13, x13, x15
    %else:
      st1${mem} z6.${prec}, p0, [x3, z17.${prec}, ${ncshift}]
      st1${mem} z7.${prec}, p0, [x3, z19.${prec}, ${ncshift}]
    %endif
    sub x5, x5, x7
    add x3, x3, x11

${L("noncontig_loop_cond")}:
    cmp x5, #0
    bge ${L("noncontig_loop")}

${L("noncontig_loop_tail_start")}:
    // unroll by 1 vector, predicate

    // undo the last subtract from above
    add x5, x5, x7

    // double the number of elements to process, since we want the predicate to
    // be in terms of real and imag parts, separately
    lsl x5, x5, #1
    cnt${cntprec} x7 // x7 is the number of real and imag elements processed in each loop iter

    cntb x6  // vec len in bytes

    mul x10, x2, x6
    mul x11, x4, x6
    %if routine == "waxpby":
      mul x15, x14, x6
    %endif

    b ${L("noncontig_loop_tail_cond")}

${L("noncontig_loop_tail")}:
    // Copy the element
    ld1${mem} z4.${prec}, p0/z, [x1, z16.${prec}, ${ncshift}]
    add x1, x1, x10

    %if routine == "scal" or is_axpb or routine == "scal_out_of_place":
      movi ${prec}6, #0
    %endif
    %if is_axpb:
      ld1${mem} z20.${prec}, p0/z, [x3, z17.${prec}, ${ncshift}]
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z2.${cprec}, z20.${cprec}, #90
    %elif is_axp or is_vecadd:
        ld1${mem} z6.${prec}, p0/z, [x3, z17.${prec}, ${ncshift}]
    %endif
    %if is_vecadd:
      %if "conj" in suffix:
        fcadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}, ${rotation}
      %else:
        fadd z6.${cprec}, p0/m, z6.${cprec}, z4.${cprec}
      %endif
    %else:
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, #0
      fcmla z6.${cprec}, p0/m, z4.${cprec}, z0.${cprec}, ${rotation}
    %endif

    %if routine == "waxpby":
      st1${mem} {z6.${prec}}, p0, [x13, z22.${prec}, ${ncshift}]
      add x13, x13, x15
    %else:
      st1${mem} {z6.${prec}}, p0, [x3, z17.${prec}, ${ncshift}]
    %endif

    sub x5, x5, x7
    add x3, x3, x11

${L("noncontig_loop_tail_cond")}:
    whilelt p0.${cprec}, xzr, x5
    b.first ${L("noncontig_loop_tail")}

    ldp x29, x30, [sp], #16
    ret
    ${epilogue(func_name)}

    %endfor

  %endfor

%endfor
