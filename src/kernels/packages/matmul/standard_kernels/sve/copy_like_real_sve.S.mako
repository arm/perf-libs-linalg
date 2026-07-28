## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

// Template for real versions of copy-like functions (copy, scal and axpy varieties)
//
// Assumed input parameters for copy-like functions:
// x0 : size of vectors
// x1 : Pointer to vec0
// x2 : vec0 Stride
// x3 : Pointer to vec1
// x4 : vec1 Stride
// Note that some copy-like kernels differ in the arguments passed in. This
// is handled in the template by moving parameters into the expected registers.

<%
  L = lambda name: ".L{}_{}{}".format(name, prec, routine)
%>

## Routine name to operation mapping:
## copy : y = x
## scal : x = a * x
## scal_out_of_place : y = a * x
## scal_real_cplx : x = a * x, where a is real
## axpy : y = a * x + y
## axpby : y = a * x + b * y
## waxpby : w = a * x + b * y
## vecacc : y = x + y

%for routine in ["copy", "scal", "scal_out_of_place", "scal_real_cplx", "axpy", "axpby", "waxpby", "vecadd"]:

  %for prec, mem, x, shift, ncshift in [ ("s", "w", "w", "#2", "sxtw #2"), ("d", "d", "x", "#3", "lsl #3") ]:

    <%
      if routine != "scal_real_cplx":
        prec_load = prec
      else:
        prec_load = "d"
        shift = "#3" if prec == "s" else "#4"
        x, mem, ncshift = "x", "d", "lsl #3"

      func_name = prec + routine + "_sve_kernel"
      is_axp = "axp" in routine
      is_axpb = "axpb" in routine
      is_scal = "scal" in routine
      is_vecadd = "vecadd" in routine
    %>

    ${prologue(func_name)}
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    // Put the input parameters in the expected order for axpy!
    %if routine == "waxpby":
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
      mov x5, x2
      mov x2, x3
      mov x3, x5
    %endif

    %if routine == "scal" or routine == "scal_real_cplx":
      // Copy x pointer, incx into "y, incy" regs expected in this impl
      mov x3, x1
      mov x4, x2
    %endif

    %if routine != "copy" or not is_vecadd:
      // Broadcast scalar
      mov z0.${prec}, ${prec}0
      %if is_axpb:
        mov z1.${prec}, ${prec}1
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
    // We already know incx is not strided so just go straight to mixed_stride_incy
    bne ${L("use_mixed_strides_incy_loop")}


    /*******************/
    /* CONTIGUOUS LOOP */
    /*******************/

    ptrue p0.${prec}       // ignore predication

    lsl x0, x0, ${shift}
    add x9, x1, x0         // end pointer

// unroll by 2 vectors
    cntb x6, all, mul #2   // 2x vec len in bytes
    sub x8, x9, x6         // unrolled loop should go no further than x8

%if routine == "scal" or routine == "copy" or routine == "scal_out_of_place":
    // Get the vector length
    cntb x5
    sub x5, x5, #1

    mov x2, #1

    lsl x2, x2, ${shift}

    // If x1 is not a multiple of x2 skip alignment
    sub x10, x2, #1
    ands x10, x1, x10
    bne ${L("contiguous_loop2_cond")}
    b ${L("contiguous_loop2_align_cond")}

    ${L("contiguous_loop2_align")}:
    %if routine == "scal_real_cplx":
      ldp ${prec}2, ${prec}3, [x1]
      fmul ${prec}2, ${prec}2, ${prec}0
      fmul ${prec}3, ${prec}3, ${prec}0
      stp ${prec}2, ${prec}3, [x3]
    %else:
      ldr ${prec}2, [x1]
      %if routine == "scal" or routine == "scal_out_of_place":
        fmul ${prec}2, ${prec}2, ${prec}0
      %endif

      str ${prec}2, [x3]

    %endif

    add x1, x1, x2
    add x3, x3, x2

    ${L("contiguous_loop2_align_cond")}:
        ands x10, x1, x5
        ccmp x9, x1, #4, ne
        bne ${L("contiguous_loop2_align")}
        b ${L("contiguous_loop2_cond")}
%else:
    b ${L("contiguous_loop2_cond")}
%endif

${L("contiguous_loop2")}:
    ld1${mem} z2.${prec_load}, p0/z, [x1]
    ld1${mem} z3.${prec_load}, p0/z, [x1, #1, mul vl]
    add x1, x1, x6

    %if is_axp or is_vecadd:
      ld1${mem} z4.${prec}, p0/z, [x3]
      ld1${mem} z5.${prec}, p0/z, [x3, #1, mul vl]
      %if is_axpb:
        fmul z4.${prec}, z4.${prec}, z1.${prec}
        fmul z5.${prec}, z5.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z4.${prec}, p0/m, z4.${prec}, z2.${prec}
        fadd z5.${prec}, p0/m, z5.${prec}, z3.${prec}
      %else:
        fmla z4.${prec}, p0/m, z2.${prec}, z0.${prec}
        fmla z5.${prec}, p0/m, z3.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z4.${prec}, z2.${prec}, z0.${prec}
      fmul z5.${prec}, z3.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3]
      st1${mem} z3.${prec}, p0, [x3, #1, mul vl]
    %elif routine == "waxpby":
      st1${mem} z4.${prec}, p0, [x13]
      st1${mem} z5.${prec}, p0, [x13, #1, mul vl]
      add x13, x13, x6
    %else:
      st1${mem} z4.${prec_load}, p0, [x3]
      st1${mem} z5.${prec_load}, p0, [x3, #1, mul vl]
    %endif
    add x3, x3, x6

${L("contiguous_loop2_cond")}:
    cmp x1, x8
    // End of the loop, check for exits
    ble ${L("contiguous_loop2")}


// unroll by 1 vector, predicate
    cntb x6                // vec len in bytes

    b ${L("contiguous_loop1_cond")}

${L("contiguous_loop1")}:
    ld1${mem} z2.${prec_load}, p0/z, [x1]
    add x1, x1, x6

    %if is_axp or is_vecadd:
      ld1${mem} z4.${prec}, p0/z, [x3]
      %if is_axpb:
        fmul z4.${prec}, z4.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z4.${prec}, p0/m, z4.${prec}, z2.${prec}
      %else:
        fmla z4.${prec}, p0/m, z2.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z4.${prec}, z2.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3]
    %elif routine == "waxpby":
      st1${mem} z4.${prec}, p0, [x13]
      add x13, x13, x6
    %else:
      st1${mem} z4.${prec_load}, p0, [x3]
    %endif
    add x3, x3, x6

${L("contiguous_loop1_cond")}:
    whilelt p0.b, x1, x9
    // End of the loop, check for exits
    b.first ${L("contiguous_loop1")}

    ldp x29, x30, [sp], #16
    ret

${L("use_mixed_strides_incy_loop_check")}:
    cmp x4, #1
    bne ${L("use_noncontiguous_loop")}
    // Otherwise fall into mixed_stride where incx strided, incy isn't

    /***************************************/
    /* MIXED STRIDED LOOP INCX !=1, INCY=1 */
    /***************************************/

${L("use_mixed_strides_incx_loop")}:

    // The address offsets for the first vector
    %if routine == "scal_real_cplx" and prec == "d":
      // Two elts per complex number --> 2*n elts
      // We don't do this for single complex
      // as we treat it as loading doubles, so load 2 elements at once
      // next to each other anyway
      lsl x0, x0, #1
      // increase stride accordingly for indices
      lsl ${x}6, x2, #1
      index z4.${prec}, #0, ${x}6
      // imaginary parts are adjacent to the real parts and we need them together
      index z5.${prec}, #1, ${x}6
      zip1 z4.${prec}, z4.${prec}, z5.${prec}
    %else:
      index z4.${prec_load}, #0, ${x}2
    %endif

    // Multiply incx by the vector size
    cnt${mem} x5
    mul x10, x2, x5

    // The address offsets for the second vector
    %if routine == "scal_real_cplx" and prec == "d":
      index z6.${prec}, ${x}10, ${x}6
      add ${x}10, ${x}10, #1
      index z7.${prec}, ${x}10, ${x}6
      zip1 z6.${prec}, z6.${prec}, z7.${prec}
    %else:
      index z6.${prec_load}, ${x}10, ${x}2
    %endif

    ptrue p0.${prec_load}       // ignore predication

    // unroll by 2 vectors
    // After this x5 is the loop counter, x7 is the number of elements
    // copied in the unrolled loop. We use a loop counter here because the
    // non-contiguous loop handles negative increments, so we can't
    // rely on the ble test in .Loop_cond as we do in the contig loop above.

    lsl x7, x5, #1
    sub x5, x0, x7

    cntb x6, all, mul #2   // 2x vec len in bytes

    mul x10, x2, x6

    b ${L("mixed_strides_incx_loop2_cond")}

${L("mixed_strides_incx_loop2")}:
    // Copy the x element
    ld1${mem} z2.${prec_load}, p0/z, [x1, z4.${prec_load}, ${ncshift}]
    ld1${mem} z3.${prec_load}, p0/z, [x1, z6.${prec_load}, ${ncshift}]
    add x1, x1, x10

    %if is_axp or is_vecadd:
      ld1${mem} z16.${prec}, p0/z, [x3]
      ld1${mem} z17.${prec}, p0/z, [x3, #1, mul vl]
      %if is_axpb:
        fmul z16.${prec}, z16.${prec}, z1.${prec}
        fmul z17.${prec}, z17.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
          fadd z16.${prec}, p0/m, z16.${prec}, z2.${prec}
          fadd z17.${prec}, p0/m, z17.${prec}, z3.${prec}
      %else:
          fmla z16.${prec}, p0/m, z2.${prec}, z0.${prec}
          fmla z17.${prec}, p0/m, z3.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z16.${prec}, z2.${prec}, z0.${prec}
      fmul z17.${prec}, z3.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3]
      st1${mem} z3.${prec}, p0, [x3, #1, mul vl]
    %elif routine == "waxpby":
      st1${mem} z16.${prec}, p0, [x13]
      st1${mem} z17.${prec}, p0, [x13, #1, mul vl]
      add x13, x13, x6
    %else:
      st1${mem} z16.${prec_load}, p0, [x3]
      st1${mem} z17.${prec_load}, p0, [x3, #1, mul vl]
    %endif
    sub x5, x5, x7
    add x3, x3, x6

${L("mixed_strides_incx_loop2_cond")}:
    cmp x5, #0
    bge ${L("mixed_strides_incx_loop2")}

// unroll by 1 vector, predicate

    // undo the last subtract from above
    add x5, x5, x7

    cnt${mem} x7 // x7 is the number of elements processed in each loop iter

    cntb x6                // vec len in bytes

    mul x10, x2, x6

    b ${L("mixed_strides_incx_loop1_cond")}

${L("mixed_strides_incx_loop1")}:
    // Copy the x element
    ld1${mem} z2.${prec_load}, p0/z, [x1, z4.${prec_load}, ${ncshift}]
    add x1, x1, x10

    %if is_axp or is_vecadd:
      ld1${mem} z16.${prec}, p0/z, [x3]
      %if is_axpb:
        fmul z16.${prec}, z16.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z16.${prec}, p0/m, z16.${prec}, z2.${prec}
      %else:
        fmla z16.${prec}, p0/m, z2.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z16.${prec}, z2.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3]
    %elif routine == "waxpby":
      st1${mem} z16.${prec}, p0, [x13]
      add x13, x13, x6
    %else:
      st1${mem} z16.${prec_load}, p0, [x3]
    %endif
    sub x5, x5, x7
    add x3, x3, x6

${L("mixed_strides_incx_loop1_cond")}:
    whilelt p0.${prec_load}, xzr, x5
    b.first ${L("mixed_strides_incx_loop1")}

    ldp x29, x30, [sp], #16
    ret


    /*****************************************/
    /* MIXED STRIDED LOOP INCX =1, INCY != 1 */
    /*****************************************/
${L("use_mixed_strides_incy_loop")}:

    // The address offsets for the first vector
    %if routine == "scal_real_cplx" and prec == "d":
      // Two elts per complex number --> 2*n elts
      // We don't do this for single complex
      // as we treat it as loading doubles, so load 2 elements at once
      // next to each other anyway
      lsl x0, x0, #1
      // increase stride accordingly for indices
      lsl ${x}6, x4, #1
      index z4.${prec}, #0, ${x}6
      // imaginary parts are adjacent to the real parts and we need them together
      index z5.${prec}, #1, ${x}6
      zip1 z4.${prec}, z4.${prec}, z5.${prec}
    %else:
      index z4.${prec_load}, #0, ${x}4
    %endif

    // Multiply incy by the vector size
    cnt${mem} x5
    mul x10, x4, x5

    // The address offsets for the second vector
    %if routine == "scal_real_cplx" and prec == "d":
      index z6.${prec}, ${x}10, ${x}6
      add ${x}10, ${x}10, #1
      index z7.${prec}, ${x}10, ${x}6
      zip1 z6.${prec}, z6.${prec}, z7.${prec}
    %else:
      index z6.${prec_load}, ${x}10, ${x}4
    %endif

    ptrue p0.${prec_load}       // ignore predication

    // unroll by 2 vectors
    // After this x5 is the loop counter, x7 is the number of elements
    // copied in the unrolled loop. We use a loop counter here because the
    // non-contiguous loop handles negative increments, so we can't
    // rely on the ble test in .Loop_cond as we do in the contig loop above.

    lsl x7, x5, #1
    sub x5, x0, x7

    cntb x6, all, mul #2   // 2x vec len in bytes

    mul x10, x4, x6

    b ${L("mixed_strides_incy_loop2_cond")}

${L("mixed_strides_incy_loop2")}:
    // Copy the element
    ld1${mem} z2.${prec_load}, p0/z, [x1]
    ld1${mem} z3.${prec_load}, p0/z, [x1, #1, mul vl]
    add x1, x1, x6

    %if is_axp or is_vecadd:
      ld1${mem} z16.${prec}, p0/z, [x3, z4.${prec}, ${ncshift}]
      ld1${mem} z17.${prec}, p0/z, [x3, z6.${prec}, ${ncshift}]
      %if is_axpb:
        fmul z16.${prec}, z16.${prec}, z1.${prec}
        fmul z17.${prec}, z17.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z16.${prec}, z16.${prec}, z2.${prec}
        fadd z17.${prec}, z17.${prec}, z3.${prec}
      %else:
        fmla z16.${prec}, p0/m, z2.${prec}, z0.${prec}
        fmla z17.${prec}, p0/m, z3.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z16.${prec}, z2.${prec}, z0.${prec}
      fmul z17.${prec}, z3.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3, z4.${prec}, ${ncshift}]
      st1${mem} z3.${prec}, p0, [x3, z6.${prec}, ${ncshift}]
    %elif routine == "waxpby":
      st1${mem} z16.${prec}, p0, [x13]
      st1${mem} z17.${prec}, p0, [x13, #1, mul vl]
      add x13, x13, x6
    %else:
      st1${mem} z16.${prec_load}, p0, [x3, z4.${prec_load}, ${ncshift}]
      st1${mem} z17.${prec_load}, p0, [x3, z6.${prec_load}, ${ncshift}]
    %endif
    sub x5, x5, x7
    add x3, x3, x10

${L("mixed_strides_incy_loop2_cond")}:
    cmp x5, #0
    bge ${L("mixed_strides_incy_loop2")}

// unroll by 1 vector, predicate

    // undo the last subtract from above
    add x5, x5, x7

    cnt${mem} x7 // x7 is the number of elements processed in each loop iter

    cntb x6                // vec len in bytes

    mul x10, x4, x6

    b ${L("mixed_strides_incy_loop1_cond")}

${L("mixed_strides_incy_loop1")}:
    // Copy the element
    ld1${mem} z2.${prec_load}, p0/z, [x1]
    add x1, x1, x6

    %if is_axp or is_vecadd:
      ld1${mem} z16.${prec}, p0/z, [x3, z4.${prec}, ${ncshift}]
      %if is_axpb:
        fmul z16.${prec}, z16.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z16.${prec}, z16.${prec}, z2.${prec}
      %else:
        fmla z16.${prec}, p0/m, z2.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z16.${prec}, z2.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3, z4.${prec}, ${ncshift}]
    %elif routine == "waxpby":
      st1${mem} z16.${prec}, p0, [x13]
      add x13, x13, x6
    %else:
      st1${mem} z16.${prec_load}, p0, [x3, z4.${prec_load}, ${ncshift}]
    %endif
    sub x5, x5, x7
    add x3, x3, x10

${L("mixed_strides_incy_loop1_cond")}:
    whilelt p0.${prec_load}, xzr, x5
    b.first ${L("mixed_strides_incy_loop1")}

    ldp x29, x30, [sp], #16
    ret

    /***********************/
    /* NON-CONTIGUOUS LOOP */
    /***********************/
${L("use_noncontiguous_loop")}:

    // The address offsets for the first vector
    %if routine == "scal_real_cplx" and prec == "d":
      // Two numbers in complex numbers so going to go through 2n total elements
      // We don't do this for cs as we treat it as loading doubles, so load 2 elements at once
      // next to each other anyway
      lsl x0, x0, #1
      // increase stride accordingly for indices
      lsl ${x}6, x2, #1
      index z4.${prec}, #0, ${x}6
      // imaginary parts are adjacent to the real parts and we need them together
      index z5.${prec}, #1, ${x}6
      zip1 z4.${prec}, z4.${prec}, z5.${prec}
      mov z5.${prec}, z4.${prec}
    %else:
      index z4.${prec_load}, #0, ${x}2
      index z5.${prec_load}, #0, ${x}4
    %endif
    %if routine == "waxpby":
      index z18.${prec_load}, #0, ${x}14
    %endif

    // Multiply increments by the vector size
    cnt${mem} x5
    mul x10, x2, x5
    mul x11, x4, x5

    %if routine == "waxpby":
      mul x15, x14, x5
    %endif

    // The address offsets for the second vector
    %if routine == "scal_real_cplx" and prec == "d":
      index z6.${prec}, ${x}10, ${x}6
      add ${x}10, ${x}10, #1
      index z7.${prec}, ${x}10, ${x}6
      zip1 z6.${prec}, z6.${prec}, z7.${prec}
      mov z7.${prec}, z6.${prec}
    %else:
      index z6.${prec_load}, ${x}10, ${x}2
      index z7.${prec_load}, ${x}11, ${x}4
    %endif
    %if routine == "waxpby":
      index z19.${prec_load}, ${x}15, ${x}14
    %endif

    ptrue p0.${prec_load}       // ignore predication

    // unroll by 2 vectors
    // After this x5 is the loop counter, x7 is the number of elements
    // copied in the unrolled loop. We use a loop counter here because the
    // non-contiguous loop handles negative increments, so we can't
    // rely on the ble test in .Loop_cond as we do in the contig loop above.

    lsl x7, x5, #1
    sub x5, x0, x7

    cntb x6, all, mul #2   // 2x vec len in bytes

    mul x10, x2, x6
    mul x11, x4, x6

    %if routine == "waxpby":
      mul x15, x14, x6
    %endif

    b ${L("noncontig_loop2_cond")}

${L("noncontig_loop2")}:
    // Copy the element
    ld1${mem} z2.${prec_load}, p0/z, [x1, z4.${prec_load}, ${ncshift}]
    ld1${mem} z3.${prec_load}, p0/z, [x1, z6.${prec_load}, ${ncshift}]
    add x1, x1, x10

    %if is_axp or is_vecadd:
      ld1${mem} z16.${prec}, p0/z, [x3, z5.${prec}, ${ncshift}]
      ld1${mem} z17.${prec}, p0/z, [x3, z7.${prec}, ${ncshift}]
      %if is_axpb:
        fmul z16.${prec}, z16.${prec}, z1.${prec}
        fmul z17.${prec}, z17.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z16.${prec}, z16.${prec}, z2.${prec}
        fadd z17.${prec}, z17.${prec}, z3.${prec}
      %else:
        fmla z16.${prec}, p0/m, z2.${prec}, z0.${prec}
        fmla z17.${prec}, p0/m, z3.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z16.${prec}, z2.${prec}, z0.${prec}
      fmul z17.${prec}, z3.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3, z5.${prec}, ${ncshift}]
      st1${mem} z3.${prec}, p0, [x3, z7.${prec}, ${ncshift}]
    %elif routine == "waxpby":
      st1${mem} z16.${prec}, p0, [x13, z18.${prec}, ${ncshift}]
      st1${mem} z17.${prec}, p0, [x13, z19.${prec}, ${ncshift}]
      add x13, x13, x15
    %else:
      st1${mem} z16.${prec_load}, p0, [x3, z5.${prec_load}, ${ncshift}]
      st1${mem} z17.${prec_load}, p0, [x3, z7.${prec_load}, ${ncshift}]
    %endif
    sub x5, x5, x7
    add x3, x3, x11

${L("noncontig_loop2_cond")}:
    cmp x5, #0
    bge ${L("noncontig_loop2")}

// unroll by 1 vector, predicate

    // undo the last subtract from above
    add x5, x5, x7

    cnt${mem} x7 // x7 is the number of elements processed in each loop iter

    cntb x6                // vec len in bytes

    mul x10, x2, x6
    mul x11, x4, x6
    %if routine == "waxpby":
      mul x15, x14, x6
    %endif

    b ${L("noncontig_loop1_cond")}

${L("noncontig_loop1")}:
    // Copy the element
    ld1${mem} z2.${prec_load}, p0/z, [x1, z4.${prec_load}, ${ncshift}]
    add x1, x1, x10

    %if is_axp or is_vecadd:
      ld1${mem} z16.${prec}, p0/z, [x3, z5.${prec}, ${ncshift}]
      %if is_axpb:
        fmul z16.${prec}, z16.${prec}, z1.${prec}
      %endif
      %if is_vecadd:
        fadd z16.${prec}, z16.${prec}, z2.${prec}
      %else:
        fmla z16.${prec}, p0/m, z2.${prec}, z0.${prec}
      %endif
    %elif is_scal:
      fmul z16.${prec}, z2.${prec}, z0.${prec}
    %endif

    %if routine == "copy":
      st1${mem} z2.${prec}, p0, [x3, z5.${prec}, ${ncshift}]
    %elif routine == "waxpby":
      st1${mem} z16.${prec}, p0, [x13, z18.${prec}, ${ncshift}]
      add x13, x13, x15
    %else:
      st1${mem} z16.${prec_load}, p0, [x3, z5.${prec_load}, ${ncshift}]
    %endif
    sub x5, x5, x7
    add x3, x3, x11

${L("noncontig_loop1_cond")}:
    whilelt p0.${prec_load}, xzr, x5
    b.first ${L("noncontig_loop1")}

    ldp x29, x30, [sp], #16
    ret
    ${epilogue(func_name)}

  %endfor

%endfor
