## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<% import math %>
// Kernels that are "copy-like" and complex

## Routine name to operation mapping:
## copy : y = x
## scal : x = a * x
## scal_out_of_place : y = a * x
## scal_real_cplx : x = a * x, where a is real
## axpy : y = a * x + y
## axpby : y = a * x + b * y
## waxpby : w = a * x + b * y
## vecacc : y = x + y

// Input Parameter:
#define N x0
#define DX x1
#define INCX x2
#define DY x3
#define INCY x4
#define DW x5
#define INCW x6
#define TMP x7
#define INC2 x8

## swap consecutive elements
##
##    swap_pairs([ a | b | c | d ]) := [ b | a | d | c ]
##
##    swap_pairs([ a | b ]) := [ b | a ]
##
<%def name="swap_pairs(out, inp)">
	%if kernel[0] == "c":
		rev64 ${out}.4s, ${inp}.4s
	%else:
		ext ${out}.16b, ${inp}.16b, ${inp}.16b, 8
	%endif
</%def>

<%def name="contig_kernel_two_vectors()">
	<%
		vext = ".{}{}".format(vector_elements, prec)
		alpha1 = "v4" + vext
		alpha2 = "v5" + vext
		beta1 =  "v6" + vext
		beta2 =  "v7" + vext
	%>

	// v4 = [  a.re | a.re ]
	// v5 = [ -a.im | a.im ]
	// v6 = [  b.re | b.re ]
	// v7 = [ -b.im | b.im ]
	// v20, v21 = [ x0.re | x0.im ]
	ldp q20, q21, [DX], ${vector_bytes * 2}

	// v22, v23 = [ x0.im | x0.re ]
	${swap_pairs("v22", "v20")}
	${swap_pairs("v23", "v21")}

	%if is_scal or is_scal_out_of_place:
		// v16, v17 = [ x0.re * a.re | x0.im * a.re ]
		fmul v16${vext}, v20${vext}, ${alpha1}
		fmul v17${vext}, v21${vext}, ${alpha1}
	%else:
		// v16, v17 = [ y0.re | y0.im ]
		ldp q16, q17, [DY]
		%if is_axpby or is_waxpby:
			// v18, v19 = [ y0.im | y0.re ]
			${swap_pairs("v18", "v16")}
			${swap_pairs("v19", "v17")}

			// v16, v17 = [ y0.re * b.re | y0.im * b.re ]
			fmul v16${vext}, v16${vext}, ${beta1}
			fmul v17${vext}, v17${vext}, ${beta1}

			// v16, v17 = b * y := [ y0.re * b.re - y0.im * b.im | y0.im * b.re + y0.re * b.im ]
			fmla v16${vext}, v18${vext}, ${beta2}
			fmla v17${vext}, v19${vext}, ${beta2}
		%endif
		// v16, v17 = b * y + [ x0.re * a.re | x0.im * a.re ]
		// axpy => b = 1
		%if is_vecadd:
			fadd v16${vext}, v16${vext}, v20${vext}
			fadd v17${vext}, v17${vext}, v21${vext}
		%else:
			fmla v16${vext}, v20${vext}, ${alpha1}
			fmla v17${vext}, v21${vext}, ${alpha1}
		%endif
	%endif

	// v16, v17 = b * y + a * x
	// a * x := [ x0.re * a.re - x0.im * a.im | x0.im * a.re + x0.re * a.im ]
	// axpy => b = 1
	// scal => b = 0
	%if not (is_vecadd):
		fmla v16${vext}, v22${vext}, ${alpha2}
		fmla v17${vext}, v23${vext}, ${alpha2}
	%endif

	%if is_waxpby:
		stp q16, q17, [DW], ${vector_bytes * 2}
		add DY, DY, ${vector_bytes * 2}
	%else:
		stp q16, q17, [DY], ${vector_bytes * 2}
	%endif
</%def>
// give these args as numbers you want regs to take e.g.
// load_strided_into_vector(1,2) loads into v1 using regs v1
<%def name="load_strided_into_vector(r0,r1,strided_D,INC)">
	<% vext = ".2" + prec %>
	%if kernel[0] == "c":
		%if strided_D == "DY":
			mov TMP, ${strided_D}
			ld1 {${r0 + vext}}, [TMP], ${INC}
			ld1 {${r0 + ".d"}}[1], [TMP]
		%else:
			ld1 {${r0 + vext}}, [${strided_D}], ${INC}
			ld1 {${r0 + ".d"}}[1], [${strided_D}], ${INC}
		%endif
	%else:
		%if strided_D == "DY":
			ld1 {${r0 + vext}}, [${strided_D}]
		%else:
			ld1 {${r0 + vext}}, [${strided_D}], ${INC}
		%endif
	%endif
</%def>

<%def name="mixed_stride_kernel_one_vector(is_xstride,y_reg,x_reg)">
	<%
		str_vext = ".2{}".format(prec)
		vext = ".{}{}".format(vector_elements, prec)
		alpha1 = "v4" + vext
		alpha2 = "v5" + vext
		beta1 =  "v6" + vext
		beta2 =  "v7" + vext

		x_vector = "v" + str(x_reg)
		x_swap = "v" + str(x_reg + 2)

		y_vector = "v" + str(y_reg)
		y_swap = "v" + str(y_reg + 2)
		y_q = "q" + str(y_reg)
	%>

	%if not is_scal_out_of_place or is_xstride:
		%if is_xstride:
			${load_strided_into_vector(x_vector,x_swap,"DX","INCX")}
			${swap_pairs(x_swap,x_vector)}
		%else:
			${load_strided_into_vector(y_vector,y_swap,"DY","INCY")}
			${swap_pairs(y_swap,y_vector)}
		%endif
	%endif

	%if is_scal_out_of_place:
		fmul ${y_vector + vext}, ${x_vector + vext}, ${alpha1}
	%else:
		%if is_axpby or is_waxpby:
			fmul ${y_vector + vext}, ${y_vector + vext}, ${beta1}
			fmla ${y_vector + vext}, ${y_swap + vext}, ${beta2}
		%endif
		%if is_vecadd:
			fadd ${y_vector + vext}, ${y_vector + vext}, ${x_vector + vext}
		%else:
			fmla ${y_vector + vext}, ${x_vector + vext}, ${alpha1}
		%endif
	%endif
	%if not (is_vecadd):
		fmla ${y_vector + vext}, ${x_swap + vext}, ${alpha2}
	%endif

	%if is_waxpby:
		str ${y_q}, [DW], ${vector_bytes}
		%if is_xstride:
			add DY, DY, ${vector_bytes}
		%else:
			%if kernel[0] == "c":
				add DY, DY, INC2

			%else:
				add DY, DY, INCY
			%endif
		%endif
	%else:
		%if is_xstride:
			str ${y_q}, [DY], ${vector_bytes}
		%else:
			st1 {${y_vector + str_vext}}, [DY], INCY
			%if kernel[0] == "c":
				st1 {${y_vector + ".d"}}[1], [DY], INCY
			%endif
		%endif
	%endif
</%def>

<%def name="mixed_stride_kernel(strided_D)">
	<%
		is_xstride = strided_D == "DX"
		is_ystride = strided_D == "DY"
	%>
	// need this check as scal_out_of_place doesn't want to load DY at all
	%if not is_scal_out_of_place or is_ystride:
		%if is_ystride:
			ldp q20, q21, [DX], ${vector_bytes * 2}
			${swap_pairs("v22", "v20")}
			${swap_pairs("v23", "v21")}
		%else:
			ldp q16, q17, [DY]
			${swap_pairs("v18", "v16")}
			${swap_pairs("v19", "v17")}
		%endif
	%endif

	${mixed_stride_kernel_one_vector(is_xstride,16,20)}
	${mixed_stride_kernel_one_vector(is_xstride,17,21)}
</%def>

<%def name="scalar_kernel(is_xstride,is_ystride)">
<%
	vext = ".2" + prec
	alpha1 = "v4" + vext
	alpha2 = "v5" + vext
	beta1 =  "v6" + vext
	beta2 =  "v7" + vext
%>
	## The strided kernel uses the same algorithm as the contiguous kernel.
	%if is_xstride:
		ld1 {v20${vext}}, [DX]
		add DX, DX, INCX
	%else:
		ld1 {v20${vext}}, [DX], ${element_bytes * 2}
	%endif
	${swap_pairs("v22", "v20")}
	%if is_scal or is_scal_out_of_place:
		fmul v16${vext}, v20${vext}, ${alpha1}
	%else:
		ld1 {v16${vext}}, [DY]
		%if is_axpby or is_waxpby:
			${swap_pairs("v18", "v16")}
			fmul v16${vext}, v16${vext}, ${beta1}
			fmla v16${vext}, v18${vext}, ${beta2}
		%endif
		%if is_vecadd:
			fadd v16${vext}, v16${vext}, v20${vext}
		%else:
			fmla v16${vext}, v20${vext}, ${alpha1}
		%endif
	%endif

	%if not (is_vecadd):
		fmla v16${vext}, v22${vext}, ${alpha2}
	%endif

	%if is_waxpby:
		st1 {v16${vext}}, [DW]
		add DW, DW, INCW
		add DY, DY, INCY
	%else:
		%if is_ystride:
			st1 {v16${vext}}, [DY]
			add DY, DY, INCY
		%else:
			st1 {v16${vext}}, [DY], ${element_bytes * 2}
		%endif
	%endif
</%def>

<%def name="setup_inputs()">
	%if is_scal:
		// The output is the input
		mov DY, DX
		mov INCY, INCX
	%elif is_axpy or is_axpby or is_scal_out_of_place:
		// y and incx are swapped in the axp interface
		mov TMP, INCX
		mov INCX, DY
		mov DY, TMP
	%elif is_waxpby:
		// We want to transpose the input parameters
		// x, y, z, incx, incy, incz -> x, incx, y, incy, z, incz
		// y: x2 -> x3 ; x13 is w
		mov TMP, x3
		mov x3, x2
		// incx: x4 -> x2
		mov x2, x4
		// incy: x5 -> x4
		mov x4, x5
		mov x5, TMP
	%endif
</%def>

<%def name="shift_inputs()">
	// shift register numbers up since we already move alpha and beta to FP registers.
	%if kernel[0] == "z":
		%if is_scal or is_axpy:
			mov x1, x3
			mov x2, x4
		%else:
			mov x1, x3
			mov x2, x6
		%endif
		%if is_axpy or is_axpby or is_scal_out_of_place or is_waxpby or is_vecadd:
			%if is_axpy:
				// axpy has no beta
				mov x3, x5
				mov x4, x6
			%else:
				mov x3, x7
				ldr x4, [sp, #16]
			%endif
			%if is_waxpby:
				ldr x5, [sp, #24]
				ldr x6, [sp, #32]
			%endif
		%endif
	%else:
		%if is_scal or is_axpy or is_vecadd:
			mov x1, x2
			mov x2, x3
		%else:
			mov x1, x2
			mov x2, x4
		%endif
		%if is_axpy or is_axpby or is_scal_out_of_place or is_waxpby or is_vecadd:
			%if is_axpy or is_vecadd:
				// axpy has no beta
				mov x3, x4
				mov x4, x5
			%else:
				mov x3, x5
				mov x4, x6
			%endif
			%if is_waxpby:
				mov x5, x7
				ldr x6, [sp, #16]
			%endif
		%endif
	%endif
</%def>

<%def name="move_alpha()">
	%if kernel[0] == "z":
		fmov d0, x1
		fmov d1, x2
	%else:
		lsr x8, x1, #0x20
		fmov s0, w1
		fmov s1, w8
	%endif
</%def>

<%def name="move_beta()">
	%if kernel[0] == "z":
		fmov d2, x4
		fmov d3, x5
	%else:
		lsr x8, x3, #0x20
		fmov s2, w3
		fmov s3, w8
	%endif
</%def>

<%def name="setup_alpha(conj_x=False)">
	%if kernel[0] == "z":
		%if conj_x:
			// setup vectors in following scheme
			// v4.2d = [  a.re | -a.re ]
			// v5.2d = [  a.im | a.im  ]
			dup v5.2d, v1.d[0]
			fneg d4, d0
			ins v4.d[1], v4.d[0]
			ins v4.d[0], v0.d[0]
		%else:
			// setup vectors in following scheme
			// v4.2d = [  a.re | a.re ]
			// v5.2d = [ -a.im | a.im ]
			dup v4.2d, v0.d[0]
			fneg d5, d1
			mov v5.d[1], v1.d[0]
		%endif
	%else:
		%if conj_x:
			// setup vectors in following scheme
			// v4.4s = [ a.re | -a.re | a.re | -a.re ]
			// v5.4s = [ a.im |  a.im | a.im |  a.im ]
			dup v4.4s, v0.s[0]
			dup v5.4s, v1.s[0]
			fneg v4.2d, v4.2d
		%else:
			// setup vectors in following scheme
			// v4.4s = [  a.re | a.re |  a.re | a.re ]
			// v5.4s = [ -a.im | a.im | -a.im | a.im ]
			dup v4.4s, v0.s[0]
			fneg s5, s1
			mov v5.s[1], v1.s[0]
			mov v5.s[2], v5.s[0]
			mov v5.s[3], v5.s[1]
		%endif
	%endif
</%def>

<%def name="setup_beta()">
	%if kernel[0] == "z":
		// setup vectors in following scheme
		// v6.2d = [  b.re | b.re ]
		// v7.2d = [ -b.im | b.im ]
		dup v6.2d, v2.d[0]
		fneg d7, d3
		mov v7.d[1], v3.d[0]
	%else:
		// setup vectors in following scheme
		// v6.2d = [  b.re | b.re |  b.re | b.re ]
		// v7.2d = [ -b.im | b.im | -b.im | b.im ]
		dup v6.4s, v2.s[0]
		fneg s7, s3
		mov v7.s[1], v3.s[0]
		mov v7.s[2], v7.s[0]
		mov v7.s[3], v7.s[1]
	%endif
</%def>

%for routine in [ "scal","scal_out_of_place","axpy", "axpy_no_prefetch", "axpby", "waxpby", "vecadd"]:
	%for prec, cprec in [("s", "c"), ("d", "z")]:
		<%
		prec = prec
		kernel = cprec + routine + "_kernel"
		is_scal = "scal" == routine
		is_axpy = "axpy" == routine
		is_axpby = "axpby" == routine
		is_scal_out_of_place = "scal_out_of_place" == routine
		is_waxpby = "waxpby" == routine
		is_vecadd = "vecadd" == routine

		# Convenience variables for the number of bytes/elements in a vector
		element_bytes = 4 if prec == 's' else 8
		vector_bytes = 16
		vector_elements = vector_bytes // element_bytes

		# Optimized for G2.
		contig_unroll_factor = 1
		strided_unroll_factor = 2
		%>

		${prologue(kernel)}
		stp x29, x30, [sp, #-16]!
		mov x29, sp

		%if target_os in ('windows', 'windows_arm64ec'):
			${move_alpha()}
			%if is_axpby or is_waxpby:
				${move_beta()}
			%endif
			${shift_inputs()}
		%endif
		${setup_inputs()}
		${setup_alpha()}
.L${kernel}_alpha_ready:
		%if is_axpby or is_waxpby:
			${setup_beta()}
		%endif

.L${kernel}_contiguous_check:
		// First check if this is 'contiguous' (i.e. incx and incy are 1)
		%if is_waxpby:
			cmp INCW, #1
			bne .L${kernel}_use_noncontiguous_loop
		%endif

		cmp INCX, #1
		bne .L${kernel}_maybe_mixed_stride_check_incy

		cmp INCY, #1
		bne .L${kernel}_use_mixed_stride_incy_loop
		lsl INCX, INCX, ${int(math.log(element_bytes*2,2))}
		lsl INCY, INCY, ${int(math.log(element_bytes*2,2))}
		%if is_waxpby:
			lsl INCW, INCW, ${int(math.log(element_bytes*2,2))}
		%endif

		// align x - works only on contiguous data
		%if is_scal:
			// Load x11 with the variable to compare against for pointer alignment
			mov x11, #15
			// Load x2 with the number of bytes to shift by
			mov x2, #1
			%if prec == "c":
				lsl x2, x2, #2
			%else:
				lsl x2, x2, #3
			%endif

			// If DX is not a multiple of x2 skip alignment (no matter how many elements we load, it will never be aligned)
			sub TMP, x2, #1
			ands TMP, DX, TMP
			bne .L${kernel}_contiguous_loop2_check
			b .L${kernel}_contiguous_loop_align_cond

			.L${kernel}_contiguous_loop_align:
				// if it is not already aligned, load and process single elements until it is
				subs N, N, #1
				// Call scalar kernel kernel to process one element
				${scalar_kernel(False,False)}

				beq .L${kernel}_contiguous_loop2_check

			.L${kernel}_contiguous_loop_align_cond:
				// Check if its already aligned
				ands TMP, DX, x11
				bne .L${kernel}_contiguous_loop_align
		%endif

.L${kernel}_contiguous_loop2_check:
		<%
		unroll_factor = contig_unroll_factor
		assert unroll_factor & (unroll_factor - 1) == 0, "unroll factor must be a power of 2"
		%>
		// Check if n is large enough to use the 2 vector kernel
		cmp N, ${vector_elements * unroll_factor}
		// If not, branch to the scalar kernel check
		blt .L${kernel}_contiguous_loop1_check

.L${kernel}_contiguous_loop2:

		%for _ in range(unroll_factor):
			${contig_kernel_two_vectors()}
		%endfor
		// Update the counter
		sub N, N, ${vector_elements * unroll_factor}
		b .L${kernel}_contiguous_loop2_check

.L${kernel}_contiguous_loop1_check:
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_contiguous_loop1:
		// Loop until N is 0
		sub N, N, 1

		${scalar_kernel(False,False)}

		b .L${kernel}_contiguous_loop1_check


.L${kernel}_maybe_mixed_stride_check_incy:
		cmp INCY, #1
		bne .L${kernel}_use_noncontiguous_loop
		// Otherwise fall into mixed_stride where inxc strided, incy isn't


.L${kernel}_use_mixed_stride_incx_loop:
		lsl INCX, INCX, ${int(math.log(element_bytes*2,2))}
		lsl INCY, INCY, ${int(math.log(element_bytes*2,2))}
		%if is_waxpby:
			lsl INCW, INCW, ${int(math.log(element_bytes*2,2))}
		%endif

.L${kernel}_mixed_stride_incx_loop2_check:
		// Check if n is large enough to use the 2 vector kernel
		<%
		unroll_factor = contig_unroll_factor
		assert unroll_factor & (unroll_factor - 1) == 0, "unroll factor must be a power of 2"
		%>
		cmp N, ${vector_elements * unroll_factor}
		// Otherwise, branch to the scalar one
		blt .L${kernel}_mixed_stride_incx_loop1_check

.L${kernel}_mixed_stride_incx_loop2:

		%for _ in range(unroll_factor):
			${mixed_stride_kernel("DX")}
		%endfor

		sub N, N, ${vector_elements * unroll_factor}
		b .L${kernel}_mixed_stride_incx_loop2_check

.L${kernel}_mixed_stride_incx_loop1_check:
		// Check if N is 0, exit if so, otherwise enter the scalar loop
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_mixed_stride_incx_loop1:

		// Loop until N is 0
		sub N, N, 1

		${scalar_kernel(True,False)}

		b .L${kernel}_mixed_stride_incx_loop1_check


.L${kernel}_use_mixed_stride_incy_loop:
		lsl INCX, INCX, ${int(math.log(element_bytes*2,2))}
		lsl INCY, INCY, ${int(math.log(element_bytes*2,2))}
		%if is_waxpby:
			lsl INCW, INCW, ${int(math.log(element_bytes*2,2))}
			lsl INC2, INCY, #1
		%endif

.L${kernel}_mixed_stride_incy_loop2_check:
		// Check if n is large enough to use the 2 vector kernel
		<%
		unroll_factor = contig_unroll_factor
		assert unroll_factor & (unroll_factor - 1) == 0, "unroll factor must be a power of 2"
		%>
		cmp N, ${vector_elements * unroll_factor}
		// Otherwise, branch to the scalar one
		blt .L${kernel}_mixed_stride_incy_loop1_check

.L${kernel}_mixed_stride_incy_loop2:

		%for _ in range(unroll_factor):
			${mixed_stride_kernel("DY")}
		%endfor

		sub N, N, ${vector_elements * unroll_factor}
		b .L${kernel}_mixed_stride_incy_loop2_check

.L${kernel}_mixed_stride_incy_loop1_check:
		// Exit if N is 0, otherwise enter the scalar loop
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_mixed_stride_incy_loop1:
		// Loop until N is a multiple of unroll_factor * vector widths
		sub N, N, 1

		${scalar_kernel(False,True)}

		b .L${kernel}_mixed_stride_incy_loop1_check


.L${kernel}_use_noncontiguous_loop:

		lsl INCX, INCX, ${int(math.log(element_bytes*2,2))}
		lsl INCY, INCY, ${int(math.log(element_bytes*2,2))}
		%if is_waxpby:
			lsl INCW, INCW, ${int(math.log(element_bytes*2,2))}
		%endif
		<% unroll_factor = strided_unroll_factor %>
		cmp N, ${unroll_factor}
		blt .L${kernel}_noncontiguous_loop_unrolled_end

.L${kernel}_noncontiguous_loop_unrolled:
		%for _ in range(unroll_factor):
			${scalar_kernel(True,True)}
		%endfor

		sub N, N, ${unroll_factor}
		cmp N, ${unroll_factor}
		bge .L${kernel}_noncontiguous_loop_unrolled

.L${kernel}_noncontiguous_loop_unrolled_end:
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_noncontiguous_loop_tail:
		${scalar_kernel(True,True)}

		sub N, N, 1
		cmp N, 0
		bne .L${kernel}_noncontiguous_loop_tail

.L${kernel}_end:
		ldp x29, x30, [sp], #16
		ret
		${epilogue(kernel)}
	%endfor
%endfor

// conj(x) kernels

## Generate conj(x) variants of copy-like kernels. To handle the conjugation of
## the input vector, we effectively push the conjugation into the alpha scalar
## and then jump to the appropriate non-conj kernel generated in the main loop
## above.
%for routine in [ "scal_out_of_place", "axpy", "axpby", "vecadd" ]:
	%for prec, cprec in [("s", "c"), ("d", "z")]:
		<%
		kernel = cprec + routine + "_conj_kernel"
		non_conj_kernel = cprec + routine + "_kernel"
		is_scal_out_of_place = "scal_out_of_place" == routine
		is_axpy = "axpy" == routine
		is_axpby = "axpby" == routine
		is_vecadd = "vecadd" == routine
		%>
		${prologue(kernel)}
		stp x29, x30, [sp, #-16]!
		mov x29, sp

		%if target_os in ('windows', 'windows_arm64ec'):
			${move_alpha()}
			%if is_axpby or is_waxpby:
				${move_beta()}
			%endif
			${shift_inputs()}
		%endif
		${setup_inputs()}
		${setup_alpha(conj_x=True)}
		// Branch to non-conjugate kernel
		b .L${non_conj_kernel}_alpha_ready
	%endfor
%endfor
