## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
<% import math %>
//Kernels that are "copy-like" and real

//Input Parameter:
#define N x0
#define DX x1
#define INCX x2
#define DY x3
#define INCY x4
#define DW x5
#define INCW x6
#define TMP x7
// Generic increments of either vector, *2, *3 and *4.
#define INC2 x8
#define INC3 x9
#define INC4 x10

<%def name="contig_kernel_four_vectors()">
	ldp ${tmp_q[5]}, ${tmp_q[6]}, [DX]
	ldp ${tmp_q[7]}, ${tmp_q[8]}, [DX, ${vector_bytes * 2}]
	add DX, DX, ${vector_bytes * 4}

	%if is_scal or is_scal_out_of_place or is_scal_real_cplx:
		fmul ${tmp_v[1]}, ${tmp_v[5]}, ${alpha_v}
		fmul ${tmp_v[2]}, ${tmp_v[6]}, ${alpha_v}
		fmul ${tmp_v[3]}, ${tmp_v[7]}, ${alpha_v}
		fmul ${tmp_v[4]}, ${tmp_v[8]}, ${alpha_v}
	%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
		ldp ${tmp_q[1]}, ${tmp_q[2]}, [DY]
		ldp ${tmp_q[3]}, ${tmp_q[4]}, [DY, ${vector_bytes * 2}]
		%if is_axpby or is_waxpby:
			fmul ${tmp_v[1]}, ${tmp_v[1]}, ${beta_v}
			fmul ${tmp_v[2]}, ${tmp_v[2]}, ${beta_v}
			fmul ${tmp_v[3]}, ${tmp_v[3]}, ${beta_v}
			fmul ${tmp_v[4]}, ${tmp_v[4]}, ${beta_v}
		%endif
		%if is_vecadd:
			fadd ${tmp_v[1]}, ${tmp_v[1]}, ${tmp_v[5]}
			fadd ${tmp_v[2]}, ${tmp_v[2]}, ${tmp_v[6]}
			fadd ${tmp_v[3]}, ${tmp_v[3]}, ${tmp_v[7]}
			fadd ${tmp_v[4]}, ${tmp_v[4]}, ${tmp_v[8]}
		%else:
			fmla ${tmp_v[1]}, ${tmp_v[5]}, ${alpha_v}
			fmla ${tmp_v[2]}, ${tmp_v[6]}, ${alpha_v}
			fmla ${tmp_v[3]}, ${tmp_v[7]}, ${alpha_v}
			fmla ${tmp_v[4]}, ${tmp_v[8]}, ${alpha_v}
		%endif
	%endif

	%if is_copy:
		stp ${tmp_q[5]}, ${tmp_q[6]}, [DY]
		stp ${tmp_q[7]}, ${tmp_q[8]}, [DY, ${vector_bytes * 2}]
	%elif is_waxpby:
		stp ${tmp_q[1]}, ${tmp_q[2]}, [DW]
		stp ${tmp_q[3]}, ${tmp_q[4]}, [DW, ${vector_bytes * 2}]
		add DW, DW, ${vector_bytes * 4}
	%else:
		stp ${tmp_q[1]}, ${tmp_q[2]}, [DY]
		stp ${tmp_q[3]}, ${tmp_q[4]}, [DY, ${vector_bytes * 2}]
	%endif
	add DY, DY, ${vector_bytes * 4}
</%def>

<%def name="contig_kernel_two_vectors()">
	ldp ${tmp_q[5]}, ${tmp_q[6]}, [DX], ${vector_bytes * 2}

	%if is_scal or is_scal_out_of_place or is_scal_real_cplx:
		fmul ${tmp_v[1]}, ${tmp_v[5]}, ${alpha_v}
		fmul ${tmp_v[2]}, ${tmp_v[6]}, ${alpha_v}
	%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
		ldp ${tmp_q[1]}, ${tmp_q[2]}, [DY]
		%if is_axpby or is_waxpby:
			fmul ${tmp_v[1]}, ${tmp_v[1]}, ${beta_v}
			fmul ${tmp_v[2]}, ${tmp_v[2]}, ${beta_v}
		%endif
		%if is_vecadd:
			fadd ${tmp_v[1]}, ${tmp_v[1]}, ${tmp_v[5]}
			fadd ${tmp_v[2]}, ${tmp_v[2]}, ${tmp_v[6]}
		%else:
			fmla ${tmp_v[1]}, ${tmp_v[5]}, ${alpha_v}
			fmla ${tmp_v[2]}, ${tmp_v[6]}, ${alpha_v}
		%endif
	%endif

	%if is_copy:
		stp ${tmp_q[5]}, ${tmp_q[6]}, [DY], ${vector_bytes * 2}
	%elif is_waxpby:
		stp ${tmp_q[1]}, ${tmp_q[2]}, [DW]
		add DW, DW, ${vector_bytes * 2}
		add DY, DY, ${vector_bytes * 2}
	%else:
		stp ${tmp_q[1]}, ${tmp_q[2]}, [DY], ${vector_bytes * 2}
	%endif
</%def>

<%def name="contig_kernel_one_vector()">
	ldr ${tmp_q[5]}, [DX], ${vector_bytes * 1}

	%if is_scal or is_scal_out_of_place or is_scal_real_cplx:
		fmul ${tmp_v[1]}, ${tmp_v[5]}, ${alpha_v}
	%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
		ldr ${tmp_q[1]}, [DY]
		%if is_axpby or is_waxpby:
			fmul ${tmp_v[1]}, ${tmp_v[1]}, ${beta_v}
		%endif
		%if is_vecadd:
			fadd ${tmp_v[1]}, ${tmp_v[1]}, ${tmp_v[5]}
		%else:
			fmla ${tmp_v[1]}, ${tmp_v[5]}, ${alpha_v}
		%endif
	%endif

	%if is_copy:
		str ${tmp_q[5]}, [DY], ${vector_bytes * 1}
	%elif is_waxpby:
		str ${tmp_q[1]}, [DW]
		add DW, DW, ${vector_bytes * 1}
		add DY, DY, ${vector_bytes * 1}
	%else:
		str ${tmp_q[1]}, [DY], ${vector_bytes * 1}
	%endif
</%def>

<%def name="contig_kernel_scalar(element_bytes)">
	ldr ${tmp_scalar[0]}, [DX], ${element_bytes}

	%if is_scal or is_scal_out_of_place or is_scal_real_cplx:
		fmul ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}
	%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
		ldr ${tmp_scalar[1]}, [DY]
		%if is_axpby or is_waxpby:
			fmul ${tmp_scalar[1]}, ${beta}, ${tmp_scalar[1]}
		%endif
		%if is_vecadd:
			fadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${tmp_scalar[1]}
		%else:
			fmadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}, ${tmp_scalar[1]}
		%endif
	%endif

	%if is_waxpby:
		str ${tmp_scalar[0]}, [DW], ${element_bytes}
		add DY, DY, ${element_bytes}
	%else:
		str ${tmp_scalar[0]}, [DY], ${element_bytes}
	%endif
</%def>

<%def name="mixed_stride_one_vector(nonstrided_v, strided_v, strided_D, inc_strided, prec)">
	<%
	if strided_D == "DX":
		strided_mult = alpha_v
		nonstrided_mult = beta_v
		dy_v = nonstrided_v
		dx_v = strided_v
	else:
		strided_mult = beta_v
		nonstrided_mult = alpha_v
		dy_v = strided_v
		dx_v = nonstrided_v
	# If its single precision one vector is 4 elements, otherwise for
	# doubles it will be 2 so when we need to assign the right increment.
	if prec == "s":
		inc_v = "INC4"
	else:
		inc_v = "INC2"
	%>
	// When copy or scal are strided in DX they need strided loads
	// otherwise we load using ldp earlier. Everything else needs some strided loads.
	%if not (is_copy or is_scal_out_of_place) or strided_D == "DX":
		ldr ${tmp_scalar[strided_v]}, [${strided_D}]
		ldr ${tmp_scalar[2]}, [${strided_D}, ${inc_strided}]
		mov ${tmp_v_clean[strided_v]}[1], ${tmp_v_clean[2]}[0]
		%if prec == "s":
			ldr ${tmp_scalar[3]}, [${strided_D}, INC2]
			ldr ${tmp_scalar[4]}, [${strided_D}, INC3]
			mov ${tmp_v_clean[strided_v]}[2], ${tmp_v_clean[3]}[0]
			mov ${tmp_v_clean[strided_v]}[3], ${tmp_v_clean[4]}[0]
		%endif
	%endif

	%if is_scal_out_of_place or is_axpy or is_vecadd:
		%if is_axpy:
			fmla ${tmp_v[dy_v]}, ${tmp_v[dx_v]}, ${alpha_v}
		%elif is_vecadd:
			fadd ${tmp_v[dy_v]}, ${tmp_v[dy_v]}, ${tmp_v[dx_v]}
		%else:
			fmul ${tmp_v[dy_v]}, ${tmp_v[dx_v]}, ${alpha_v}
		%endif
		%if strided_D == "DX":
			mov ${tmp_v_bytes[dx_v]}, ${tmp_v_bytes[dy_v]}
		%endif
	%elif is_axpby or is_waxpby:
		fmul ${tmp_v[strided_v]}, ${tmp_v[strided_v]}, ${strided_mult}
 		fmla ${tmp_v[strided_v]}, ${tmp_v[nonstrided_v]}, ${nonstrided_mult}
	%endif

	%if is_waxpby:
		str ${tmp_q[strided_v]}, [DW], ${vector_bytes}
		%if strided_D == "DY":
			add DY, DY, ${inc_v}
		%else:
			add DY, DY, ${vector_bytes}
		%endif
	%else:
		%if strided_D == "DX":
			str ${tmp_q[strided_v]}, [DY], ${vector_bytes}
		%else:
			str ${tmp_scalar[strided_v]}, [DY]
			mov ${tmp_v_clean[strided_v]}[0], ${tmp_v_clean[strided_v]}[1]
			str ${tmp_scalar[strided_v]}, [DY, INCY]
			%if prec == "s":
				mov ${tmp_v_clean[strided_v]}[0], ${tmp_v_clean[strided_v]}[2]
				str ${tmp_scalar[strided_v]}, [DY, INC2]
				mov ${tmp_v_clean[strided_v]}[0], ${tmp_v_clean[strided_v]}[3]
				str ${tmp_scalar[strided_v]}, [DY, INC3]
			%endif
			add DY, DY, ${inc_v}
		%endif
	%endif
	%if strided_D == "DX":
		add DX, DX, ${inc_v}
	%endif
</%def>

<%def name="mixed_stride_two_vectors(strided_D,prec)">
	<%
	if strided_D == "DX":
		inc = "INCX"
		nonstrided_D = "DY"
	else:
		inc = "INCY"
		nonstrided_D = "DX"
	%>

	%if not (is_copy or is_scal_out_of_place) or strided_D == "DY":
		// When copy or scal are strided in DX they only need strided loads
		// so this can't be used.
		%if strided_D == "DY":
			ldp ${tmp_q[5]}, ${tmp_q[6]}, [${nonstrided_D}], ${vector_bytes * 2}
		%else:
			ldp ${tmp_q[5]}, ${tmp_q[6]}, [${nonstrided_D}]
		%endif
	%endif

	%if is_copy:
		${mixed_stride_one_vector(5,5,strided_D,inc,prec)}
		${mixed_stride_one_vector(6,6,strided_D,inc,prec)}
	%else:
		${mixed_stride_one_vector(5,1,strided_D,inc,prec)}
		${mixed_stride_one_vector(6,1,strided_D,inc,prec)}
	%endif
</%def>

%for routine in [ "copy", "scal", "scal_real_cplx", "scal_out_of_place", "axpy", "axpy_no_prefetch", "axpby", "waxpby", "vecadd"]:
## Routine name to operation mapping:
## copy : y = x
## scal : x = a * x
## scal_out_of_place : y = a * x
## scal_real_cplx : x = a * x, where a is real
## axpy : y = a * x + y
## axpby : y = a * x + b * y
## waxpby : w = a * x + b * y
## vecacc : y = x + y
	%for prec, element_bytes in [("s", 4), ("d", 8)]:
		<%
		kernel = prec + routine + "_kernel"
		is_copy = "copy" == routine
		is_scal = "scal" == routine
		is_axpy = "axpy" == routine
		is_axpby = "axpby" == routine
		is_waxpby = "waxpby" == routine
		is_scal_real_cplx = "scal_real_cplx" == routine
		is_scal_out_of_place = "scal_out_of_place" == routine
		is_vecadd = "vecadd" == routine


		# Convenience variables for the number of bytes/elements in a vector
		vector_bytes = 16
		vector_elements = vector_bytes // element_bytes

		# fp inputs
		alpha = "{}0".format(prec)
		alpha_v = "v0.{}[0]".format(prec)
		beta = "{}1".format(prec)
		beta_v = "v1.{}[0]".format(prec)

		# v16-v31 are used for temporaries
		tmp_scalar = ["{}{}".format(prec, i) for i in range(16, 32)]
		tmp_q = ["q{}".format(i) for i in range(16, 32)]
		tmp_v = ["v{}.{}{}".format(i, vector_elements, prec) for i in range(16, 32)]
		tmp_v_clean = ["v{}.{}".format(i, prec) for i in range(16,32)]
		tmp_v_bytes = ["v{}.16b".format(i) for i in range(16,32)]

		strided_unroll_factor = 4
		%>

		${prologue(kernel)}
		stp x29, x30, [sp, #-16]!
		mov x29, sp

		%if is_scal or is_scal_real_cplx:
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

		%if is_waxpby:
			cmp INCW, #1
			bne .L${kernel}_use_noncontiguous_loop
		%endif
		// First check if this is 'contiguous' (i.e. incx and incy are 1)
		cmp INCX, #1
		// if INCX is strided branch down and check if we want mixed or completely strided
		bne .L${kernel}_noncontiguous_check_incy
		cmp INCY, #1
		// We already know INCX isn't strided so just go straight to mixed_stride
		// where only incy strided
		bne .L${kernel}_use_mixed_stride_incy_loop

		%if is_scal_real_cplx:
			// CS/ZD have 2n numbers, so double length
			lsl x0, x0, #1
		%endif

		// align x
		%if is_copy or is_scal or is_scal_real_cplx:
			mov x11, #15

			mov x2, #1

			%if prec == "s":
				lsl x2, x2, #2
			%else:
				lsl x2, x2, #3
			%endif

			// If DX is not a multiple of x2 skip alignment
			sub TMP, x2, #1
			ands TMP, DX, TMP
			bne .L${kernel}_contiguous_four_vectors_check
			b .L${kernel}_contiguous_loop2_align_cond

			.L${kernel}_contiguous_loop2_align:
				ldr ${prec}2, [DX]
				add DX, DX, x2
				%if not is_copy:
					fmul ${prec}2, ${prec}2, ${prec}0
				%endif

				str ${prec}2, [DY]
				add DY, DY, x2

				subs N, N, #1
				beq .L${kernel}_end

			.L${kernel}_contiguous_loop2_align_cond:
				ands TMP, DX, x11
				bne .L${kernel}_contiguous_loop2_align
		%endif

.L${kernel}_contiguous_four_vectors_check:
		// if N isn't larger or equal to 4 * vector_elements, process two vectors
		cmp N, ${4 * vector_elements}
		bge .L${kernel}_contiguous_four_vectors

.L${kernel}_contiguous_two_vectors_check:
		// if N isn't larger or equal to 2 * vector_elements, process one vector
		cmp N, ${2 * vector_elements}
		bge .L${kernel}_contiguous_two_vectors

.L${kernel}_contiguous_one_vector_check:
		// if N isn't larger or equal to vector_elements, use the scalar loop.
		cmp N, ${1 * vector_elements}
		bge .L${kernel}_contiguous_one_vector
		// if zero, exit, otherwise process it (the scalar loop will break if entered with 0)
		cmp N, 0
		beq .L${kernel}_end
		b .L${kernel}_contiguous_scalar

.L${kernel}_contiguous_four_vectors:
		${contig_kernel_four_vectors()}

		subs N, N, ${4 * vector_elements}
		beq .L${kernel}_end

		b .L${kernel}_contiguous_four_vectors_check

.L${kernel}_contiguous_two_vectors:
		// unroll 8 single prec or 4 double prec elements
		${contig_kernel_two_vectors()}

		subs N, N, ${2 * vector_elements}
		beq .L${kernel}_end

		b .L${kernel}_contiguous_two_vectors_check

.L${kernel}_contiguous_one_vector:
		// unroll 4 single prec or 2 double prec elements
		${contig_kernel_one_vector()}

		subs N, N, ${1 * vector_elements}
		beq .L${kernel}_end

		b .L${kernel}_contiguous_one_vector_check

.L${kernel}_contiguous_scalar:
		## this uses an s or d register to process 1 element
		beq .L${kernel}_end
		${contig_kernel_scalar(element_bytes)}
		subs N, N, 1
		beq .L${kernel}_end
		b .L${kernel}_contiguous_scalar

.L${kernel}_noncontiguous_check_incy:
		cmp INCY, #1
		bne .L${kernel}_use_noncontiguous_loop
		// Otherwise fall into mixed_stride where inxc strided, incy isn't


.L${kernel}_use_mixed_stride_incx_loop:
		lsl INCX, INCX, ${int(math.log(element_bytes,2))}
		lsl INC2, INCX, #1
		add INC3, INC2, INCX
		lsl INC4, INC2, #1

.L${kernel}_mixed_stride_incx_loop2_check:
		// Check if n is large enough to use the 2 vector kernel
		cmp N, ${2 * vector_elements}
		blt .L${kernel}_mixed_stride_loop1_incx_check

.L${kernel}_mixed_stride_incx_loop2:

		${mixed_stride_two_vectors("DX",prec)}

		sub N, N, ${2 * vector_elements}
		cmp N, ${2 * vector_elements}
		bge .L${kernel}_mixed_stride_incx_loop2

.L${kernel}_mixed_stride_loop1_incx_check:
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_mixed_stride_incx_loop1:

		ldr ${tmp_scalar[0]}, [DX]
		add DX, DX, INCX
		sub N, N, 1
		%if is_scal_out_of_place:
			fmul ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}
		%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
			ldr ${tmp_scalar[1]}, [DY]
			%if is_axpby or is_waxpby:
				fmul ${tmp_scalar[1]}, ${beta}, ${tmp_scalar[1]}
			%endif
			%if is_vecadd:
				fadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${tmp_scalar[1]}
			%else:
				fmadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}, ${tmp_scalar[1]}
			%endif
		%endif

		%if is_waxpby:
			str ${tmp_scalar[0]}, [DW], ${element_bytes}
			add DY, DY, ${element_bytes}
		%else:
			str ${tmp_scalar[0]}, [DY], ${element_bytes}
		%endif

		b .L${kernel}_mixed_stride_loop1_incx_check


.L${kernel}_use_mixed_stride_incy_loop:
		lsl INCY, INCY, ${int(math.log(element_bytes,2))}
		lsl INC2, INCY, #1
		add INC3, INC2, INCY
		lsl INC4, INC2, #1

.L${kernel}_mixed_stride_incy_loop2_check:
		// Check if n is large enough to use the 2 vector kernel
		cmp N, ${2 * vector_elements}
		blt .L${kernel}_mixed_stride_incy_loop1_check

.L${kernel}_mixed_stride_incy_loop2:

		${mixed_stride_two_vectors("DY",prec)}

		sub N, N, ${2 * vector_elements}
		cmp N, ${2 * vector_elements}
		bge .L${kernel}_mixed_stride_incy_loop2

.L${kernel}_mixed_stride_incy_loop1_check:
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_mixed_stride_incy_loop1:

		ldr ${tmp_scalar[0]}, [DX], ${element_bytes}
		sub N, N, 1
		%if is_scal_out_of_place:
			fmul ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}
		%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
			ldr ${tmp_scalar[1]}, [DY]
			%if is_axpby or is_waxpby:
				fmul ${tmp_scalar[1]}, ${beta}, ${tmp_scalar[1]}
			%endif
			%if is_vecadd:
				fadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${tmp_scalar[1]}
			%else:
				fmadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}, ${tmp_scalar[1]}
			%endif
		%endif

		%if is_waxpby:
			str ${tmp_scalar[0]}, [DW], ${element_bytes}
		%else:
			str ${tmp_scalar[0]}, [DY]
		%endif
		add DY, DY, INCY

		b .L${kernel}_mixed_stride_incy_loop1_check


.L${kernel}_use_noncontiguous_loop:
		<% element_bytes = element_bytes * 2 if is_scal_real_cplx else element_bytes %>
		mov TMP, ${element_bytes}
		mul INCX, INCX, TMP
		mul INCY, INCY, TMP
		%if is_waxpby:
			mul INCW, INCW, TMP
		%endif

		<% unroll_factor = strided_unroll_factor %>
		cmp N, ${unroll_factor}
		blt .L${kernel}_noncontiguous_loop_unrolled_end

.L${kernel}_noncontiguous_loop_unrolled:
		%for _ in range(unroll_factor):
			%if is_scal_real_cplx:
				ldp ${tmp_scalar[0]}, ${tmp_scalar[1]}, [DX]
			%else:
				ldr ${tmp_scalar[0]}, [DX]
			%endif

			add DX, DX, INCX
			%if is_scal or is_scal_out_of_place or is_scal_real_cplx:
				fmul ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}
				%if is_scal_real_cplx:
					fmul ${tmp_scalar[1]}, ${tmp_scalar[1]}, ${alpha}
				%endif
			%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
				ldr ${tmp_scalar[1]}, [DY]
				%if is_axpby or is_waxpby:
					fmul ${tmp_scalar[1]}, ${beta}, ${tmp_scalar[1]}
				%endif
				%if is_vecadd:
					fadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${tmp_scalar[1]}
				%else:
					fmadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}, ${tmp_scalar[1]}
				%endif
			%endif

			%if is_waxpby:
				str ${tmp_scalar[0]}, [DW]
				add DW, DW, INCW
			%elif is_scal_real_cplx:
				stp ${tmp_scalar[0]}, ${tmp_scalar[1]}, [DY]
			%else:
				str ${tmp_scalar[0]}, [DY]
			%endif

			add DY, DY, INCY
		%endfor

		sub N, N, ${unroll_factor}
		cmp N, ${unroll_factor}
		bge .L${kernel}_noncontiguous_loop_unrolled

.L${kernel}_noncontiguous_loop_unrolled_end:
		cmp N, 0
		beq .L${kernel}_end

.L${kernel}_noncontiguous_loop_tail:

		%if is_scal_real_cplx:
			ldp ${tmp_scalar[0]}, ${tmp_scalar[1]}, [DX]
		%else:
			ldr ${tmp_scalar[0]}, [DX]
		%endif
		add DX, DX, INCX

		%if is_scal or is_scal_out_of_place or is_scal_real_cplx:
			fmul ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}
			%if is_scal_real_cplx:
				fmul ${tmp_scalar[1]}, ${tmp_scalar[1]}, ${alpha}
			%endif
		%elif is_axpy or is_axpby or is_waxpby or is_vecadd:
			ldr ${tmp_scalar[1]}, [DY]
			%if is_axpby or is_waxpby:
				fmul ${tmp_scalar[1]}, ${beta}, ${tmp_scalar[1]}
			%endif
			%if is_vecadd:
				fadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${tmp_scalar[1]}
			%else:
				fmadd ${tmp_scalar[0]}, ${tmp_scalar[0]}, ${alpha}, ${tmp_scalar[1]}
			%endif
		%endif

		%if is_waxpby:
			str ${tmp_scalar[0]}, [DW]
			add DW, DW, INCW
		%elif is_scal_real_cplx:
			stp ${tmp_scalar[0]}, ${tmp_scalar[1]}, [DY]
		%else:
			str ${tmp_scalar[0]}, [DY]
		%endif

		add DY, DY, INCY

		sub N, N, 1
		cmp N, 0
		bne .L${kernel}_noncontiguous_loop_tail

.L${kernel}_end:
		ldp x29, x30, [sp], #16
		ret
		${epilogue(kernel)}
	%endfor
%endfor
