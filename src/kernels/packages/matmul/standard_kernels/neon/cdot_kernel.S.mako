## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

%if target_os in ('windows', 'windows_arm64ec'):
	n .req x1
	dx .req x2
	dy .req x3
	incx .req x4
	incy .req x5
%else:
	n .req x0
	dx .req x1
	dy .req x2
	incx .req x3
	incy .req x4
%endif
tmp .req x8

ret_re_0_s .req s0
ret_im_0_s .req s1
ret_re_0_d .req d0
ret_im_0_d .req d1
ret_re_0_v .req v0
ret_im_0_v .req v1

//skip to 16 so that we don't have to save any regs to the stack
x_re_0_s .req s16
x_im_0_s .req s17
x_re_0_d .req d16
x_im_0_d .req d17

y_re_0_s .req s18
y_im_0_s .req s19
y_re_0_d .req d18
y_im_0_d .req d19

x_re_0_v .req v16
x_im_0_v .req v17
x_re_1_v .req v20
x_im_1_v .req v21

y_re_0_v .req v18
y_im_0_v .req v19
y_re_1_v .req v22
y_im_1_v .req v23

ret_re_1_s .req s24
ret_im_1_s .req s25
ret_re_1_v .req v24
ret_im_1_v .req v25

/*
This kernel works in the following way:

cdot(int n, float *x, float *y) {
	//loop_8
	while(n>8) {
		//process 8 elements
		n-=8;
	}
	//loop_1
	while(n>0) {
		//process 1 elements
		--n;
	}
}

Point of consideration, this kernel uses two accumulator registers in the
vectorized loop which may have some impact on the accuracy (increased accuracy)
of the computation, if this becomes an issue the loop_8 could be place inside
an outer loop which would then reduce the accumulators at a higher frequency.
ie reduce every 8 iteration of unrolled loops (ie every 64 element) and store
the reduction in an accumulator that is difference to the compute accumulators

This kernel only modified x0, x1, x2 and vector registers >= 16.

This means that there is no reason to save any of the registers to the stack
in accordance with the aarch64 PCS -- other than the fp and lr which perf reads
for call stack info
*/

## A single iteration of a strided kernel.
##
## acc_id is the id of the accumulation registers. The real values are stored
## in r${2 * acc_id} and the imaginary values are in r${2 * acc_id + 1}
##
<%def name="strided_kernel(acc_id)">
	ldp ${prec}16, ${prec}17, [dx]
	ldp ${prec}24, ${prec}25, [dy]
	add dx, dx, incx
	add dy, dy, incy
	fmadd ${prec}${2 * acc_id}, ${prec}16, ${prec}24, ${prec}${2 * acc_id}
	${scalar_accumulate_0} ${prec}${2 * acc_id}, ${prec}17, ${prec}25, ${prec}${2 * acc_id}
	fmadd ${prec}${2 * acc_id + 1}, ${prec}16, ${prec}25, ${prec}${2 * acc_id + 1}
	${scalar_accumulate_1} ${prec}${2 * acc_id + 1}, ${prec}17, ${prec}24, ${prec}${2 * acc_id + 1}
</%def>

<%
modes=[
	( "cdot",      "fmls", "fmla", "fmsub", "fmadd", 4, "s"),
	( "cdot_conj", "fmla", "fmls", "fmadd", "fmsub", 4, "s"),
	( "zdot",      "fmls", "fmla", "fmsub", "fmadd", 2, "d"),
	( "zdot_conj", "fmla", "fmls", "fmadd", "fmsub", 2, "d")
]
%>
%for mode in modes:
	<%
		name = mode[0]
		func_name = name + "_kernel"
		prec = mode[6]

		vector_size = 16
		vector_elems = mode[5]
		elem_size = 2 * vector_size // vector_elems

		suffix = ".{}{}".format(vector_elems, prec)
		scalar_accumulate_0 = mode[3]
		scalar_accumulate_1 = mode[4]
	%>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	cmp incx, #1
	bne .L${name}_use_noncontiguous_loop
	cmp incy, #1
	bne .L${name}_use_noncontiguous_loop

	dup ret_re_0_v.2d, xzr
	dup ret_im_0_v.2d, xzr

	dup ret_re_1_v.2d, xzr
	dup ret_im_1_v.2d, xzr

//start of the loop that process 8 elements
//if there are fewer than 8 elements then the body of the loop will not execute
.L_${mode[0]}_loop_2:
	cmp n, ${vector_elems * 2 - 1}
	b .L_${mode[0]}_loop_2_end
.L_${mode[0]}_loop_2_body:
	sub n, n, ${vector_elems * 2}

	ld2 { x_re_0_v${suffix}, x_im_0_v${suffix} }, [dx]
	add dx, dx, ${vector_size * 2}
	ld2 { y_re_0_v${suffix}, y_im_0_v${suffix} }, [dy]
	add dy, dy, ${vector_size * 2}

	ld2 { x_re_1_v${suffix}, x_im_1_v${suffix} }, [dx]
	add dx, dx, ${vector_size * 2}
	ld2 { y_re_1_v${suffix}, y_im_1_v${suffix} }, [dy]
	add dy, dy, ${vector_size * 2}

	cmp n, ${vector_elems * 2 - 1}

	fmla ret_re_0_v${suffix}, x_re_0_v${suffix}, y_re_0_v${suffix}
	fmla ret_im_0_v${suffix}, x_re_0_v${suffix}, y_im_0_v${suffix}

	fmla ret_re_1_v${suffix}, x_re_1_v${suffix}, y_re_1_v${suffix}
	fmla ret_im_1_v${suffix}, x_re_1_v${suffix}, y_im_1_v${suffix}

	${mode[1]} ret_re_0_v${suffix}, x_im_0_v${suffix}, y_im_0_v${suffix}
	${mode[2]} ret_im_0_v${suffix}, x_im_0_v${suffix}, y_re_0_v${suffix}
	${mode[1]} ret_re_1_v${suffix}, x_im_1_v${suffix}, y_im_1_v${suffix}
	${mode[2]} ret_im_1_v${suffix}, x_im_1_v${suffix}, y_re_1_v${suffix}

.L_${mode[0]}_loop_2_end:
	bgt .L_${mode[0]}_loop_2_body

	fadd ret_re_0_v${suffix}, ret_re_0_v${suffix}, ret_re_1_v${suffix}
	fadd ret_im_0_v${suffix}, ret_im_0_v${suffix}, ret_im_1_v${suffix}

	faddp ret_re_0_v${suffix}, ret_re_0_v${suffix}, ret_re_0_v${suffix}
	faddp ret_im_0_v${suffix}, ret_im_0_v${suffix}, ret_im_0_v${suffix}

	%if prec == "s":
		faddp ret_re_0_v${suffix}, ret_re_0_v${suffix}, ret_re_0_v${suffix}
		faddp ret_im_0_v${suffix}, ret_im_0_v${suffix}, ret_im_0_v${suffix}
	%endif

.L_${mode[0]}_loop_1:
	cmp n, 0
	b .L_${mode[0]}_loop_1_end
.L_${mode[0]}_loop_1_body:
	subs n, n, 1
	ldp x_re_0_${prec}, x_im_0_${prec}, [dx]
	add dx, dx, ${elem_size}
	ldp y_re_0_${prec}, y_im_0_${prec}, [dy]
	add dy, dy, ${elem_size}

	fmadd ret_re_0_${prec}, x_re_0_${prec}, y_re_0_${prec}, ret_re_0_${prec}
	${mode[3]} ret_re_0_${prec}, x_im_0_${prec}, y_im_0_${prec}, ret_re_0_${prec}

	fmadd ret_im_0_${prec}, x_re_0_${prec}, y_im_0_${prec}, ret_im_0_${prec}
	${mode[4]} ret_im_0_${prec}, x_im_0_${prec}, y_re_0_${prec}, ret_im_0_${prec}

.L_${mode[0]}_loop_1_end:
	//if zero return
	bne .L_${mode[0]}_loop_1_body
	//falls through to end
	b .L_${func_name}_end

.L${name}_use_noncontiguous_loop:

	<%
	unroll_factor = 4
	assert unroll_factor <= 4, "Register allocation assumes an unroll_factor <= 4."
	%>

	// Initialize the accumulator
	%for i in range(2 * unroll_factor):
		fmov d${i}, xzr
	%endfor

	// Scale incx and incy to enable pointer arithmetic
	mov tmp, ${elem_size}
	mul incx, incx, tmp
	mul incy, incy, tmp

	cmp n, ${unroll_factor}
	b.lt .L${name}_noncontiguous_loop_end

.L${name}_noncontiguous_loop:
	%for i in range(unroll_factor):
		${strided_kernel(i)}
	%endfor
	sub n, n, ${unroll_factor}
	cmp n, ${unroll_factor}
	b.ge .L${name}_noncontiguous_loop

	%for i in range(1, unroll_factor):
		fadd ${prec}0, ${prec}0, ${prec}${2 * i}
		fadd ${prec}1, ${prec}1, ${prec}${2 * i + 1}
	%endfor

.L${name}_noncontiguous_loop_end:
	cmp n, 0
	beq .L_${func_name}_end

.L${name}_noncontiguous_loop_tail:
	${strided_kernel(0)}
	sub n, n, 1
	cmp n, 0
	bne .L${name}_noncontiguous_loop_tail

.L_${func_name}_end:
	ldp x29, x30, [sp], #16

%if target_os in ('windows', 'windows_arm64ec'):
	str ret_re_0_${prec}, [x0]
	str ret_im_0_${prec}, [x0, #${int(vector_size / vector_elems)}]
%endif

	//the return value is already in the right place s0 and s1
	ret

	${epilogue(func_name)}

%endfor
