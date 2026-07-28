## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

<%def name="load_vectors(first_reg, second_reg, offset, T)">
	%if T=='d':
		ldp q${x_vectors[first_reg]}, q${x_vectors[second_reg]}, [${dx}, ${offset*elem_size}]
		ldp q${y_vectors[first_reg]}, q${y_vectors[second_reg]}, [${dy}, ${offset*elem_size}]
	%elif T=='s':
		ldr q${x_vectors[first_reg]}, [${dx}, ${offset*elem_size}]
		ldr q${y_vectors[first_reg]}, [${dy}, ${offset*elem_size}]
	%else:
		ldr q${x_vectors[first_reg]}, [${dx}, ${offset*elem_size}]
		ldr q${y_vectors[first_reg]}, [${dy}, ${offset*elem_size}]
		fcvtl v${x_vectors[second_reg]}.2d, v${x_vectors[first_reg]}.2s
		fcvtl2 v${x_vectors[first_reg]}.2d, v${x_vectors[first_reg]}.4s
		fcvtl v${y_vectors[second_reg]}.2d, v${y_vectors[first_reg]}.2s
		fcvtl2 v${y_vectors[first_reg]}.2d, v${y_vectors[first_reg]}.4s
	%endif
</%def>

<%def name="multiply_vectors(first_idx, second_idx, T)">
	%if T=='s':
		fmla v${output_registers[first_idx]}.4s, v${x_vectors[first_idx]}.4s, v${y_vectors[first_idx]}.4s
	%else:
		fmla v${output_registers[first_idx]}.2d, v${x_vectors[first_idx]}.2d, v${y_vectors[first_idx]}.2d
		fmla v${output_registers[second_idx]}.2d, v${x_vectors[second_idx]}.2d, v${y_vectors[second_idx]}.2d
	%endif
</%def>

<%def name="handle_elements(no_of_elements, T)">
	%for i in range(0, (no_of_elements//2), 2):
		${load_vectors(i, i+1, i*2, T)}
		${multiply_vectors(i, i+1, T)}
	%endfor
	add ${dx}, ${dx}, ${no_of_elements}*${elem_size}
	add ${dy}, ${dy}, ${no_of_elements}*${elem_size}
</%def>

<%def name="scalar_update(T)">
	%if T=='s':
		ldr s16, [${dx}]
		ldr s24, [${dy}]
		fmla v0.4s, v16.4s, v24.4s
	%elif T=='ds':
		ldr s16, [${dx}]
		ldr s24, [${dy}]
		fcvtl v16.2d, v16.2s
		fcvtl v24.2d, v24.2s
		fmla v0.2d, v16.2d, v24.2d
	%else:
		ldr d16, [${dx}]
		ldr d24, [${dy}]
		fmla v0.2d, v16.2d, v24.2d
	%endif
</%def>

<%def name="do_reduction(T)">
	%if T=='s':
		fadd v0.4s, v0.4s, v2.4s
		fadd v4.4s, v4.4s, v6.4s
		fadd v0.4s, v0.4s, v4.4s
		faddp v0.4s, v0.4s, v0.4s
		faddp s0, v0.2s
	%else:
		fadd v0.2d, v0.2d, v1.2d
		fadd v2.2d, v2.2d, v3.2d
		fadd v4.2d, v4.2d, v5.2d
		fadd v6.2d, v6.2d, v7.2d
		fadd v0.2d, v0.2d, v2.2d
		fadd v4.2d, v4.2d, v6.2d
		fadd v0.2d, v0.2d, v4.2d
		faddp d0, v0.2d
	%endif
</%def>

## One iteration of the strided kernel. acc_id is the id of the register used for accumulation.
<%def name="strided_kernel(acc_id)">
	ldr ${prec_in}16, [${dx}]
	ldr ${prec_in}24, [${dy}]
	add ${dx}, ${dx}, ${incx}
	add ${dy}, ${dy}, ${incy}
	%if T == 'ds':
		fcvtl v16.2d, v16.2s
		fcvtl v24.2d, v24.2s
	%endif
	fmadd ${prec_out}${acc_id}, ${prec_out}16, ${prec_out}24, ${prec_out}${acc_id}
</%def>

% for T in [ 's', 'd', 'ds' ]:
<%
T = T # Make T global
if T=='d':
	elem_size = 8
else:
	elem_size = 4
name = T + "dot"
n = "x0"
dx = "x1"
dy = "x2"
incx = "x3"
incy = "x4"
prec_in = T[1] if len(T) == 2 else T[0]
prec_out = T[0]
x_vectors = ["16", "17", "18", "19", "20", "21", "22", "23"]
y_vectors = ["24", "25", "26", "27", "28", "29", "30", "31"]
output_registers = ["0", "1", "2", "3", "4", "5", "6", "7"]
%>
	<% func_name = name + "_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	cmp ${incx}, #1
	bne .L${name}_use_noncontiguous_loop
	cmp ${incy}, #1
	bne .L${name}_use_noncontiguous_loop

	fmov d0, xzr
	%for element in output_registers:
	dup v${element}.2d, xzr
	%endfor

	// Unrolling by 16 is the largest acceptable unroll factor for this loop.
	%if T != 's':
		.L${name}_loop32:
		cmp ${n}, 32
		blt .L${name}_loop16
		sub ${n}, ${n}, 32

		// 16
		${handle_elements(16, T)}

		PRFM PLDL1KEEP, [${dx}, #896]
		PRFM PLDL1KEEP, [${dy}, #896]
		PRFM PLDL1KEEP, [${dx}, #896+64]
		PRFM PLDL1KEEP, [${dy}, #896+64]

		//next 16
		${handle_elements(16, T)}

		PRFM PLDL1KEEP, [${dx}, #896]
		PRFM PLDL1KEEP, [${dy}, #896]
		PRFM PLDL1KEEP, [${dx}, #896+64]
		PRFM PLDL1KEEP, [${dy}, #896+64]

		b .L${name}_loop32
	%endif

	.L${name}_loop16:
	cmp ${n}, 16
	blt .L${name}_loop8
	sub ${n}, ${n}, 16
	// 8
	${handle_elements(8, T)}
	//next 8
	${handle_elements(8, T)}
	b .L${name}_loop16

	.L${name}_loop8:
	cmp ${n}, 8
	blt .L${name}_loop4
	sub ${n}, ${n}, 8
	// 4
	${handle_elements(4, T)}
	//next 4
	${handle_elements(4, T)}
	b .L${name}_loop8

	.L${name}_loop4:
	cmp ${n}, 4
	blt .L${name}_loop1
	sub ${n}, ${n}, 4
	${handle_elements(4, T)}
	b .L${name}_loop4

	.L${name}_loop1:
	cmp ${n}, 1
	blt .L${name}_loop_end
	sub ${n}, ${n}, 1

	${scalar_update(T)}

	add ${dx}, ${dx}, ${elem_size}
	add ${dy}, ${dy}, ${elem_size}

	b .L${name}_loop1


	.L${name}_loop_end:
	${do_reduction(T)}

	b .L${name}_kernel_end

.L${name}_use_noncontiguous_loop:

	<% unroll_factor = 4 %>

	// Initialize the accumulator
	%for i in range(unroll_factor):
		fmov d${i}, xzr
	%endfor

	// Scale incx and incy to enable pointer arithmetic
	mov x5, ${elem_size}
	mul ${incx}, ${incx}, x5
	mul ${incy}, ${incy}, x5

	cmp ${n}, ${unroll_factor}
	b.lt .L${name}_noncontiguous_loop_end

.L${name}_noncontiguous_loop:
	%for i in range(unroll_factor):
		${strided_kernel(i)}
	%endfor
	sub ${n}, ${n}, ${unroll_factor}
	cmp ${n}, ${unroll_factor}
	b.ge .L${name}_noncontiguous_loop

	%for i in range(1, unroll_factor):
		fadd ${prec_out}0, ${prec_out}0, ${prec_out}${i}
	%endfor

.L${name}_noncontiguous_loop_end:
	cmp ${n}, 0
	beq .L${name}_kernel_end

.L${name}_noncontiguous_loop_tail:
	${strided_kernel(0)}
	sub ${n}, ${n}, 1
	cmp ${n}, 0
	bne .L${name}_noncontiguous_loop_tail

.L${name}_kernel_end:
	ldp x29, x30, [sp], #16
	ret

	${epilogue(func_name)}
% endfor
