## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

/* PREFACE: This should be largely identical to zgemv_t, since we don't need ld2
* for loading from x in the non-transposed case. Changes are highlighted by
* comments starting with 'INC2:'. */

<%def name="zgemv(t)">

	<%
		fmlx0 = 'fmla'
		fmlx1 = 'fmla' if t == 'c' else 'fmls'
		fmlx2 = 'fmla'
		fmlx3 = 'fmls' if t == 'c' else 'fmla'
	%>

	# We need this for logarithms when adding across many vectors in a tree
	<% import math %>
	## This is how many multiplications per load we're doing. If this is 2, we step
	## COLUMN += 2 * LDA, for example, since we're processing 2 columns at once.
	## This means we only load from X once per 2 muls, hence the name.
	<% NUM_COLUMNS = 7 %>
	<% ORIGINAL_NUM_COLUMNS = NUM_COLUMNS %> ## Used for the tail loop later on
	<% NUM_PRELOAD = 7 %> ## How many registers to consume for preloading
	<% assert NUM_PRELOAD <= NUM_COLUMNS %>\

	INC_X          .req x5 // Since we don't care about INC_X and INC_Y here, these are just scratch registers.
	INC_Y          .req x7

	M              .req x0
	N              .req x1
	A              .req x2
	LDA            .req x3
	X              .req x4
	Y              .req x6
	ALPHA          .req v0
	BETA           .req v1
	M_ROUND_DOWN_8 .req x9  // M rounded down to the nearest multiple of the vector
	// register size * 8, then multiplied by 4 (since floats
	// are 4 bytes). Used for loops unrolled 8 times.
	M_ROUND_DOWN_4 .req x10 // For loops unrolled 4 times.
	M_ROUND_DOWN_2 .req x11 // For loops unrolled 2 times.
	M_ROUND_DOWN_1 .req x12 // For loops that aren't unrolled.
	## x13, x14 unavailable on Arm64EC
	N_ROUND_DOWN   .req x15 // Rounding down N since we process multiple columns at once
	X_ADDR         .req x16
	INTER_COLUMN_STEP .req x17 // The step needed to get from the end of one column to the start of the next
	INTER_COLUMN_STEP_TAIL .req x3 // The step needed to get from the end of one column to the start of the next in the tail loop

	// Each A_ADDR stands for a different address in the A matrix - we loop over
	// columns in the matrix in multiples of ${NUM_COLUMNS} to avoid loading X
	// multiple times.
	A_ADDR_0 .req x2 // A_ADDR_0 aliases A
	## Avoid using x18, x23, x24, x28 and x29 when allocating general-purpose registers
	## so that our code remains portable, including on Arm64EC. We also disallow x30,
	## because it can break backtraces and perf profiles.
	<% gpreg = 17 %>
	%for column in range(1, NUM_COLUMNS):
		<% gpreg += 1 %>
		%if gpreg == 18:
			<% gpreg += 1 %>
		%endif
		%if gpreg == 23:
			<% gpreg += 2 %>
		%endif
		<% assert gpreg <= 27, f"gpreg exceeded 27 (got {gpreg})" %>
		A_ADDR_${column} .req x${gpreg}
	%endfor

	// .req the various vector registers in use
	X_REG_X_V .req v2
	X_REG_X_Q .req q2
	X_REG_X_D .req d2
	X_REG_Y_V .req v3
	X_REG_Y_Q .req q3
	X_REG_Y_D .req d3
	%for column in range(NUM_COLUMNS):
		ACC_REG_X_${column}_V .req v${4 + 2*(NUM_PRELOAD + column)}
		ACC_REG_X_${column}_D .req d${4 + 2*(NUM_PRELOAD + column)}
		ACC_REG_Y_${column}_V .req v${5 + 2*(NUM_PRELOAD + column)}
		ACC_REG_Y_${column}_D .req d${5 + 2*(NUM_PRELOAD + column)}
	%endfor
	%for pl in range(NUM_PRELOAD):
		A_REG_X_${pl}_V .req v${4 + 2*pl}
		A_REG_X_${pl}_Q .req q${4 + 2*pl}
		A_REG_X_${pl}_D .req d${4 + 2*pl}
		A_REG_Y_${pl}_V .req v${5 + 2*pl}
		A_REG_Y_${pl}_Q .req q${5 + 2*pl}
		A_REG_Y_${pl}_D .req d${5 + 2*pl}
	%endfor

	<% func_name = "zgemv_inc2_" + t + "_neon_kernel" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-160]!
	mov x29, sp
	stp  d8,  d9, [sp, #16]
	stp d10, d11, [sp, #32]
	stp d12, d13, [sp, #48]
	stp d14, d15, [sp, #64]
	stp x19, x20, [sp, #80]
	stp x21, x22, [sp, #96]
	stp x23, x24, [sp, #112]
	stp x25, x26, [sp, #128]
	stp x27, x28, [sp, #144]

	// Setup alpha & beta
	mov v0.d[1], v1.d[0]
	mov v1.d[0], v2.d[0]
	mov v1.d[1], v3.d[0]

	mov x5, #${NUM_COLUMNS-1}
	// Setup inter-column-step and LDA in bytes. Since INTER_COLUMN_STEP_TAIL aliases LDA, set it up as x7 first.
	sub x7, LDA, M
	lsl x7, x7, #4
	sub INTER_COLUMN_STEP, LDA, M
	lsl INTER_COLUMN_STEP, INTER_COLUMN_STEP, #4
	lsl LDA, LDA, #4 // 16b per element
	madd INTER_COLUMN_STEP, LDA, x5, INTER_COLUMN_STEP
	mov x5, #${NUM_COLUMNS}

	// Init N_ROUND_DOWN and M_ROUND_DOWN values
	udiv N_ROUND_DOWN, N, x5
	mul N_ROUND_DOWN, N_ROUND_DOWN, x5
	lsl N, N, #4
	lsl N_ROUND_DOWN, N_ROUND_DOWN, #4
	//INC2: Extra lsl here
	add N, Y, N, lsl #1
	add N_ROUND_DOWN, Y, N_ROUND_DOWN, lsl #1

	asr M_ROUND_DOWN_8, M, #5
	lsl M_ROUND_DOWN_8, M_ROUND_DOWN_8, #9
	asr M_ROUND_DOWN_4, M, #4
	lsl M_ROUND_DOWN_4, M_ROUND_DOWN_4, #8
	asr M_ROUND_DOWN_2, M, #3
	lsl M_ROUND_DOWN_2, M_ROUND_DOWN_2, #7
	asr M_ROUND_DOWN_1, M, #2
	lsl M_ROUND_DOWN_1, M_ROUND_DOWN_1, #6

	// INC2: Extra lsl
	add M_ROUND_DOWN_8, X, M_ROUND_DOWN_8, lsl #1 // Offset to X, to use as a CMP op
	add M_ROUND_DOWN_4, X, M_ROUND_DOWN_4, lsl #1
	add M_ROUND_DOWN_2, X, M_ROUND_DOWN_2, lsl #1
	add M_ROUND_DOWN_1, X, M_ROUND_DOWN_1, lsl #1

	// Init A_ADDR_X
	%for column in range(1, NUM_COLUMNS):
		mov x5, ${column}
		madd A_ADDR_${column}, LDA, x5, A
	%endfor

	// Actually init INTER_COLUMN_STEP_TAIL.
	mov INTER_COLUMN_STEP_TAIL, x7

	cmp M, 200
	bgt .L_${t}_size_big

	## Big kernels we prefetch, small we don't
	%for size in ["small", "big"]:
		.L_${t}_size_${size}:

		// INC2: extra lsl
		lsl M, M, #5
		add M, M, X


		## Generate one loop like normal, and another to clean up the tail columns
		%for is_tail, tail in [(False, ''), (True, 'tail')]:
			<%
				NUM_COLUMNS = 1 if is_tail else ORIGINAL_NUM_COLUMNS
			%>
			// Loop over the columns
			b .L_${t}_loop_cond_${size}_${tail}
			.L_${t}_loop_${size}_${tail}:

			mov X_ADDR, X

			## Create an unrolled loop with the given unroll level and the given register
			## to use as a limit for X_ADDR (for example, M_ROUND_DOWN). This doesn't
			## handle tails, and just loops X_ADDR until the limit. Make sure limit -
			## X_ADDR is divisible by level * 4 * 4, otherwise this will go over the limit
			## slightly.
			<%def name="loop_unrolled(level, limit, size, is_tail, tail)">
				b .L_${t}_column_loop_unrlvl${level}_cond_${size}_${tail}
				.L_${t}_column_loop_unrlvl${level}_${size}_${tail}:
				## Keep track of the current reg we're using / preloading
				<% curr_a_reg = 0 %>
				%for l in range(level):
					// INC2: single structure LD2s
					ld2 { X_REG_X_V.d, X_REG_Y_V.d }[0], [X_ADDR]
					add X_ADDR, X_ADDR, #32
					ld2 { X_REG_X_V.d, X_REG_Y_V.d }[1], [X_ADDR]
					add X_ADDR, X_ADDR, #32
					## Since we're preloading registers, initially load them in if this is the
					## first unroll
					%if is_tail:
						ld2 { A_REG_X_0_V.2d, A_REG_Y_0_V.2d }, [A_ADDR_0], #32
						${fmlx0} ACC_REG_X_0_V.2d, A_REG_X_0_V.2d, X_REG_X_V.2d
						${fmlx1} ACC_REG_X_0_V.2d, A_REG_Y_0_V.2d, X_REG_Y_V.2d
						${fmlx2} ACC_REG_Y_0_V.2d, A_REG_X_0_V.2d, X_REG_Y_V.2d
						${fmlx3} ACC_REG_Y_0_V.2d, A_REG_Y_0_V.2d, X_REG_X_V.2d
					%else:
						%if l == 0:
							%for pl in range(NUM_PRELOAD):
								ld2 { A_REG_X_${pl}_V.2d, A_REG_Y_${pl}_V.2d }, [A_ADDR_${pl}], #32
							%endfor
						%endif
						%for column in range(NUM_COLUMNS):
							<% preloading_next = (column >= NUM_COLUMNS - NUM_PRELOAD) %>
							<% curr_preload = (column+NUM_PRELOAD) if not preloading_next else ((column+NUM_PRELOAD)%NUM_COLUMNS) %>
							## Prefetch if this is a multiple of 4
							%if l %2 == 0 and size == "big":
								prfm pldl1keep, [A_ADDR_${column}, #128]
							%endif
							${fmlx0} ACC_REG_X_${column}_V.2d, A_REG_X_${curr_a_reg}_V.2d, X_REG_X_V.2d
							${fmlx1} ACC_REG_X_${column}_V.2d, A_REG_Y_${curr_a_reg}_V.2d, X_REG_Y_V.2d
							${fmlx2} ACC_REG_Y_${column}_V.2d, A_REG_X_${curr_a_reg}_V.2d, X_REG_Y_V.2d
							${fmlx3} ACC_REG_Y_${column}_V.2d, A_REG_Y_${curr_a_reg}_V.2d, X_REG_X_V.2d
							## Preload the next register - use different instructions if we're at the end
							## of the current mini-unroll
							%if l < level-1:
								ld2 { A_REG_X_${curr_a_reg}_V.2d, A_REG_Y_${curr_a_reg}_V.2d }, [A_ADDR_${curr_preload}], #32
							%endif
							## Increment the preload target reg num
							<% curr_a_reg = (curr_a_reg + 1) %NUM_PRELOAD %>
						%endfor
					%endif
				%endfor

				.L_${t}_column_loop_unrlvl${level}_cond_${size}_${tail}:
				cmp X_ADDR, ${limit}
				blt .L_${t}_column_loop_unrlvl${level}_${size}_${tail}
			</%def>

			// Init registers for accumulating
			%for column in range(NUM_COLUMNS):
				movi ACC_REG_X_${column}_V.2d, #0
				movi ACC_REG_Y_${column}_V.2d, #0
			%endfor

			// Use a progressively less unrolled loop
			${loop_unrolled(4, "M_ROUND_DOWN_4", size, is_tail, tail)}
			${loop_unrolled(2, "M_ROUND_DOWN_2", size, is_tail, tail)}
			${loop_unrolled(1, "M_ROUND_DOWN_1", size, is_tail, tail)}

			// Finish the loop with scalar instructions
			b .L_${t}_column_loop_tail_cond_${size}_${tail}
			.L_${t}_column_loop_tail_${size}_${tail}:
			ldr X_REG_X_D, [X_ADDR]
			ldr X_REG_Y_D, [X_ADDR, #8]
			%for column in range(NUM_COLUMNS):
				ldr A_REG_X_${column}_D, [A_ADDR_${column}]
				ldr A_REG_Y_${column}_D, [A_ADDR_${column}, #8]
				${fmlx0} ACC_REG_X_${column}_V.2d, A_REG_X_${column}_V.2d, X_REG_X_V.2d
				${fmlx1} ACC_REG_X_${column}_V.2d, A_REG_Y_${column}_V.2d, X_REG_Y_V.2d
				${fmlx2} ACC_REG_Y_${column}_V.2d, A_REG_X_${column}_V.2d, X_REG_Y_V.2d
				${fmlx3} ACC_REG_Y_${column}_V.2d, A_REG_Y_${column}_V.2d, X_REG_X_V.2d
				add A_ADDR_${column}, A_ADDR_${column}, #16
			%endfor
			//INC2: 32 not 16 here
			add X_ADDR, X_ADDR, #32
			.L_${t}_column_loop_tail_cond_${size}_${tail}:
			cmp X_ADDR, M
			blt .L_${t}_column_loop_tail_${size}_${tail}

			// Prefetch the next loop
			%for column in range(NUM_COLUMNS):
				%if is_tail:
					prfm pldl1keep, [A_ADDR_${column}, INTER_COLUMN_STEP_TAIL]
				%else:
					prfm pldl1keep, [A_ADDR_${column}, INTER_COLUMN_STEP]
				%endif
			%endfor

			// Sum across accumulator registers
			%for column in range (NUM_COLUMNS):
				faddp ACC_REG_X_${column}_V.2d, ACC_REG_X_${column}_V.2d, ACC_REG_X_${column}_V.2d
				faddp ACC_REG_Y_${column}_V.2d, ACC_REG_Y_${column}_V.2d, ACC_REG_Y_${column}_V.2d
				// Store in Y
				//INC2: 32 * not 16 *
				ldr X_REG_X_D, [Y, #${32 * column}]
				ldr X_REG_Y_D, [Y, #${32 * column + 8}]
				fmla X_REG_X_V.2d, ACC_REG_X_${column}_V.2d, ALPHA.d[0]
				fmls X_REG_X_V.2d, ACC_REG_Y_${column}_V.2d, ALPHA.d[1]
				fmla X_REG_Y_V.2d, ACC_REG_X_${column}_V.2d, ALPHA.d[1]
				fmla X_REG_Y_V.2d, ACC_REG_Y_${column}_V.2d, ALPHA.d[0]
				str X_REG_X_D, [Y, #${32 * column}]
				str X_REG_Y_D, [Y, #${32 * column + 8}]
				## Increment A_ADDR_X
				%if is_tail:
					add A_ADDR_${column}, A_ADDR_${column}, INTER_COLUMN_STEP_TAIL
				%else:
					add A_ADDR_${column}, A_ADDR_${column}, INTER_COLUMN_STEP
				%endif
			%endfor
			// INC2: 32 not 16
			add Y, Y, #${32 * NUM_COLUMNS}
			.L_${t}_loop_cond_${size}_${tail}:
			%if is_tail:
				cmp Y, N
			%else:
				cmp Y, N_ROUND_DOWN
			%endif
			blt .L_${t}_loop_${size}_${tail}

		%endfor // End tail / notail loop
		b .L_${t}_cleanup
	%endfor

	.L_${t}_cleanup:
	ldp  d8,  d9, [sp, #16]
	ldp d10, d11, [sp, #32]
	ldp d12, d13, [sp, #48]
	ldp d14, d15, [sp, #64]
	ldp x19, x20, [sp, #80]
	ldp x21, x22, [sp, #96]
	ldp x23, x24, [sp, #112]
	ldp x25, x26, [sp, #128]
	ldp x27, x28, [sp, #144]
	ldp x29, x30, [sp], #160
	ret

	${epilogue(func_name)}

</%def>

${zgemv('t')}
${zgemv('c')}
