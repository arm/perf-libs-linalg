## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

<%def name="cgemv(t)">

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

	%if target_os in ('windows', 'windows_arm64ec'):
	// We use x2 and x7 which originally store alpha and beta as scratch after we already move them to FP registers.
	TMP1           .req x2
	TMP2           .req x7

	M              .req x0
	N              .req x1
	A              .req x3
	LDA            .req x4
	X              .req x5
	Y              .req x6
	%else:
	// Since we don't care about INC_X and INC_Y here, these are just scratch registers.
	TMP1           .req x5
	TMP2           .req x7

	M              .req x0
	N              .req x1
	A              .req x2
	LDA            .req x3
	X              .req x4
	Y              .req x6
	%endif
	ALPHA          .req v0
	BETA           .req v1
	M_ROUND_DOWN_8 .req x9  // M rounded down to the nearest multiple of the vector
	// register size * 8, then multiplied by 4 (since floats
	// are 4 bytes). Used for loops unrolled 8 times.
	M_ROUND_DOWN_4 .req x10 // For loops unrolled 4 times.
	M_ROUND_DOWN_2 .req x11 // For loops unrolled 2 times.
	M_ROUND_DOWN_1 .req x12 // For loops that aren't unrolled.
	N_ROUND_DOWN   .req x15 // Rounding down N since we process multiple columns at once
	X_ADDR         .req x16
	INTER_COLUMN_STEP .req x17 // The step needed to get from the end of one column to the start of the next
	%if target_os in ('windows', 'windows_arm64ec'):
	INTER_COLUMN_STEP_TAIL .req x4 // The step needed to get from the end of one column to the start of the next in the tail loop
	%else:
	INTER_COLUMN_STEP_TAIL .req x3 // The step needed to get from the end of one column to the start of the next in the tail loop
	%endif

	// Each A_ADDR stands for a different address in the A matrix - we loop over
	// columns in the matrix in multiples of ${NUM_COLUMNS} to avoid loading X
	// multiple times.
	%if target_os in ('windows', 'windows_arm64ec'):
	A_ADDR_0 .req x3 // A_ADDR_0 aliases A
	%else:
	A_ADDR_0 .req x2 // A_ADDR_0 aliases A
	%endif
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
	X_REG_X_S .req s2
	X_REG_Y_V .req v3
	X_REG_Y_Q .req q3
	X_REG_Y_S .req s3
	%for column in range(NUM_COLUMNS):
		ACC_REG_X_${column}_V .req v${4 + 2*(NUM_PRELOAD + column)}
		ACC_REG_X_${column}_S .req s${4 + 2*(NUM_PRELOAD + column)}
		ACC_REG_Y_${column}_V .req v${5 + 2*(NUM_PRELOAD + column)}
		ACC_REG_Y_${column}_S .req s${5 + 2*(NUM_PRELOAD + column)}
	%endfor
	%for pl in range(NUM_PRELOAD):
		A_REG_X_${pl}_V .req v${4 + 2*pl}
		A_REG_X_${pl}_Q .req q${4 + 2*pl}
		A_REG_X_${pl}_S .req s${4 + 2*pl}
		A_REG_Y_${pl}_V .req v${5 + 2*pl}
		A_REG_Y_${pl}_Q .req q${5 + 2*pl}
		A_REG_Y_${pl}_S .req s${5 + 2*pl}
	%endfor

	<% func_name = "cgemv_" + t + "_neon_kernel" %>
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

	%if target_os in ('windows', 'windows_arm64ec'):
	// We are using x6 as scratch for now before moving Y to x6
	// move alpha from GPR to v0
	lsr x6, x2, #0x20
	fmov s0, w2
	fmov s1, w6
	// move beta from GPR to v1
	lsr x6, x7, #0x20
	fmov s2, w7
	fmov s3, w6
	// load Y pointer from sp
	ldr Y, [sp, #160]
	%endif
	// Setup alpha & beta
	mov v0.s[1], v1.s[0]
	mov v1.s[0], v2.s[0]
	mov v1.s[1], v3.s[0]

	asr N_ROUND_DOWN, N, #3
	lsl N_ROUND_DOWN, N_ROUND_DOWN, #5
	add N_ROUND_DOWN, N_ROUND_DOWN, Y

	mov TMP1, #${NUM_COLUMNS-1}
	// Setup inter-column-step and LDA in bytes. Since INTER_COLUMN_STEP_TAIL aliases LDA, set it up as TMP2 first.
	sub TMP2, LDA, M
	lsl TMP2, TMP2, #3
	sub INTER_COLUMN_STEP, LDA, M
	lsl INTER_COLUMN_STEP, INTER_COLUMN_STEP, #3
	lsl LDA, LDA, #3 // 8b per float
	madd INTER_COLUMN_STEP, LDA, TMP1, INTER_COLUMN_STEP
	mov TMP1, #${NUM_COLUMNS}

	// Init N_ROUND_DOWN and M_ROUND_DOWN values
	udiv N_ROUND_DOWN, N, TMP1
	mul N_ROUND_DOWN, N_ROUND_DOWN, TMP1
	lsl N, N, #3
	lsl N_ROUND_DOWN, N_ROUND_DOWN, #3
	add N, N, Y
	add N_ROUND_DOWN, N_ROUND_DOWN, Y
	asr M_ROUND_DOWN_8, M, #5
	lsl M_ROUND_DOWN_8, M_ROUND_DOWN_8, #8
	asr M_ROUND_DOWN_4, M, #4
	lsl M_ROUND_DOWN_4, M_ROUND_DOWN_4, #7
	asr M_ROUND_DOWN_2, M, #3
	lsl M_ROUND_DOWN_2, M_ROUND_DOWN_2, #6
	asr M_ROUND_DOWN_1, M, #2
	lsl M_ROUND_DOWN_1, M_ROUND_DOWN_1, #5

	add M_ROUND_DOWN_8, M_ROUND_DOWN_8, X // Offset to X, to use as a CMP op
	add M_ROUND_DOWN_4, M_ROUND_DOWN_4, X
	add M_ROUND_DOWN_2, M_ROUND_DOWN_2, X
	add M_ROUND_DOWN_1, M_ROUND_DOWN_1, X

	// Init A_ADDR_X
	%for column in range(1, NUM_COLUMNS):
		mov TMP1, ${column}
		madd A_ADDR_${column}, LDA, TMP1, A
	%endfor

	// Actually init INTER_COLUMN_STEP_TAIL.
	mov INTER_COLUMN_STEP_TAIL, TMP2

	cmp M, 200
	bgt .L_${t}_size_big

	## Big kernels we prefetch, small we don't
	%for size in ["small", "big"]:
		.L_${t}_size_${size}:

		lsl M, M, #3
		add M, M, X


		## Generate one loop like normal, and another to clean up the tail columns
		%for is_tail, tail in [(False, ''), (True, 'tail')]:
			<%
				NUM_COLUMNS = 1 if is_tail else ORIGINAL_NUM_COLUMNS
				CURR_INTER_COLUMN_STEP = 'INTER_COLUMN_STEP_TAIL' if is_tail else 'INTER_COLUMN_STEP'
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
			<%def name="loop_unrolled(level, limit, size, is_tail, tail, t)">
				b .L_${t}_column_loop_unrlvl${level}_cond_${size}_${tail}
				.L_${t}_column_loop_unrlvl${level}_${size}_${tail}:
				## Keep track of the current reg we're using / preloading
				<% curr_a_reg = 0 %>
				%for l in range(level):
					ld2 { X_REG_X_V.4s, X_REG_Y_V.4s }, [X_ADDR], #32
					## Since we're preloading registers, initially load them in if this is the
					## first unroll
					%if is_tail:
						ld2 { A_REG_X_0_V.4s, A_REG_Y_0_V.4s }, [A_ADDR_0], #32
						${fmlx0} ACC_REG_X_0_V.4s, A_REG_X_0_V.4s, X_REG_X_V.4s
						${fmlx1} ACC_REG_X_0_V.4s, A_REG_Y_0_V.4s, X_REG_Y_V.4s
						${fmlx2} ACC_REG_Y_0_V.4s, A_REG_X_0_V.4s, X_REG_Y_V.4s
						${fmlx3} ACC_REG_Y_0_V.4s, A_REG_Y_0_V.4s, X_REG_X_V.4s
					%else:
						%if l == 0:
							%for pl in range(NUM_PRELOAD):
								ld2 { A_REG_X_${pl}_V.4s, A_REG_Y_${pl}_V.4s }, [A_ADDR_${pl}], #32
							%endfor
						%endif
						%for column in range(NUM_COLUMNS):
							<% preloading_next = (column >= NUM_COLUMNS - NUM_PRELOAD) %>
							<% curr_preload = (column+NUM_PRELOAD) if not preloading_next else ((column+NUM_PRELOAD)%NUM_COLUMNS) %>
							## Prefetch if this is a multiple of 4
							%if l %2 == 0 and size == "big":
								prfm pldl1keep, [A_ADDR_${column}, #128]
							%endif
							${fmlx0} ACC_REG_X_${column}_V.4s, A_REG_X_${curr_a_reg}_V.4s, X_REG_X_V.4s
							${fmlx1} ACC_REG_X_${column}_V.4s, A_REG_Y_${curr_a_reg}_V.4s, X_REG_Y_V.4s
							${fmlx2} ACC_REG_Y_${column}_V.4s, A_REG_X_${curr_a_reg}_V.4s, X_REG_Y_V.4s
							${fmlx3} ACC_REG_Y_${column}_V.4s, A_REG_Y_${curr_a_reg}_V.4s, X_REG_X_V.4s
							## Preload the next register - use different instructions if we're at the end
							## of the current mini-unroll
							## Don't preload if we're at the end of the unrolled section
							%if l < level-1:
								ld2 { A_REG_X_${curr_a_reg}_V.4s, A_REG_Y_${curr_a_reg}_V.4s }, [A_ADDR_${curr_preload}], #32
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
				movi ACC_REG_X_${column}_V.4s, #0
				movi ACC_REG_Y_${column}_V.4s, #0
			%endfor

			// Use a progressively less unrolled loop
			${loop_unrolled(4, "M_ROUND_DOWN_4", size, is_tail, tail, t)}
			${loop_unrolled(2, "M_ROUND_DOWN_2", size, is_tail, tail, t)}
			${loop_unrolled(1, "M_ROUND_DOWN_1", size, is_tail, tail, t)}

			// Finish the loop with scalar instructions
			b .L_${t}_column_loop_tail_cond_${size}_${tail}
			.L_${t}_column_loop_tail_${size}_${tail}:
			ldr X_REG_X_S, [X_ADDR]
			ldr X_REG_Y_S, [X_ADDR, #4]
			%for column in range(NUM_COLUMNS):
				ldr A_REG_X_${column}_S, [A_ADDR_${column}]
				ldr A_REG_Y_${column}_S, [A_ADDR_${column}, #4]
				${fmlx0} ACC_REG_X_${column}_V.4s, A_REG_X_${column}_V.4s, X_REG_X_V.4s
				${fmlx1} ACC_REG_X_${column}_V.4s, A_REG_Y_${column}_V.4s, X_REG_Y_V.4s
				${fmlx2} ACC_REG_Y_${column}_V.4s, A_REG_X_${column}_V.4s, X_REG_Y_V.4s
				${fmlx3} ACC_REG_Y_${column}_V.4s, A_REG_Y_${column}_V.4s, X_REG_X_V.4s
				add A_ADDR_${column}, A_ADDR_${column}, #8
			%endfor
			add X_ADDR, X_ADDR, #8
			.L_${t}_column_loop_tail_cond_${size}_${tail}:
			cmp X_ADDR, M
			blt .L_${t}_column_loop_tail_${size}_${tail}

			// Prefetch the next loop
			%for column in range(NUM_COLUMNS):
				prfm pldl1keep, [A_ADDR_${column}, ${CURR_INTER_COLUMN_STEP}]
			%endfor

			// Sum across accumulator registers
			%for column in range (NUM_COLUMNS):
				faddp ACC_REG_X_${column}_V.4s, ACC_REG_X_${column}_V.4s, ACC_REG_X_${column}_V.4s
				faddp ACC_REG_X_${column}_V.2s, ACC_REG_X_${column}_V.2s, ACC_REG_X_${column}_V.2s
				faddp ACC_REG_Y_${column}_V.4s, ACC_REG_Y_${column}_V.4s, ACC_REG_Y_${column}_V.4s
				faddp ACC_REG_Y_${column}_V.2s, ACC_REG_Y_${column}_V.2s, ACC_REG_Y_${column}_V.2s
				// Store in Y
				ldr X_REG_X_S, [Y, #${8 * column}]
				ldr X_REG_Y_S, [Y, #${8 * column + 4}]
				fmla X_REG_X_V.2s, ACC_REG_X_${column}_V.2s, ALPHA.s[0]
				fmls X_REG_X_V.2s, ACC_REG_Y_${column}_V.2s, ALPHA.s[1]
				fmla X_REG_Y_V.2s, ACC_REG_X_${column}_V.2s, ALPHA.s[1]
				fmla X_REG_Y_V.2s, ACC_REG_Y_${column}_V.2s, ALPHA.s[0]
				str X_REG_X_S, [Y, #${8 * column}]
				str X_REG_Y_S, [Y, #${8 * column + 4}]
				## Increment A_ADDR_X
				%if is_tail:
					add A_ADDR_${column}, A_ADDR_${column}, INTER_COLUMN_STEP_TAIL
				%else:
					add A_ADDR_${column}, A_ADDR_${column}, INTER_COLUMN_STEP
				%endif
			%endfor
			add Y, Y, #${8 * NUM_COLUMNS}
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

${cgemv('t')}
${cgemv('c')}
