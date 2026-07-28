## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

## # Brief preface
##
## This file generates a GEMV for all the input types - s, d, c, and z.
## Since this is the transposed case, we dot the vector X with each column of
## the matrix A, storing each result in each element of Y. X is of size M, and
## Y is of the N.
##
## # Processing multiple columns per loop iter
## Since we need to keep iterating over X, we lose some performance since we
## need to load from X a lot. To reduce this, we can process multiple columns at
## once - where C is the number of columns being processed: load 1 vector from X,
## load C vectors from A, multiply them together & accumulate in separate
## registers, then increment X and repeat. By the time we reach the bottom of the
## matrix, we've performed C dot products.
##
## Unfortunately, this means we need a loop 'tail' for the right hand side of
## the matrix - let's say we process 8 columns at once, unless N is a multiple of
## 8 we can't process everything. This is what the NUM_COLUMN_PROGRESSION mako
## variable is - First we process NUM_COLUMN_PROGRESSION[0] columns, then
## NUM_COLUMN_PROGRESSION[1] columns, and so on. This is all done via a mako
## for loop. 1 must be present in the list, and must be the smallest item in
## the list.
##
## # Register preloading
## Registers are preloaded. The amount of registers used is stored in
## NUM_PRELOAD - the correct instructions are then generated with a bunch of messy
## modulos in the mako template. This is complicated slightly by SVE's predicate
## system - if we want to predicate a load, we can't also predicate the fmlas with
## the same predicate until the fmlas 'catch up' with the loads. 2 predicates are
## therefore used - LD_PRED for loads, and CP_PRED for computation.

## How much to loop unroll
<% UNROLL_PROGRESSION = [1] %>
## How many columns to process in one loop
<% NUM_COLUMN_PROGRESSION = [12, 8, 4, 1] %>
## How many registers to dedicate to preloading. For each element in
## NUM_COLUMN_PROGRESSION, the element must be evenly divisible by NUM_PRELOAD,
## or NUM_PRELOAD should be evenly divisible by the element. (This is checked
## via an assertion). This is because when the loop rolls around, we can't
## statically generate enough of a loop to properly account for the remainder -
## we'd need to change the registers needed at runtime, incurring overhead.
<% NUM_PRELOAD = 4 %>

<% assert all(map(lambda x : x % NUM_PRELOAD == 0 or NUM_PRELOAD % x == 0, NUM_COLUMN_PROGRESSION)) %>

<% MAX_NUM_UNROLLS = max(UNROLL_PROGRESSION) %>
<% MAX_NUM_COLUMNS = max(NUM_COLUMN_PROGRESSION) %>
<% MAX_PRELOAD = NUM_PRELOAD %>

<% assert min(NUM_COLUMN_PROGRESSION) == 1 %>
<% assert min(UNROLL_PROGRESSION) == 1 %>

M        .req x0
N        .req x1
A        .req x2
LDA      .req x3
X        .req x4
Y        .req x6
INC_Y    .req x7
ROW      .req x8  // Row of the matrix
COLUMN   .req x9  // Column of the matrix (offset into the row)
A_ADDR   .req x2  // The address of the current row, to be used for loads
ORIG_LDA .req x10

// Req scratch registers
SCRATCH_0_z .req z1
SCRATCH_0_v .req v1
SCRATCH_0_s .req s1
SCRATCH_0_d .req d1
SCRATCH_0_q .req q1
ISCRATCH_0 .req x5
ISCRATCH_1 .req x12

LD_PRED .req p5
CP_PRED .req p0

VEC_SIZE .req x11 // Stores the size of the vector for this problem size

## Avoid using x18 or x29 when allocating general-purpose registers
## so that our code remains portable
<% gpreg = 12 %>
// Req the matrix addresses
% for r in range(MAX_NUM_COLUMNS):
	<% gpreg += 1 %>
	%if gpreg == 18 or gpreg == 29:
		<% gpreg += 1 %>
	%endif
	A_ADDR_${r} .req x${gpreg}
% endfor

// Req the preloading A registers
% for pr in range(NUM_PRELOAD):
A_REG_${pr}_z .req z${6 + pr}
A_REG_${pr}_v .req v${6 + pr}
A_REG_${pr}_s .req s${6 + pr}
A_REG_${pr}_d .req d${6 + pr}
A_REG_${pr}_q .req q${6 + pr}
% endfor

// Req the accumulator registers
% for r in range(MAX_NUM_COLUMNS):
ACC_REG_${r}_z .req z${6 + NUM_PRELOAD + r}
ACC_REG_${r}_v .req v${6 + NUM_PRELOAD + r}
ACC_REG_${r}_s .req s${6 + NUM_PRELOAD + r}
ACC_REG_${r}_d .req d${6 + NUM_PRELOAD + r}
ACC_REG_${r}_q .req q${6 + NUM_PRELOAD + r}
% endfor

<%def name="gemv(p, is_complex, conjugate=False)">

ALPHA_z .req z0
ALPHA_v .req v0
ALPHA_s .req s0
ALPHA_d .req d0
ALPHA_q .req q0

## neonldp is the SIMD & FP register used for the given type, mainly used for
## ldrs.
<%
if not is_complex: neonldp = p
else: neonldp = 'd' if p == 's' else 'q'
%>

## Load / store insrtuctions
<% ld1z = 'ld1w' if p == 's' else 'ld1d' %>
<% st1z = 'st1w' if p == 's' else 'st1d' %>
<% incz = 'incw' if p == 's' else 'incd' %>
<% ## Get sizes of elements & the name of the function
t_type = 'c' if conjugate else 't'
if p == 's':
  shiftsize = 2
  esize = 4 if not is_complex else 8 ## Size of element
  func_name = 'cgemv_'+t_type+'_sve_kernel' if is_complex else 'sgemv_t_sve_kernel'
else:
  shiftsize = 3
  esize = 8 if not is_complex else 16
  func_name = 'zgemv_'+t_type+'_sve_kernel' if is_complex else 'dgemv_t_sve_kernel'
%>

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

  mov VEC_SIZE, #0
  ${incz} VEC_SIZE

  // Init matrix addresses
% for r in range(MAX_NUM_COLUMNS):
  mov ISCRATCH_0, #${r * esize}
  madd A_ADDR_${r}, LDA, ISCRATCH_0, A
% endfor

  ptrue p1.${p}

% if is_complex:
  pfalse p2.b
  // Used later for complex faddv
  zip1 p3.${p}, p1.${p}, p2.${p} // Real
  zip1 p4.${p}, p2.${p}, p1.${p} // Imag

  // Init ALPHA
  dup z0.${p}, z0.${p}[0]
  dup z1.${p}, z1.${p}[0]
  zip1 ALPHA_z.${p}, z0.${p}, z1.${p}
% endif

  mov ROW, #0
% if is_complex:
  lsl M, M, #1
% endif

## Save old LDA value to calculate the new LDA value on future loop
## iterations
  mov ORIG_LDA, LDA

## Original_func_name 0 and 1 are for restoring the label prefix after running
## with some num columns (in the case of original_func_name_0) or after running
## with some unroll depth (in the case of original_func_name_0). Think of it like
## a name stack.
## TODO: Implement an explicit name stack in Mako.
<% original_func_name_0 = func_name %>
% for NUM_COLUMNS in NUM_COLUMN_PROGRESSION:
    <% func_name = original_func_name_0 + "_columns_" + str(NUM_COLUMNS) %>


    ## Set up the correct LDA for the A_ADDR_X registers
    mov ISCRATCH_0, ${NUM_COLUMNS}
    mul LDA, ORIG_LDA, ISCRATCH_0

    b .L${func_name}_row_loop_cond
  .L${func_name}_row_loop:

    // Init accumulators
  % for c in range(NUM_COLUMNS):
    dup ACC_REG_${c}_z.${p}, #0
  % endfor

    mov COLUMN, #0

  whilelt LD_PRED.${p}, COLUMN, M
  whilelt CP_PRED.${p}, COLUMN, M

    ## Unroll the loop
    <% original_func_name_1 = func_name %>
  % for NUM_UNROLLS in UNROLL_PROGRESSION:
      <% func_name = original_func_name_1 + "_unroll_" + str(NUM_UNROLLS) %>

	<% NUM_PRELOAD = min(MAX_PRELOAD, NUM_COLUMNS) %>
      // Preload all A_REG registers with matrix values initially
    <% INITIAL_PRELOAD_SIZE = min([NUM_PRELOAD, NUM_COLUMNS * NUM_UNROLLS]) %>
    % for ii in range(INITIAL_PRELOAD_SIZE):
      ${ld1z} A_REG_${ii}_z.${p}, LD_PRED/z, [A_ADDR_${ii}, COLUMN, lsl #${shiftsize}]
    % endfor

    ${ld1z} z3.${p}, LD_PRED/z, [X, COLUMN, lsl #${shiftsize}]
      b .L${func_name}_column_loop_cond
    .L${func_name}_column_loop:

      ## 'pr' just counts up each time we do a preload. This is used to calculate
      ## the value of prm.
      <% pr = 0 %>  ## pr = preload
      <% prm = 0 %> ## prm == preload mod
    % for level in range(NUM_UNROLLS):
        // Load from the vector, mul by the matrix registers (A_REG_*) and
        // accumulate in z2
      % for c in range(NUM_COLUMNS):
        % if is_complex:
          % if not conjugate:
			  fcmla ACC_REG_${c}_z.${p}, CP_PRED/m, A_REG_${prm}_z.${p}, z3.${p}, #0  // real
			  fcmla ACC_REG_${c}_z.${p}, CP_PRED/m, A_REG_${prm}_z.${p}, z3.${p}, #90 // imag
          % else:
			  fcmla ACC_REG_${c}_z.${p}, CP_PRED/m, A_REG_${prm}_z.${p}, z3.${p}, #0  // real
			  fcmla ACC_REG_${c}_z.${p}, CP_PRED/m, A_REG_${prm}_z.${p}, z3.${p}, #270 // imag
          % endif
        % else:
			fmla ACC_REG_${c}_z.${p}, CP_PRED/m, z3.${p}, A_REG_${prm}_z.${p}
        % endif
        % if (pr + INITIAL_PRELOAD_SIZE) % NUM_COLUMNS == 0:
			## Increment the column if there's no more to preload for this value
			## of COLUMN
			${incz} COLUMN
			## Check that we're not at the end of the loop
			whilelt LD_PRED.${p}, COLUMN, M
        % endif
	  % if (pr + INITIAL_PRELOAD_SIZE - NUM_PRELOAD + 1) % NUM_COLUMNS == 0:
		  whilelt CP_PRED.${p}, COLUMN, M
		% if NUM_UNROLLS == 1:
			b.none .L${func_name}_end_column_loop
		% endif
		  ${ld1z} z3.${p}, LD_PRED/z, [X, COLUMN, lsl #${shiftsize}]
	  % endif
        ## Preload the pr reg
        ${ld1z} A_REG_${prm}_z.${p}, LD_PRED/z, [A_ADDR_${(c + NUM_PRELOAD) % NUM_COLUMNS}, COLUMN, lsl #${shiftsize}]
        ## Increment the preload register (pr)
        <% pr += 1 %>
        <% prm = pr % NUM_PRELOAD %>
      % endfor
    % endfor

    .L${func_name}_column_loop_cond:
    % if NUM_UNROLLS > 1:
      sub ISCRATCH_0, M, COLUMN
      mov ISCRATCH_1, ${NUM_UNROLLS}
      mul ISCRATCH_1, VEC_SIZE, ISCRATCH_1
      cmp ISCRATCH_0, ISCRATCH_1
      bge .L${func_name}_column_loop
    % else:
    b .L${func_name}_column_loop
    % endif

  .L${func_name}_end_column_loop:

  % endfor ## % for NUM_UNROLLS in UNROLL_PROGRESSION:

    <% func_name = original_func_name_1 %>

    // Now z2 has a vector full of all the multiplications, we just need to sum
    // across the vector, multiply by alpha, then add to y

  % for c in range(NUM_COLUMNS):
      // Sum across vector
    % if is_complex:
      faddv SCRATCH_0_${p}, p4, ACC_REG_${c}_z.${p} // Imag
      faddv ACC_REG_${c}_${p}, p3, ACC_REG_${c}_z.${p} // Real
      mov ACC_REG_${c}_v.${p}[1], SCRATCH_0_v.${p}[0]
    % else:
      faddv ACC_REG_${c}_${p}, p1, ACC_REG_${c}_z.${p}
    % endif

      // Load & add to Y
      ldr SCRATCH_0_${neonldp}, [Y]
    % if is_complex:
      fcmla SCRATCH_0_z.${p}, p1/m, ACC_REG_${c}_z.${p}, ALPHA_z.${p}, #0  // real
      fcmla SCRATCH_0_z.${p}, p1/m, ACC_REG_${c}_z.${p}, ALPHA_z.${p}, #90 // imag
    % else:
      fmla SCRATCH_0_${p}, ACC_REG_${c}_${p}, ALPHA_v.${p}[0]
    % endif
      str SCRATCH_0_${neonldp}, [Y]

      // Increment Y
    % if is_complex:
      add Y, Y, INC_Y, lsl #${shiftsize+1}
    % else:
      add Y, Y, INC_Y, lsl #${shiftsize}
    % endif

  % endfor

    // Move row ptrs
  % if is_complex:
    % for c in range(NUM_COLUMNS):
      add A_ADDR_${c}, A_ADDR_${c}, LDA, lsl #${shiftsize+1}
    % endfor
  % else:
    % for c in range(NUM_COLUMNS):
      add A_ADDR_${c}, A_ADDR_${c}, LDA, lsl #${shiftsize}
    % endfor
  % endif

    add ROW, ROW, #${NUM_COLUMNS}


  .L${func_name}_row_loop_cond:
    sub ISCRATCH_0, N, ROW
    cmp ISCRATCH_0, #${NUM_COLUMNS}
    bge .L${func_name}_row_loop

% endfor ## % for NUM_COLUMNS in NUM_COLUMN_PROGRESSION:

## Reset func_name (since we changed it for the loop)
<% func_name = original_func_name_0 %>

.L${func_name}_cleanup:
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

.unreq ALPHA_z
.unreq ALPHA_v
.unreq ALPHA_s
.unreq ALPHA_d
.unreq ALPHA_q

  ${epilogue(func_name)}

</%def>

${gemv("s", False)}
${gemv("d", False)}
${gemv("s", True, conjugate=False)}
${gemv("s", True, conjugate=True)}
${gemv("d", True, conjugate=False)}

${gemv("d", True, conjugate=True)}
