## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

<% import math %>
## Regarding loop / column unrolling, anything above 16 hasn't been tested
## How much to loop unroll
<% UNROLL_PROGRESSION = [12, 8, 4, 2, 1] %>
## How many columns to process in one loop
<% NUM_COLUMN_PROGRESSION = [12, 8, 4, 2, 1] %>
## How many registers to dedicate to preloading.
## OPTIMIZE: This uses a simpler preloading scheme than in gemv_t, where we
## don't preload across loops, as we load and store from [Y, ROW] so ROW can't
## change (which it needs to, in order to preload). The solution would be to
## either implement a preloading scheme like in the Neon version (where we
## actually increment the A_ADDR values, a little more complex but allows for
## immediate addressing), or setup 2 versions of the ROW register and somehow
## alternate between them (?).
<% NUM_PRELOAD = 8 %>

<% MAX_NUM_UNROLLS = max(UNROLL_PROGRESSION) %>
<% MAX_NUM_COLUMNS = max(NUM_COLUMN_PROGRESSION) %>
<% MAX_PRELOAD = NUM_PRELOAD %>

##<% assert min(NUM_COLUMN_PROGRESSION) == 1 %>
<% assert min(UNROLL_PROGRESSION) == 1 %>

M        .req x0
N        .req x1
A        .req x2
LDA      .req x3
X        .req x4
INC_X    .req x5
Y        .req x6
ROW      .req x8  // Row of the matrix
COLUMN   .req x9  // Column of the matrix (offset into the row)

ORIG_LDA .req x10

ISCRATCH_0 .req x2
ISCRATCH_1 .req x7

VEC_SIZE .req x11

ACC_REG_z .req z4
ACC_REG_v .req v4
ACC_REG_s .req s4
ACC_REG_d .req d4
ACC_REG_q .req q4

## Avoid using x18 or x29 when allocating general-purpose registers
## so that our code remains portable
<% gpreg = 11 %>
// .req the matrix address registers
%for c in range(MAX_NUM_COLUMNS):
	<% gpreg += 1 %>
	%if gpreg == 18 or gpreg == 29:
		<% gpreg += 1 %>
	%endif
	A_ADDR_${c} .req x${gpreg}  // The address of the current column, to be used for loads
%endfor

// .req the X value registers
%for c in range(MAX_NUM_COLUMNS):
    X_REG_${c}_z .req z${5 + c}
    X_REG_${c}_v .req v${5 + c}
    X_REG_${c}_s .req s${5 + c}
    X_REG_${c}_d .req d${5 + c}
    X_REG_${c}_q .req q${5 + c}
%endfor

// .req the preload registers
%for pr in range(NUM_PRELOAD):
    A_REG_${pr}_z .req z${5 + MAX_NUM_COLUMNS + pr}
    A_REG_${pr}_v .req v${5 + MAX_NUM_COLUMNS + pr}
    A_REG_${pr}_s .req s${5 + MAX_NUM_COLUMNS + pr}
    A_REG_${pr}_d .req d${5 + MAX_NUM_COLUMNS + pr}
    A_REG_${pr}_q .req q${5 + MAX_NUM_COLUMNS + pr}
%endfor

<%def name="gemv(p, is_complex)">

## Size of element
<%
if p == 's':
    shiftsize = 2
    esize = 8 if is_complex else 4
    func_name = 'cgemv_n_sve_kernel' if is_complex else 'sgemv_n_sve_kernel'
else:
    shiftsize = 3
    esize = 16 if is_complex else 8
    func_name = 'zgemv_n_sve_kernel' if is_complex else 'dgemv_n_sve_kernel'
%>

## Choose optimal loop unroll and columns for different datatypes
<% UNROLL_PROGRESSION = [12, 8, 4, 2, 1] if is_complex else [4, 2, 1] %>
<% NUM_COLUMN_PROGRESSION = [1] if is_complex else [12, 8, 4, 2, 1] %>

## Load / store insrtuctions
<% ld1z = 'ld1w' if p == 's' else 'ld1d' %>
<% ld1rz = 'ld1rw' if esize == 4 else ('ld1rd' if esize == 8 else 'ld1rqd') %>
<% st1z = 'st1w' if p == 's' else 'st1d' %>
<% incz = 'incw' if p == 's' else 'incd' %>

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

    // Init A_ADDR values
    %for c in range(MAX_NUM_COLUMNS):
        mov ISCRATCH_1, ${c * esize}
        madd A_ADDR_${c}, LDA, ISCRATCH_1, A
    %endfor

    ptrue p1.${p}

    %if is_complex:
        lsl M, M, #1
    %endif

    mov ORIG_LDA, LDA
    mov COLUMN, #0
    <% orig_func_name = func_name %>
    %for NUM_COLUMNS in NUM_COLUMN_PROGRESSION:
        <% func_name = orig_func_name + "_column_" + str(NUM_COLUMNS) %>

        mov ISCRATCH_1, #${NUM_COLUMNS}
        mul LDA, ORIG_LDA, ISCRATCH_1

        b .L${func_name}_column_loop_cond
        .L${func_name}_column_loop:

        mov ROW, #0

        // Load elements from X (equal to the number of columns we're
        // processing right now)
        %for c in range(NUM_COLUMNS):
            %if c * esize > 112 and ld1rz == 'ld1rqd':
                ## If we're here, we can't use immediate offsetting since it is
                ## out of range for ld1rqd.
                ## Unfortunately ld1r* REQUIRES lsl #3, hence why we're
                ## figuring out the addressing a bit differently (we actually
                ## need lsl 4, hence the c * 2)
                mov ISCRATCH_1, #${c * 2}
                ld1rqd {X_REG_${c}_z.d}, p1/z, [X, ISCRATCH_1, lsl #3]
            %else:
                %if is_complex:
                    ${ld1rz} {X_REG_${c}_z.d}, p1/z, [X, #${c * esize}]
                %else:
                    ${ld1rz} {X_REG_${c}_z.${p}}, p1/z, [X, #${c * esize}]
                %endif
            %endif
        %endfor

        ptrue p0.${p}

        <% orig_func_name_2 = func_name %>
        %for NUM_UNROLLS in UNROLL_PROGRESSION:
            <% func_name = orig_func_name_2 + "_unroll_" + str(NUM_UNROLLS) %>

            b .L${func_name}_row_loop_cond
            .L${func_name}_row_loop:
            %for level in range(NUM_UNROLLS):
                ${ld1z} ACC_REG_z.${p}, p0/z, [Y, ROW, lsl #${shiftsize}]           // Load Y
                ## Initially preload as much as we can
                <% NUM_PRELOAD = min(MAX_PRELOAD, NUM_COLUMNS) %>
                %for pr in range(NUM_PRELOAD):
                    ${ld1z} A_REG_${pr}_z.${p}, p0/z, [A_ADDR_${pr}, ROW, lsl #${shiftsize}] // Load A
                %endfor
                %for c in range(NUM_COLUMNS):
                    <% pr = c % MAX_PRELOAD %>
                    %if is_complex:
                        fcmla ACC_REG_z.${p}, p0/m, A_REG_${pr}_z.${p}, X_REG_${c}_z.${p}, #0
                        fcmla ACC_REG_z.${p}, p0/m, A_REG_${pr}_z.${p}, X_REG_${c}_z.${p}, #90
                    %else:
                        fmla ACC_REG_z.${p}, p0/m, A_REG_${pr}_z.${p}, X_REG_${c}_z.${p}
                    %endif
                    %if c + MAX_PRELOAD < NUM_COLUMNS:
                        ${ld1z} A_REG_${pr}_z.${p}, p0/z, [A_ADDR_${c + MAX_PRELOAD}, ROW, lsl #${shiftsize}]
                    %endif
                %endfor
                ${st1z} {ACC_REG_z.${p}}, p0, [Y, ROW, lsl #${shiftsize}]           // Store Y
                ${incz} ROW
            %endfor
            .L${func_name}_row_loop_cond:
            %if NUM_UNROLLS > 1:
                sub ISCRATCH_0, M, ROW
                mov ISCRATCH_1, ${NUM_UNROLLS}
                mul ISCRATCH_1, VEC_SIZE, ISCRATCH_1
                cmp ISCRATCH_0, ISCRATCH_1
                bge .L${func_name}_row_loop
            %else:
                whilelt p0.${p}, ROW, M
                b.first .L${func_name}_row_loop
            %endif

        %endfor
        <% func_name = orig_func_name_2 %>

        // Move column
        add COLUMN, COLUMN, #${NUM_COLUMNS}
        %for c in range(NUM_COLUMNS):
            <% add_lsl = shiftsize + 1 if is_complex else shiftsize %>
            add A_ADDR_${c}, A_ADDR_${c}, LDA, lsl #${add_lsl}
        %endfor
        mov ISCRATCH_1, #${NUM_COLUMNS}
        mul ISCRATCH_1, ISCRATCH_1, INC_X
        add X, X, ISCRATCH_1, lsl #${add_lsl}

        .L${func_name}_column_loop_cond:
        sub ISCRATCH_0, N, COLUMN
        cmp ISCRATCH_0, #${NUM_COLUMNS}
        bge   .L${func_name}_column_loop

    %endfor
    <% func_name = orig_func_name %>

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
    ${epilogue(func_name)}

</%def>

${gemv("s", False)}
${gemv("d", False)}
${gemv("s", True)}
${gemv("d", True)}
