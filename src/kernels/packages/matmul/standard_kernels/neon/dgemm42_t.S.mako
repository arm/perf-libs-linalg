## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

/**
  For use in other functions, e.g. TRSM - just include the file & call the mako function (dgemm42_t).

  Performs a 4 x k x 2 transposed DGEMM into 2x2 C registers.

  Clobbers all vector registers >= 8, gemms into 8 C registers which should be
  reqed as C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7, with the result in the bottom
  part of each C reg, organized like
  0 1
  2 3
  4 5
  6 7.
  @param A - pointer to A (const)
  @param B - pointer to B (const)
  @param A_STRIDE - lda (const)
  @param B_STRIDE - ldb (const)
  @param K - K dimension (const)
  @param fmla - The fmla instruction to use (can be changed to 'fmls')
 */
<%def name="dgemm42_t(func_name, A, B, A_STRIDE, B_STRIDE, K, fmla='fmla')">

// A single unroll uses a full register block
<% unroll_progression = [4, 2, 1] %>
<% max_unroll = max(unroll_progression) %>
<% assert max_unroll == 4 %>


<%
## Size of register blocks in register
A_BLOCK_W = 1
A_BLOCK_H = 4
B_BLOCK_W = 2
B_BLOCK_H = 1
NUM_A_REG = A_BLOCK_W*A_BLOCK_H*4
NUM_B_REG = B_BLOCK_W*B_BLOCK_H*4
%>

<% func_name += '_dgemm42_t' %>

// A block goes top to bottom, left to right
// B block goes left to right, top to bottom
%for ii in range(NUM_A_REG):
  A_${ii} .req v${8+ii}
  A_${ii}_q .req q${8+ii}
%endfor
%for ii in range(NUM_B_REG):
  B_${ii} .req v${8+ii+NUM_A_REG}
  B_${ii}_q .req q${8+ii+NUM_A_REG}
%endfor

  DG42_A_PTR_0 .req ISCRATCH_1
  DG42_A_PTR_1 .req ISCRATCH_2
  DG42_A_PTR_2 .req ISCRATCH_3
  DG42_A_PTR_3 .req ISCRATCH_4
  DG42_B_PTR_0 .req ISCRATCH_5
  DG42_B_PTR_1 .req ISCRATCH_6

  DG42_K .req ISCRATCH_11

  mov DG42_K, ${K}

  mov DG42_A_PTR_0, ${A}
  mov DG42_B_PTR_0, ${B}

  // Setup all pointers
  add DG42_A_PTR_1, DG42_A_PTR_0, ${A_STRIDE}
  add DG42_A_PTR_2, DG42_A_PTR_0, ${A_STRIDE}, lsl #1
  add DG42_A_PTR_3, DG42_A_PTR_1, ${A_STRIDE}, lsl #1
  add DG42_B_PTR_1, DG42_B_PTR_0, ${B_STRIDE}

  <% orig_func_name = func_name %>

  %for unroll in unroll_progression:
    <% func_name += "_unroll" + str(unroll) %>

    cmp DG42_K, #${unroll*2}
    blt .L${func_name}_loop_skip

    // Preload A and B blocks, we can preload max 4 across since max unroll is 4
    %for ii in range(A_BLOCK_H):
      %for jj in range(unroll):
        ldr A_${ii+4*jj}_q, [DG42_A_PTR_${ii}], #16
      %endfor
    %endfor
    %for ii in range(B_BLOCK_W):
      %for jj in range(unroll):
        ldr B_${ii+2*jj}_q, [DG42_B_PTR_${ii}], #16
      %endfor
    %endfor

    <% orig_func_name_2 = func_name %>
    %for do_preload in [True, False]:
      <% func_name += "_preload" + str(do_preload) %>
      b .L${func_name}_loop_cond
      .L${func_name}_loop:

      // Do actual unrolled GEMM. Each unroll does a row of the B registers & a column of a registers
      %for ii in range(unroll):
        <% A_0 = "A_" + str(ii*4 + 0) %>\
        <% A_1 = "A_" + str(ii*4 + 1) %>\
        <% A_2 = "A_" + str(ii*4 + 2) %>\
        <% A_3 = "A_" + str(ii*4 + 3) %>\
        <% B_0 = "B_" + str(ii*2)     %>\
        <% B_1 = "B_" + str(ii*2 + 1)   %>\

        // Do a 2x2x1 reg block gemm (6 registers), into the C registers,
        // expecting them to be horizontally added later.
        ${fmla} C_0.2d, ${A_0}.2d, ${B_0}.2d
        ${fmla} C_1.2d, ${A_0}.2d, ${B_1}.2d
        %if do_preload:
          ldr ${A_0}_q, [DG42_A_PTR_0], #16
        %endif
        ${fmla} C_2.2d, ${A_1}.2d, ${B_0}.2d
        ${fmla} C_3.2d, ${A_1}.2d, ${B_1}.2d
        %if do_preload:
          ldr ${A_1}_q, [DG42_A_PTR_1], #16
        %endif
        ${fmla} C_4.2d, ${A_2}.2d, ${B_0}.2d
        ${fmla} C_5.2d, ${A_2}.2d, ${B_1}.2d
        %if do_preload:
          ldr ${A_2}_q, [DG42_A_PTR_2], #16
        %endif
        ${fmla} C_6.2d, ${A_3}.2d, ${B_0}.2d
        // Preload B_0
        %if do_preload:
          ldr ${B_0}_q, [DG42_B_PTR_0], #16
        %endif
        ${fmla} C_7.2d, ${A_3}.2d, ${B_1}.2d
        %if do_preload:
          // Preload B_1
          ldr ${B_1}_q, [DG42_B_PTR_1], #16
          ldr ${A_3}_q, [DG42_A_PTR_3], #16
        %endif
      %endfor

      sub DG42_K, DG42_K, #${unroll*2}
      .L${func_name}_loop_cond:
      cmp DG42_K, #${unroll*(4 if do_preload else 2)}
      bge .L${func_name}_loop
      <% func_name = orig_func_name_2 %>
    %endfor // do_preload

    .L${func_name}_loop_skip:

    <% func_name = orig_func_name %>
  %endfor

  // Cleanup loop
  b .L${func_name}_cleanup_loop_cond
  .L${func_name}_cleanup_loop:
  ldr d8, [DG42_A_PTR_0], #8
  ldr d9, [DG42_A_PTR_1], #8
  ldr d10, [DG42_A_PTR_2], #8
  ldr d11, [DG42_A_PTR_3], #8
  ldr d12, [DG42_B_PTR_0], #8
  ldr d13, [DG42_B_PTR_1], #8
  ${fmla} C_0.2d, v8.2d,  v12.2d
  ${fmla} C_1.2d, v8.2d,  v13.2d
  ${fmla} C_2.2d, v9.2d,  v12.2d
  ${fmla} C_3.2d, v9.2d,  v13.2d
  ${fmla} C_4.2d, v10.2d, v12.2d
  ${fmla} C_5.2d, v10.2d, v13.2d
  ${fmla} C_6.2d, v11.2d, v12.2d
  ${fmla} C_7.2d, v11.2d, v13.2d
  sub DG42_K, DG42_K, #1
  .L${func_name}_cleanup_loop_cond:
  cmp DG42_K, #1
  bge .L${func_name}_cleanup_loop

  %for ii in range(8):
    faddp C_${ii}_d, C_${ii}.2d
  %endfor

%for ii in range(NUM_A_REG):
  %if ii == NUM_A_REG-1:
    .unreq A_${ii}
    .unreq A_${ii}_q
  %else:
    .unreq A_${ii}
    .unreq A_${ii}_q
  %endif
%endfor
%for ii in range(NUM_B_REG):
  .unreq B_${ii}
  .unreq B_${ii}_q
%endfor
  .unreq DG42_A_PTR_0
  .unreq DG42_A_PTR_1
  .unreq DG42_A_PTR_2
  .unreq DG42_A_PTR_3
  .unreq DG42_B_PTR_0
  .unreq DG42_B_PTR_1

</%def>