## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

/**
  For use in other functions, e.g. TRSM - just include the file & call the mako function (dgemm42_n).

  Performs a 4 x k x 2 DGEMM into 2x2 C registers.

  Clobbers all vector registers >= 4, gemms into 4 C registers which should be reqed as C_0, C_1, C_2, and C_3.

  @param A - pointer to A (const)
  @param B - pointer to B (const)
  @param A_STRIDE - lda (const)
  @param B_STRIDE - ldb (const)
  @param K - K dimension (const)
  @param fmla - the fmla instruction to use. Change to 'fmls' here to use fmls - this is useful in TRSM. default = 'fmla'.
 */
<%def name="dgemm42_n(func_name, A, B, A_STRIDE, B_STRIDE, K, fmla='fmla')">

// A single unroll uses a full register block
<% unroll_progression = [4, 2, 1] %>
<% max_unroll = max(unroll_progression) %>

<%
## Size of register blocks in register
A_BLOCK_W = 2
A_BLOCK_H = 2
B_BLOCK_W = 2
B_BLOCK_H = 1
NUM_A_REG = A_BLOCK_W*A_BLOCK_H*4
NUM_B_REG = B_BLOCK_W*B_BLOCK_H*4
%>

<% func_name += '_dgemm42_n' %>

// Reg blocks go from right to left, top to bottom. Number A reg differently
// under the hood because we LDP those, so they must be contiguous.
%for ii in range(NUM_A_REG):
  %if ii == NUM_A_REG-1:
    A_${ii} .req v${4+ii}
    A_${ii}_q .req q${4+ii}
  %else:
    A_${ii} .req v${4+(ii * 2) % (NUM_A_REG-1)}
    A_${ii}_q .req q${4+(ii * 2) % (NUM_A_REG-1)}
  %endif
%endfor
%for ii in range(NUM_B_REG):
  B_${ii} .req v${4+ii+NUM_A_REG}
  B_${ii}_q .req q${4+ii+NUM_A_REG}
%endfor

  DG42_A_PTR_0 .req ISCRATCH_1
  DG42_A_PTR_1 .req ISCRATCH_2
  DG42_A_PTR_2 .req ISCRATCH_3
  DG42_A_PTR_3 .req ISCRATCH_4
  DG42_A_PTR_4 .req ISCRATCH_5
  DG42_A_PTR_5 .req ISCRATCH_6
  DG42_A_PTR_6 .req ISCRATCH_7
  DG42_A_PTR_7 .req ISCRATCH_8
  DG42_B_PTR_0 .req ISCRATCH_9
  DG42_B_PTR_1 .req ISCRATCH_10

  DG42_K .req ISCRATCH_11

  mov DG42_K, ${K}

  mov DG42_A_PTR_0, ${A}
  mov DG42_B_PTR_0, ${B}

  // Setup all pointers
  add DG42_A_PTR_1, DG42_A_PTR_0, ${A_STRIDE}
  add DG42_A_PTR_2, DG42_A_PTR_0, ${A_STRIDE}, lsl #1
  add DG42_A_PTR_3, DG42_A_PTR_1, ${A_STRIDE}, lsl #1
  add DG42_A_PTR_4, DG42_A_PTR_2, ${A_STRIDE}, lsl #1
  add DG42_A_PTR_5, DG42_A_PTR_3, ${A_STRIDE}, lsl #1
  add DG42_A_PTR_6, DG42_A_PTR_4, ${A_STRIDE}, lsl #1
  add DG42_A_PTR_7, DG42_A_PTR_5, ${A_STRIDE}, lsl #1
  add DG42_B_PTR_1, DG42_B_PTR_0, ${B_STRIDE}

  <% orig_func_name = func_name %>

  %for unroll in unroll_progression:
    <% func_name += "_unroll" + str(unroll) %>

    cmp DG42_K, #${unroll*2}
    blt .L${func_name}_loop_skip

    mov ISCRATCH_0, #${unroll*2}

    // Preload A and B blocks
    %for ii in range(min(A_BLOCK_W*unroll, 8)):
      ldp A_${ii}_q, A_${ii+8}_q, [DG42_A_PTR_${ii}]
      madd DG42_A_PTR_${ii}, ${A_STRIDE}, ISCRATCH_0, DG42_A_PTR_${ii}
    %endfor
    %for ii in range(min(B_BLOCK_H*unroll, 4)):
      ldr B_${2*ii}_q, [DG42_B_PTR_0], #16
      ldr B_${2*ii+1}_q, [DG42_B_PTR_1], #16
    %endfor

    <% orig_func_name_2 = func_name %>
    %for do_preload in [True, False]:
      <% func_name += "_preload" + str(do_preload) %>
      b .L${func_name}_loop_cond
      .L${func_name}_loop:

      // Do actual unrolled GEMM. Each unroll does a row of the B registers - when
      // ii >= max_unroll / 2, we're 'looped' back on the registers, using the overflows
      // - in this case, use B_O_* registers. When preloading, preload the
      // register 'before' your register in the block. Use the get_preload_reg() function.
      %for ii in range(unroll):
        <% A_0   = "A_" + str((ii*2)   % (max_unroll*2))        %>\
        <% A_0_q = "A_" + str((ii*2)   % (max_unroll*2)) + "_q" %>\
        <% A_1   = "A_" + str((ii*2+1) % (max_unroll*2))        %>\
        <% A_1_q = "A_" + str((ii*2+1) % (max_unroll*2)) + "_q" %>\
        <% A_2   = "A_" + str((ii*2)   % (max_unroll*2) + 8)        %>\
        <% A_2_q = "A_" + str((ii*2)   % (max_unroll*2) + 8) + "_q" %>\
        <% A_3   = "A_" + str((ii*2+1) % (max_unroll*2) + 8)        %>\
        <% A_3_q = "A_" + str((ii*2+1) % (max_unroll*2) + 8) + "_q" %>\
        <% B_0 =  "B_" + str((ii*2)    % (max_unroll*2)) %>\
        <% B_1 =  "B_" + str((ii*2+1)  % (max_unroll*2)) %>\

        // Do a 2x2x1 reg block gemm (6 registers).
        ${fmla} C_0.2d, ${A_0}.2d, ${B_0}.d[0]
        ${fmla} C_1.2d, ${A_0}.2d, ${B_1}.d[0]
        ${fmla} C_2.2d, ${A_2}.2d, ${B_0}.d[0]
        ${fmla} C_3.2d, ${A_2}.2d, ${B_1}.d[0]
        %if do_preload:
          ldp ${A_0_q}, ${A_2_q}, [DG42_A_PTR_${(ii*2)%(2*unroll)}]
          madd DG42_A_PTR_${(ii*2)%(2*unroll)}, ${A_STRIDE}, ISCRATCH_0, DG42_A_PTR_${(ii*2)%(2*unroll)}
        %endif
        ${fmla} C_0.2d, ${A_1}.2d, ${B_0}.d[1]
        ${fmla} C_1.2d, ${A_1}.2d, ${B_1}.d[1]
        ${fmla} C_2.2d, ${A_3}.2d, ${B_0}.d[1]
        ${fmla} C_3.2d, ${A_3}.2d, ${B_1}.d[1]
        %if do_preload:
          ldp ${A_1_q}, ${A_3_q}, [DG42_A_PTR_${(ii*2+1)%(2*unroll)}]
          madd DG42_A_PTR_${(ii*2+1)%(2*unroll)}, ${A_STRIDE}, ISCRATCH_0, DG42_A_PTR_${(ii*2+1)%(2*unroll)}
          ldr ${B_0}_q, [DG42_B_PTR_0], #16
          ldr ${B_1}_q, [DG42_B_PTR_1], #16
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
  ldp q0, q1, [DG42_A_PTR_0]
  ldr d2, [DG42_B_PTR_0]
  ldr d3, [DG42_B_PTR_1]
  ${fmla} C_0.2d, v0.2d, v2.d[0]
  ${fmla} C_1.2d, v0.2d, v3.d[0]
  ${fmla} C_2.2d, v1.2d, v2.d[0]
  ${fmla} C_3.2d, v1.2d, v3.d[0]
  sub DG42_K, DG42_K, #1
  .L${func_name}_cleanup_loop_cond:
  cmp DG42_K, #1
  bge .L${func_name}_cleanup_loop

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
  .unreq DG42_A_PTR_4
  .unreq DG42_A_PTR_5
  .unreq DG42_A_PTR_6
  .unreq DG42_A_PTR_7
  .unreq DG42_B_PTR_0
  .unreq DG42_B_PTR_1
</%def>
