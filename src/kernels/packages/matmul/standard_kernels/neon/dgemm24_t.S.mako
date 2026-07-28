## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

/**
  Clobbers all vector registers >= 4, gemms into 4 C registers which should be
  reqed as C_0, C_1, C_2, and C_3. Needs ISCRATCH_1 to ISCRATCH_5 inclusive.
    @param A - pointer to A (const)
    @param B - pointer to B (const)
    @param A_STRIDE - lda (const)
    @param B_STRIDE - ldb (const)
    @param K - K dimension (const)
 */
<%def name="dgemm24_t(func_name, A, B, A_STRIDE, B_STRIDE, K, fmla='fmla')">

// A single unroll uses a full register block
<% unroll_progression = [1] %>
<% max_unroll = max(unroll_progression) %>

<%
## Size of register blocks in register
A_BLOCK_W = 2
A_BLOCK_H = 1
B_BLOCK_W = 2
B_BLOCK_H = 2
NUM_A_REG = A_BLOCK_W*A_BLOCK_H*4
NUM_B_REG = B_BLOCK_W*B_BLOCK_H*4
%>

<% func_name += '_dgemm24_t' %>

// Reg blocks go from right to left.
%for ii in range(NUM_A_REG):
  A_${ii} .req v${4+ii}
  A_${ii}_q .req q${4+ii}
%endfor
%for ii in range(NUM_B_REG):
  B_${ii} .req v${4+ii+NUM_A_REG}
  B_${ii}_q .req q${4+ii+NUM_A_REG}
%endfor

  DG42_A_PTR_0 .req ISCRATCH_1
  DG42_A_PTR_1 .req ISCRATCH_2
  DG42_B_PTR_0 .req ISCRATCH_3
  DG42_B_PTR_1 .req ISCRATCH_4

  DG24_K .req ISCRATCH_5

  mov DG24_K, ${K}

  mov DG42_A_PTR_0, ${A}
  mov DG42_B_PTR_0, ${B}

  // Setup pointers
  add DG42_B_PTR_1, DG42_B_PTR_0, ${B_STRIDE}
  add DG42_A_PTR_1, DG42_A_PTR_0, ${A_STRIDE}

  <% orig_func_name = func_name %>

  %for unroll in unroll_progression:
    <% func_name += "_unroll" + str(unroll) %>

    cmp DG24_K, #${unroll*A_BLOCK_W}
    blt .L${func_name}_loop_skip

    // Preload A and B blocks
    %for ii in range(unroll):
      ldr A_${2*ii+0}_q, [DG42_A_PTR_0]
      ldr A_${2*ii+1}_q, [DG42_A_PTR_1]
      add DG42_A_PTR_0, DG42_A_PTR_0, ${A_STRIDE}, lsl #1
      add DG42_A_PTR_1, DG42_A_PTR_1, ${A_STRIDE}, lsl #1
    %endfor
    %for ii in range(unroll):
      ldp B_${4*ii+0}_q, B_${4*ii+1}_q, [DG42_B_PTR_0]
      ldp B_${4*ii+2}_q, B_${4*ii+3}_q, [DG42_B_PTR_1]
      add DG42_B_PTR_0, DG42_B_PTR_0, ${B_STRIDE}, lsl #1
      add DG42_B_PTR_1, DG42_B_PTR_1, ${B_STRIDE}, lsl #1
    %endfor

    <% orig_func_name_2 = func_name %>
    %for do_preload in [True, False]:
      <% func_name += "_preload" + str(do_preload) %>
      b .L${func_name}_loop_cond
      .L${func_name}_loop:

      %for ii in range(unroll):
        <% A_0 = "A_" + str(ii*2)   %>\
        <% A_1 = "A_" + str(ii*2+1) %>\
        <% B_0 = "B_" + str(ii*4)   %>\
        <% B_1 = "B_" + str(ii*4+1) %>\
        <% B_2 = "B_" + str(ii*4+2) %>\
        <% B_3 = "B_" + str(ii*4+3) %>\

        // Do a 1x2x2 reg block gemm (6 registers).
        ${fmla} C_0.2d, ${A_0}.2d, ${B_0}.d[0]
        ${fmla} C_1.2d, ${A_0}.2d, ${B_0}.d[1]
        %if do_preload:
          ldr ${B_0}_q, [DG42_B_PTR_0]
        %endif
        ${fmla} C_2.2d, ${A_0}.2d, ${B_1}.d[0]
        ${fmla} C_3.2d, ${A_0}.2d, ${B_1}.d[1]
        %if do_preload:
          ldr ${B_1}_q, [DG42_B_PTR_0, #16]
          ldr ${A_0}_q, [DG42_A_PTR_0]
          add DG42_B_PTR_0, DG42_B_PTR_0, B_STRIDE, lsl #1
          add DG42_A_PTR_0, DG42_A_PTR_0, A_STRIDE, lsl #1
        %endif
        ${fmla} C_0.2d, ${A_1}.2d, ${B_2}.d[0]
        ${fmla} C_1.2d, ${A_1}.2d, ${B_2}.d[1]

        %if do_preload:
          ldr ${B_2}_q, [DG42_B_PTR_1]
        %endif
        ${fmla} C_2.2d, ${A_1}.2d, ${B_3}.d[0]
        ${fmla} C_3.2d, ${A_1}.2d, ${B_3}.d[1]
        %if do_preload:
          ldr ${B_3}_q, [DG42_B_PTR_1, #16]
          ldr ${A_1}_q, [DG42_A_PTR_1]
          add DG42_B_PTR_1, DG42_B_PTR_1, B_STRIDE, lsl #1
          add DG42_A_PTR_1, DG42_A_PTR_1, A_STRIDE, lsl #1
        %endif
      %endfor

      sub DG24_K, DG24_K, #${unroll*A_BLOCK_W}
      .L${func_name}_loop_cond:
      cmp DG24_K, #${unroll*(2*A_BLOCK_W if do_preload else A_BLOCK_W)}
      bge .L${func_name}_loop
      <% func_name = orig_func_name_2 %>
    %endfor // do_preload

    .L${func_name}_loop_skip:

    <% func_name = orig_func_name %>
  %endfor

  // Cleanup loop
  b .L${func_name}_cleanup_loop_cond
  .L${func_name}_cleanup_loop:
  ldr q4, [DG42_A_PTR_0]
  ldp q5, q6, [DG42_B_PTR_0]
  add DG42_A_PTR_0, DG42_A_PTR_0, A_STRIDE
  add DG42_B_PTR_0, DG42_B_PTR_0, B_STRIDE
  ${fmla} C_0.2d, v4.2d, v5.d[0]
  ${fmla} C_1.2d, v4.2d, v5.d[1]
  ${fmla} C_2.2d, v4.2d, v6.d[0]
  ${fmla} C_3.2d, v4.2d, v6.d[1]
  sub DG24_K, DG24_K, #1
  .L${func_name}_cleanup_loop_cond:
  cmp DG24_K, #1
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
  .unreq DG42_B_PTR_0
  .unreq DG42_B_PTR_1
</%def>