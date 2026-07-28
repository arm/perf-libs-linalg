## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
#define m x0
#define n x1
#define k x2
#define A x3
#define lda x4
#define B x5
#define ldb x6
#define C x7
#define ldc x8
#define k_idx x9
#define m_idx x10
#define n_idx x11
#define mx x12
#define lda_pf x13
#define nx x14
#define k4 x15
#define tmp1_x x16
#define tmp2_x x17
#define tmp3_x x19
#define tmp4_x x20
#define tmp5_x x21
#define tmp6_x x22
#define tmp7_x x23

#define alphabeta_v v0

<% M_UNROLL = 3 %>\
<% N_UNROLL = 6 %>\

<% ca_vr = 31 %>\
%for ni in range(N_UNROLL):
%for mi in range(M_UNROLL):
#define Ca_m${mi}_n${ni}_s s${ca_vr}
#define Ca_m${mi}_n${ni}_d d${ca_vr}
#define Ca_m${mi}_n${ni}_q q${ca_vr}
#define Ca_m${mi}_n${ni}_v v${ca_vr}
<% ca_vr -= 1 %>\
%endfor
%endfor

<% ab_vr = 0 %>\
%for mi in range(M_UNROLL):
#define A_m${mi}_s s${ab_vr}
#define A_m${mi}_d d${ab_vr}
#define A_m${mi}_q q${ab_vr}
#define A_m${mi}_v v${ab_vr}
<% ab_vr += 1 %>\
%endfor

<% N_TWIST_MAX = 0 %>\

%for ni in range(N_UNROLL):
#define B_n${ni}a_s s${ab_vr}
#define B_n${ni}a_d d${ab_vr}
#define B_n${ni}a_v v${ab_vr}
#define B_n${ni}a_q q${ab_vr}
<% ab_vr += 1 %>\
%endfor
%for ni in range(N_UNROLL):
%if ab_vr <= ca_vr:
#define B_n${ni}b_s s${ab_vr}
#define B_n${ni}b_d d${ab_vr}
#define B_n${ni}b_v v${ab_vr}
#define B_n${ni}b_q q${ab_vr}
<% ab_vr += 1 %>\
<% N_TWIST_MAX += 1 %>\
%else:
#define B_n${ni}b_s B_n${ni}a_s
#define B_n${ni}b_d B_n${ni}a_d
#define B_n${ni}b_v B_n${ni}a_v
#define B_n${ni}b_q B_n${ni}a_q
%endif
%endfor

// c block registers start from v1 to leave space
// for alpha/beta, which are loaded in v0.s[0/1].
<% c_vr = 1 %>\
%for mi in range(M_UNROLL):
#define C_m${mi}_s s${c_vr}
#define C_m${mi}_d d${c_vr}
#define C_m${mi}_q q${c_vr}
#define C_m${mi}_v v${c_vr}
<% c_vr += 1 %>\
%endfor

<% assert ab_vr <= ca_vr+1 %>\
<% assert c_vr <= ca_vr+1 %>\

#define XZERO(r) eor r, r, r
#define VZERO(r) eor r.16b, r.16b, r.16b

// labels
#define N_LOOP_START(x) 1##x
#define N_LOOP(x) 2##x
#define N_LOOP_DONE(x) 3##x
#define M_LOOP_START(x) 4##x
#define M_LOOP(x) 5##x
#define M_LOOP_DONE(x) 6##x
#define K_LOOP_START(x) 7##x
#define K_LOOP(x) 8##x
#define K_LOOP_TAIL_a(x) 9##x
#define K_LOOP_TAIL_b(x) 10##x
#define K_LOOP_DONE(x) 11##x
#define BETA_ZERO(x) 12##x
#define BETA_ONE(x) 13##x
#define EPILOGUE(x) 14##x

<%def name="round4(out, in_, tmp=None)">
	# round ${in_} down to a multiple of 4
	and ${out}, ${in_}, -4
</%def>
	<% func_name = "sgemm_small_kernel_nn_tx2" %>
	${prologue(func_name)}
	stp x29, x30, [sp, #-304]!
	mov x29, sp
	stp s0, s1, [ sp, #16 ]
	stp x19, x20, [ sp, #160 ]
	stp x21, x22, [ sp, #176 ]
	stp x23, x24, [ sp, #192 ]
	stp x25, x26, [ sp, #208 ]
	stp x27, x28, [ sp, #224 ]
	stp d8, d9, [ sp, #240 ]
	stp d10, d11, [ sp, #256 ]
	stp d12, d13, [ sp, #272 ]
	stp d14, d15, [ sp, #288 ]

	ldr ldc, [ sp, #304 ]

	mov lda_pf, #(8 << 2)
	mul lda_pf, lda_pf, lda

	${round4("k4", "k")}
	sub k4, k, k4

	fmov s0, #1.0
	fcmp s1, s0
	beq BETA_ONE(f)

	fcmp s1, #0.0
	beq BETA_ZERO(f)

<%def name="m_loop_body(munroll, nunroll, beta_zero, beta_one)">
K_LOOP_START():
<% munroll_offsets = [] %>\
<% munroll_ofs_tmp = 0 %>\
<% munroll_suffixv = [] %>\
<% munroll_suffixq = [] %>\
<% munroll_tmp = munroll %>\
%while munroll_tmp > 0:
%if munroll_tmp >= 4:
<% munroll_offsets.append(munroll_ofs_tmp) %>\
<% munroll_suffixv.append("v.4s") %>\
<% munroll_suffixq.append("q") %>\
<% munroll_tmp -= 4 %>\
<% munroll_ofs_tmp += 16 %>\
%elif munroll_tmp >= 2:
<% munroll_offsets.append(munroll_ofs_tmp) %>\
<% munroll_suffixv.append("v.2s") %>\
<% munroll_suffixq.append("d") %>\
<% munroll_tmp -= 2 %>\
<% munroll_ofs_tmp += 8 %>\
%else:
<% munroll_offsets.append(munroll_ofs_tmp) %>\
<% munroll_suffixv.append("s") %>\
<% munroll_suffixq.append("s") %>\
<% munroll_tmp -= 1 %>\
<% munroll_ofs_tmp += 4 %>\
%endif
%endwhile
<% mi_last = len(munroll_suffixq)-1 %>\
<% q_last = munroll_suffixq[-1] %>\
<% ofs_last = munroll_offsets[-1] %>\

	subs k_idx, k, #4

	mul tmp2_x, n_idx, ldb
	add tmp2_x, B, tmp2_x, lsl #2
%for ni in range(1, nunroll):
	add tmp${ni+2}_x, tmp${ni+1}_x, ldb, lsl #2
%endfor
	add tmp1_x, A, m_idx, lsl #2

%for ni in range(nunroll):
%for mi in range(len(munroll_suffixq)):
	VZERO(Ca_m${mi}_n${ni}_v)
%endfor
%endfor

	blt K_LOOP_DONE(f)
	subs k_idx, k_idx, #4

%for mi in range(len(munroll_suffixq)-1):
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	ldr A_m${mi}_${q}, [tmp1_x, #${ofs}]
%endfor

%for ni in range(nunroll):
	ldr B_n${ni}a_q, [tmp${ni+2}_x], #16
%endfor

	blt K_LOOP_TAIL_a(f)
K_LOOP():

%for ab_i in range(2):
<% ab = ["a", "b"][ab_i] %>\
<% ba = ["b", "a"][ab_i] %>\
\
<% instrs_ready = [] %>\
<% sorted0 = lambda x: sorted(x, key=lambda e: e[0]) %>\
<% instrs_ready.append((0, "ldr A_m{}_{}, [tmp1_x, #{}]".format(mi_last, q_last, ofs_last))) %>\
<% instrs_ready.append((9, "subs k_idx, k_idx, #4")) %>\
%for i in range(min(nunroll, N_TWIST_MAX)):
<% instrs_ready.append((10, "ldr B_n{}{}_q, [tmp{}_x], #16".format(i, ba, i+2))) %>\
%endfor
<% b_load_idx = N_TWIST_MAX %>\
\
%for i in range(4):
<% instrs_ready.append((1, "add tmp1_x, tmp1_x, lda, lsl #2")) %>\
%if nunroll > 2 and len(munroll_suffixq) > 2:
<% instrs_ready.append((1, "prfm pldl1keep, [tmp1_x, lda_pf]")) %>\
%endif
%for mi in range(len(munroll_suffixv)):
%if len(instrs_ready) > 0:
<% instrs_ready = sorted0(instrs_ready) %>\
	${instrs_ready[0][1]}
<% instrs_ready = instrs_ready[1:] %>\
%endif
%for ni in range(nunroll-1, -1, -1):
<% v = munroll_suffixv[mi] %>\
	fmla Ca_m${mi}_n${ni}_${v}, A_m${mi}_${v}, B_n${ni}${ab}_v.s[${i}]
%if ni in (range(2, nunroll+(nunroll % 2), 2) if nunroll > 2 else [nunroll//2]):
%while i == 3 and mi == mi_last and ni <= b_load_idx and b_load_idx < nunroll:
<% instrs_ready.append((10, "ldr B_n{}{}_q, [tmp{}_x], #16".format(b_load_idx, ba, b_load_idx+2))) %>\
<% b_load_idx += 1 %>\
%endwhile
%if len(instrs_ready) > 0:
<% instrs_ready = sorted0(instrs_ready) %>\
	${instrs_ready[0][1]}
<% instrs_ready = instrs_ready[1:] %>\
%endif
%endif
%endfor
%if mi < len(munroll_suffixq)-1 or i < 3:
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
<% instrs_ready.append((1, "ldr A_m{}_{}, [tmp1_x, #{}]".format(mi, q, ofs))) %>\
%endif
%endfor
%endfor

%for _, instr in instrs_ready:
	${instr}
%endfor

%for ni in range(b_load_idx, nunroll):
	ldr B_n${ni}${ba}_q, [tmp${ni+2}_x], #16
%endfor

%if ab_i == 0:
	blt K_LOOP_TAIL_b(f)
%else:
	bge K_LOOP(b)
%endif
%endfor

%for ab_i in range(2):
<% ab = ["a", "b"][ab_i] %>\
K_LOOP_TAIL_${ab}():
	ldr A_m${mi_last}_${q_last}, [tmp1_x, #${ofs_last}]

%for i in range(4):
	add tmp1_x, tmp1_x, lda, lsl #2
%for mi in range(len(munroll_suffixv)):
%for ni in range(nunroll):
<% v = munroll_suffixv[mi] %>\
	fmla Ca_m${mi}_n${ni}_${v}, A_m${mi}_${v}, B_n${ni}${ab}_v.s[${i}]
%endfor
%if i < 3:
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	ldr A_m${mi}_${q}, [tmp1_x, #${ofs}]
%endif
%endfor
%endfor
%if ab_i < 1:
	b K_LOOP_DONE(f)
%endif
%endfor

K_LOOP_DONE():
	subs k_idx, k4, #0
	ble K_LOOP_DONE(f)
K_LOOP():
%for mi in range(len(munroll_suffixq)):
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	ldr A_m${mi}_${q}, [tmp1_x, #${ofs}]
%endfor
	add tmp1_x, tmp1_x, lda, lsl #2

%for ni in range(nunroll):
	ldr B_n${ni}a_s, [tmp${ni+2}_x], #4
%endfor

	subs k_idx, k_idx, #1

%for mi in range(len(munroll_suffixv)):
%for ni in range(nunroll):
<% v = munroll_suffixv[mi] %>\
	fmla Ca_m${mi}_n${ni}_${v}, A_m${mi}_${v}, B_n${ni}a_v.s[0]
%endfor
%endfor

	bgt K_LOOP(b)

K_LOOP_DONE():
%if beta_zero or beta_one:
	ldr s0, [ sp, #16 ]
%else:
	ldr d0, [ sp, #16 ]
%endif

	// tmp1 = &C[n*ldc + m]
	madd tmp1_x, n_idx, ldc, m_idx
	add tmp1_x, C, tmp1_x, lsl #2

%for ni in range(nunroll):
%if beta_zero:
%for mi in range(len(munroll_suffixq)):
<% q = munroll_suffixq[mi] %>\
<% v = munroll_suffixv[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	fmul Ca_m${mi}_n${ni}_${v}, Ca_m${mi}_n${ni}_${v}, alphabeta_v.s[0]
	str Ca_m${mi}_n${ni}_${q}, [tmp${ni+1}_x, #${ofs}]
%endfor
%if ni < nunroll-1:
	add tmp${ni+2}_x, tmp${ni+1}_x, ldc, lsl #2
%endif
%else:
%for mi in range(len(munroll_suffixq)):
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	ldr C_m${mi}_${q}, [tmp${ni+1}_x, #${ofs}]
%endfor
%if ni < nunroll-1:
	add tmp${ni+2}_x, tmp${ni+1}_x, ldc, lsl #2
%endif
%if not beta_one:
%for mi in range(len(munroll_suffixv)):
<% v = munroll_suffixv[mi] %>\
	fmul Ca_m${mi}_n${ni}_${v}, Ca_m${mi}_n${ni}_${v}, alphabeta_v.s[0]
%endfor
%for mi in range(len(munroll_suffixv)):
<% v = munroll_suffixv[mi] %>\
	fmla Ca_m${mi}_n${ni}_${v}, C_m${mi}_${v}, alphabeta_v.s[1]
%endfor
%for mi in range(len(munroll_suffixq)):
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	str Ca_m${mi}_n${ni}_${q}, [tmp${ni+1}_x, #${ofs}]
%endfor
%else:
	// beta is one
%for mi in range(len(munroll_suffixv)):
<% v = munroll_suffixv[mi] %>\
	fmla C_m${mi}_${v}, Ca_m${mi}_n${ni}_${v}, alphabeta_v.s[0]
%endfor
%for mi in range(len(munroll_suffixq)):
<% q = munroll_suffixq[mi] %>\
<% ofs = munroll_offsets[mi] %>\
	str C_m${mi}_${q}, [tmp${ni+1}_x, #${ofs}]
%endfor
%endif
%endif
%endfor

	add m_idx, m_idx, #${munroll}
</%def>

<%def name="n_loop_body(nunroll, beta_zero, beta_one)">
	XZERO(m_idx)
M_LOOP_START():

<% mfactors = [M_UNROLL*4, (M_UNROLL-1)*4+2, (M_UNROLL-1)*4+1] + list(range((M_UNROLL-1)*4, 0, -1)) %>\
// m unroll factors = ${mfactors}

%for i, mfactor in enumerate(mfactors):
%if mfactor > 1:
	sub mx, m, #${mfactor-1}
%endif
M_LOOP():
%if mfactor > 1:
	cmp m_idx, mx
%else:
	cmp m_idx, m
%endif
	bge M_LOOP_DONE(f)
${m_loop_body(mfactor, nunroll, beta_zero, beta_one)}
	b M_LOOP(b)
M_LOOP_DONE():
%endfor

	add n_idx, n_idx, #${nunroll}
</%def>

<%def name="gemm_body(beta_zero=False, beta_one=False)">
N_LOOP_START():
	XZERO(n_idx)

<% nfactors = list(range(N_UNROLL, 0, -1)) %>\
// n unroll factors = ${nfactors}

%for i, nfactor in enumerate(nfactors):
%if nfactor > 1:
	sub nx, n, #${nfactor-1}
%endif
N_LOOP():
%if nfactor > 1:
	cmp n_idx, nx
%else:
	cmp n_idx, n
%endif
	bge N_LOOP_DONE(f)
${n_loop_body(nfactor, beta_zero, beta_one)}
	b N_LOOP(b)
N_LOOP_DONE():
%endfor
</%def>

${gemm_body()}
	b EPILOGUE(f)

BETA_ONE():
${gemm_body(beta_one=True)}
	b EPILOGUE(f)

BETA_ZERO():
${gemm_body(beta_zero=True)}

EPILOGUE():
	ldp x19, x20, [ sp, #160 ]
	ldp x21, x22, [ sp, #176 ]
	ldp x23, x24, [ sp, #192 ]
	ldp x25, x26, [ sp, #208 ]
	ldp x27, x28, [ sp, #224 ]
	ldp d8, d9, [ sp, #240 ]
	ldp d10, d11, [ sp, #256 ]
	ldp d12, d13, [ sp, #272 ]
	ldp d14, d15, [ sp, #288 ]
	ldp x29, x30, [sp], #304
	ret

	${epilogue(func_name)}
