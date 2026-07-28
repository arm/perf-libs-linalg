## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>

#define A x0
#define B x1
#define C x2
#define K x3
#define M x4
#define N x5
#define ldc x6
#define k_idx x7
#define alpha_z z0
#define beta_z z1

<% C_M=1 %>
<% C_N=3 %>
<% VE=8 %> ##how many elements of datatype fit in 128bits


<%
def reg_block_range(m, n):
	for i in range(m):
		for j in range(n):
			yield (i, j)
%>


<%! pup = 0 %>
<%
def preg():
	global pup
	pup = pup + 1
	return pup - 1
%>


<%! xup = 8 %>
<%
def xreg():
	global xup
	xup = xup + 1
	return xup - 1
%>

<%!  up = 0 %>
<%
def reg():
	global up
	up = up + 1
	return up - 1
%>
<%!
def rq_reg_name(i):
	return "{}_{}".format(i * 8, (i+1)*8-1)
%>

tmp1_x .req x${xreg()}
tmp2_x .req x${xreg()}
tmp3_x .req x${xreg()}
tmp4_x .req x${xreg()}

m_idx .req x${xreg()}
m_idx_t .req x${xreg()}

%for i in range(C_M):
	A_it0_m${i} .req z${reg()}
	A_it1_m${i} .req z${reg()}

	C_m${i} .req z${reg()}

	m_prd_${i} .req p${preg()}
	all_true_prd .req p${preg()}
%endfor


n_idx .req x${xreg()}
n_idx_t .req x${xreg()}

%for i in range(C_N):
	B_n${rq_reg_name(i)} .req z${reg()}
%endfor

%for m, n in reg_block_range(C_M, C_N*VE):
	Ca_m${m}_n${n} .req z${reg()}
%endfor




#define lda K
#define ldb K

<% func_name = "hgemm_sve_big" %>
#define FN hgemm_sve_big
#define FN_(l) FN##_##l

#define XZERO(r) eor r, r, r
#define VZERO(r) eor r.16b, r.16b, r.16b
#define ZZERO(r) eor r.d, r.d, r.d

	${prologue(func_name)}
	stp x29, x30, [sp, #-320]!
	mov x29, sp
	stp x19, x20, [ sp, #224 ]
	stp x21, x22, [ sp, #208 ]
	stp x23, x24, [ sp, #192 ]
	stp x25, x26, [ sp, #176 ]
	stp x27, x28, [ sp, #160 ]
	stp d8, d9, [ sp, #80 ]
	stp d10, d11, [ sp, #64 ]
	stp d12, d13, [ sp, #48 ]
	stp d14, d15, [ sp, #32 ]
	stp s0, s1, [ sp, #16 ]

	ptrue all_true_prd.b

	XZERO(n_idx)

	// check if beta is 0.0 and branch to correct loop.
	fcmp h1, #0.0
	beq .Lbzero_n_loop

	// check if beta is 1.0 and branch to correct loop.
	fmov h0, #1.0
	fcmp h1, h0
	beq .Lbone_n_loop

%for beta in ["general", "one", "zero"]:
<% L = ".Lb{}".format(beta) %>\

	${L}_n_loop:
		cmp n_idx, N
		bge ${L}_n_loop_done
		XZERO(m_idx)
	${L}_m_loop:
		cmp m_idx, M
		bge ${L}_m_loop_done

		mul tmp1_x, m_idx, lda
		add tmp1_x, A, tmp1_x, lsl #1
		mul tmp2_x, n_idx, ldb
		add tmp2_x, B, tmp2_x, lsl #1

		subs k_idx, K, #0

	%for m, n in reg_block_range(C_M, C_N*VE):
		ZZERO(Ca_m${m}_n${n})
	%endfor
		beq ${L}_k_loop_done
		subs k_idx, k_idx, #1
	%for i in range(C_M):
		addvl tmp1_x, tmp1_x, #${C_M}
		ld1h { A_it0_m${i}.h }, all_true_prd/z, [tmp1_x, ${i-1}, mul vl]
	%endfor


	%for i in range(C_N):
		ld1rqh { B_n${rq_reg_name(i)}.h }, all_true_prd/z, [tmp2_x, ${i*16}]
	%endfor

	add tmp2_x, tmp2_x, #(${C_N} << 4)

	beq ${L}_k_loop_last_0

	${L}_k_loop:
	%for it in range(2):
		<% next = (it+1) % 2 %>
		<% lds = [] %>
		<%
		for i in range(C_M):
			lds.append("ld1h {{ A_it{}_m{}.h }}, all_true_prd/z, [ tmp1_x, #-{}, mul vl ]".format(next, i, 1 - i))
		%>
		<%
		lds.extend([
			"addvl tmp1_x, tmp1_x, #{}".format(C_M),
			"add tmp2_x, tmp2_x, #({} << 4)".format(C_N),
			"subs k_idx, k_idx, #1"])
		%>
		<% x=0 %>
		%for n in range(C_N):
			%for m in range(C_M):
				%for indx in range(VE):
					fmla Ca_m${m}_n${n * VE +indx}.h, A_it${it}_m${m}.h, B_n${rq_reg_name(n)}.h[${indx}]
					<%  x=x+1 %>
					%if x % 2==0 and lds:
						${lds.pop()}
					%endif
				%endfor
			%endfor
			<%
				lds.insert(0,"ld1rqh {{ B_n{}.h }}, all_true_prd/z, [tmp2_x, #-{}]".format(rq_reg_name(n), (C_N - n)*16 ))
			%>
		%endfor

		%while lds:
			${lds.pop()}
		%endwhile
		##<% assert not lds %>
		beq ${L}_k_loop_last_${next}
	%endfor
	b ${L}_k_loop



	//stamp out the tail loops
	%for it in range(2):
		${L}_k_loop_last_${it}:
		%for n in range(C_N):
			%for m in range(C_M):
				%for indx in range(VE):
					fmla Ca_m${m}_n${n * VE +indx}.h, A_it${it}_m${m}.h, B_n${rq_reg_name(n)}.h[${indx}]
				%endfor
			%endfor
		%endfor
		b ${L}_k_loop_done
	%endfor





	${L}_k_loop_done:

	whilelt m_prd_0.h, m_idx, M

	%if C_M > 1:
		mov m_idx_t, m_idx
	%endif

	%for i in range(1, C_M):
		inch m_idx_t
		whilelt m_prd_${i}.h, m_idx_t, M
	%endfor


	ld1rh { alpha_z.h }, all_true_prd/z, [ sp, #16 ]
	ld1rh { beta_z.h }, all_true_prd/z, [ sp, #20 ]

	%for i in range(0, C_M):
	%endfor


	%for n in range(C_N*VE):
		add n_idx_t, n_idx, #${n}
		cmp n_idx_t, N
		bge ${L}_save_end

		madd m_idx_t, n_idx_t, ldc, m_idx
		add m_idx_t, C, m_idx_t, lsl #1

		%for m in range(C_M):

			%if beta == "one":
				ld1h { C_m${m}.h }, m_prd_${i}/z, [m_idx_t, #${i}, mul vl]
				fmla C_m${m}.h, all_true_prd/m, Ca_m${m}_n${n}.h, alpha_z.h
				st1h { C_m${m}.h }, m_prd_${m}, [m_idx_t, #${i}, mul vl]
			%elif beta == "zero":
				fmul Ca_m${m}_n${n}.h, all_true_prd/m, Ca_m${m}_n${n}.h, alpha_z.h
				st1h { Ca_m${m}_n${n}.h }, m_prd_${m}, [m_idx_t, #${i}, mul vl]
			%else:
				ld1h { C_m${m}.h }, m_prd_${i}/z, [m_idx_t, #${i}, mul vl]
				fmul Ca_m${m}_n${n}.h, all_true_prd/m, Ca_m${m}_n${n}.h, alpha_z.h
				fmla Ca_m${m}_n${n}.h, all_true_prd/m, C_m${m}.h, beta_z.h
				st1h { Ca_m${m}_n${n}.h }, m_prd_${m}, [m_idx_t, #${i}, mul vl]
			%endif
		%endfor
	%endfor



	${L}_save_end:
		inch m_idx, all, mul #${C_M}
		b ${L}_m_loop
	${L}_m_loop_done:
		add n_idx, n_idx, #${C_N*VE}
		b ${L}_n_loop

	${L}_n_loop_done:
		b .FN_(epilogue)
%endfor #bete_is_one

.FN_(epilogue):
	ldp x19, x20, [ sp, #224 ]
	ldp x21, x22, [ sp, #208 ]
	ldp x23, x24, [ sp, #192 ]
	ldp x25, x26, [ sp, #176 ]
	ldp x27, x28, [ sp, #160 ]
	ldp d8, d9, [ sp, #80 ]
	ldp d10, d11, [ sp, #64 ]
	ldp d12, d13, [ sp, #48 ]
	ldp d14, d15, [ sp, #32 ]
	ldp x29, x30, [sp], #320
	ret

	${epilogue(func_name)}
