## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

<%def name="cgemm_save_indiv(cgemm_save_name)">
.L${cgemm_save_name}_start:
	add t, idx_n, 0
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end0


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re0.s[0]
	dup t3, cr1_im0.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end0


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re0.s[1]
	dup t3, cr1_im0.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end0


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re0.s[2]
	dup t3, cr1_im0.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end0


	dup t2, cr1_re0.s[3]
	dup t3, cr1_im0.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end0:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 1
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end1


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re1.s[0]
	dup t3, cr1_im1.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end1


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re1.s[1]
	dup t3, cr1_im1.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end1


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re1.s[2]
	dup t3, cr1_im1.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end1


	dup t2, cr1_re1.s[3]
	dup t3, cr1_im1.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end1:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 2
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end2


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re2.s[0]
	dup t3, cr1_im2.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end2


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re2.s[1]
	dup t3, cr1_im2.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end2


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re2.s[2]
	dup t3, cr1_im2.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end2


	dup t2, cr1_re2.s[3]
	dup t3, cr1_im2.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end2:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 3
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end3


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re3.s[0]
	dup t3, cr1_im3.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end3


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re3.s[1]
	dup t3, cr1_im3.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end3


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re3.s[2]
	dup t3, cr1_im3.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end3


	dup t2, cr1_re3.s[3]
	dup t3, cr1_im3.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end3:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 4
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end4


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re4.s[0]
	dup t3, cr1_im4.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end4


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re4.s[1]
	dup t3, cr1_im4.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end4


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re4.s[2]
	dup t3, cr1_im4.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end4


	dup t2, cr1_re4.s[3]
	dup t3, cr1_im4.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end4:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 5
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end5


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re5.s[0]
	dup t3, cr1_im5.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end5


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re5.s[1]
	dup t3, cr1_im5.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end5


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re5.s[2]
	dup t3, cr1_im5.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end5


	dup t2, cr1_re5.s[3]
	dup t3, cr1_im5.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end5:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 6
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end6


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re6.s[0]
	dup t3, cr1_im6.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end6


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re6.s[1]
	dup t3, cr1_im6.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end6


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re6.s[2]
	dup t3, cr1_im6.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end6


	dup t2, cr1_re6.s[3]
	dup t3, cr1_im6.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end6:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 7
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end7


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re7.s[0]
	dup t3, cr1_im7.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end7


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re7.s[1]
	dup t3, cr1_im7.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end7


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re7.s[2]
	dup t3, cr1_im7.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end7


	dup t2, cr1_re7.s[3]
	dup t3, cr1_im7.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end7:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 8
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end8


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re8.s[0]
	dup t3, cr1_im8.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end8


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re8.s[1]
	dup t3, cr1_im8.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end8


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re8.s[2]
	dup t3, cr1_im8.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end8


	dup t2, cr1_re8.s[3]
	dup t3, cr1_im8.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end8:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

	add t, idx_n, 9
	cmp t, n
	beq .L${cgemm_save_name}_final

	cmp m, idx_m
	beq .L${cgemm_save_name}_end9


	ldr t0, [ local_c_ptr0 ]
	ldr t1, [ local_c_ptr0, 4 ]

	dup t2, cr1_re9.s[0]
	dup t3, cr1_im9.s[0]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0 ]
	str t3, [ local_c_ptr0, 4 ]



	add t, idx_m, 1
	cmp m, t
	beq .L${cgemm_save_name}_end9


	ldr t0, [ local_c_ptr0, 8 ]
	ldr t1, [ local_c_ptr0, 12 ]

	dup t2, cr1_re9.s[1]
	dup t3, cr1_im9.s[1]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 8 ]
	str t3, [ local_c_ptr0, 12 ]



	add t, idx_m, 2
	cmp m, t
	beq .L${cgemm_save_name}_end9


	ldr t0, [ local_c_ptr0, 16 ]
	ldr t1, [ local_c_ptr0, 20 ]

	dup t2, cr1_re9.s[2]
	dup t3, cr1_im9.s[2]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 16 ]
	str t3, [ local_c_ptr0, 20 ]



	add t, idx_m, 3
	cmp m, t
	beq .L${cgemm_save_name}_end9


	dup t2, cr1_re9.s[3]
	dup t3, cr1_im9.s[3]

	ldr t0, [ local_c_ptr0, 24 ]
	ldr t1, [ local_c_ptr0, 28 ]

	fadd t2, t0, t2
	fadd t3, t1, t3

	str t2, [ local_c_ptr0, 24 ]
	str t3, [ local_c_ptr0, 28 ]

.L${cgemm_save_name}_end9:
	add local_c_ptr0, local_c_ptr0, ldc, lsl 3

.L${cgemm_save_name}_final:
</%def>
