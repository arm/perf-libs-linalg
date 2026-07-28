# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import math
import numpy as np
from atomic_gemm_common import *

# Max general purpose registers
# By default we do not allow the use of x18 or x29
# In some ABIs (e.g. Windows Arm64EC) we restrict this even further
NUM_GPRS = 29
# Max Neon vector registers
NUM_VREGS = 32
# First unused general purpose register (x6). x0-x5 used for parameters.
BASE_GPR = 5

BETA_REGISTER = 1

ALPHA_REAL_REGISTER = 0
ALPHA_IMAG_REGISTER = 1
BETA_REAL_REGISTER = 2
BETA_IMAG_REGISTER = 3

FPZERO = "#0.0"

class AtomicKernelException(Exception):
    def __init__(self, m, n, k, trans, dtype):
        Exception.__init__(self, "Kernel is not atomic. m={}, n={}, k={}, trans={}, dtype={}".format(m, n, k, trans.fullname, dtype.name))

class AtomicKernelRegInfoNeon(AtomicKernelRegInfo):
    def __init__(self, dtype, target_os):
        self.dtype = dtype
        self.is_complex = dtype.is_complex
        self.target_os = target_os
        # The methods generated may use x30 as a temporary register, and it is
        # spilled and restored if overwritten. We do not use x18 or x29. This
        # means we have a selection of 29 general purpose registers and 32 vector
        # registers to use. On Windows Arm64EC we additionally cannot use x13, x14,
        # x23, x24 or x28 so we have to reduce NUM_GPRS by 5.
        super(AtomicKernelRegInfoNeon, self).__init__(NUM_GPRS-5 if self.target_os == "windows_arm64ec" else NUM_GPRS,
                                                      BASE_GPR, NUM_VREGS, 4 if self.is_complex else 2)

    def valid_kernel(self, m, n, k, trans):
        return self.fits_in_gprs(m, n, k, trans) and self.fits_in_vectors(m, n, k, trans)

    def fits_in_gprs(self, m, n, k, trans):
        ncola = m if trans.transa else k
        ncolb = k if trans.transb else n
        reg_max = max(ncola + ncolb, n) + self.base_gpreg

        if self.target_os == "windows_arm64ec":
            # don't use x13
            if reg_max >= 13:
                reg_max += 1
            # don't use x14
            if reg_max >= 14:
                reg_max += 1

        # don't use x18
        if reg_max >= 18:
            reg_max += 1

        if self.target_os == "windows_arm64ec":
            # don't use x23
            if reg_max >= 23:
                reg_max += 1
            # don't use x24
            if reg_max >= 24:
                reg_max += 1
            # don't use x28
            if reg_max >= 28:
                reg_max += 1

        # don't use x29
        if reg_max >= 29:
            reg_max += 1

        return reg_max < self.num_gpreg

    def fits_in_vectors(self, m, n, k, trans):
        # We need a vector for each column in A. We don't need to check transpose options,
        # as matrix A is always loaded in the same manner, regardless of transpose options.
        # It will always be the case that A is contiguous in the m direction
        num_a_col_vects = math.ceil(float(m) / self.dtype.max_neon_lanes)
        if self.dtype.is_complex:
            num_a_col_vects *= 2
        num_a_vects = num_a_col_vects * k
        num_result_vects = num_a_col_vects * n  # = num_c_vregs

        # B is n-contiguous if transb and k-contiguous otherwise
        b_ncon = math.ceil(float(n) / self.dtype.max_neon_lanes) * k
        b_kcon = math.ceil(float(k) / self.dtype.max_neon_lanes) * n
        num_b_vects = b_ncon if trans.transb else b_kcon

        # Can we hold A, B and the multiplication result in vectors?
        enough_for_a_mult_b = (
            num_a_vects + num_b_vects +
            num_result_vects + self.base_vreg < self.num_vreg)

        # Can we hold 2 * num_of_result_vectors? Due to the multiplication of
        # the result with C (result.shape == C.shape)
        enough_for_result_mult_c = (
            self.num_vreg - self.base_vreg >= 2 * num_result_vects)
        return enough_for_a_mult_b and enough_for_result_mult_c
    # End of function fits_in_vectors

    def generate_valid_kernel_dims(self, trans):
        ret_list = [(m, n, k)
            for m in self._get_valid_m(trans)
            for n in self._get_valid_n(m, trans)
            for k in self._get_valid_k(m, n, trans)]
        return ret_list

    def _get_valid_m(self, trans):
        # The magic numbers in this function and the others comes from
        # experiments run with a full parameter sweep for the initial
        # implementation of the atomic GEMM kernels
        return [m for m in range(1, 58) if self.valid_kernel(m, 1, 1, trans)]

    def _get_valid_n(self, m, trans):
        return [n for n in range(1, 17) if self.valid_kernel(m, n, 1, trans)]

    def _get_valid_k(self, m, n, trans):
        return [k for k in range(1, 23) if self.valid_kernel(m, n, k, trans)]

class AtomicKernelParamsNeon(AtomicGEMMKernelParams):
    def __init__(self, m, n, k, trans, dtype, target_os):
        reg_info = AtomicKernelRegInfoNeon(dtype, target_os)
        if not reg_info.valid_kernel(m, n, k, trans):
            raise AtomicKernelException(m, n, k, trans, dtype)
        self.m = m
        self.n = n
        self.k = k
        self.trans = trans
        self.dtype = dtype
        self.target_os = target_os

    def __hash__(self):
        return (self.m | (self.n << 5) | (self.k << 10) | (self.dtype.value << 15) | (int(self.trans) << 17))

    def __eq__(self, other):
        return (self.m == other.m and
                self.n == other.n and
                self.k == other.k and
                self.trans == other.trans and
                self.dtype == other.dtype and
                self.target_os == other.target_os)

    @property
    def routine_name(self):
        return "kernel_{}gemm_{}_{}_{}_{}".format(
            self.dtype.name, self.m, self.n, self.k, self.trans.fullname)
# End of class AtomicKernelParamsNeon

class AtomicGEMMKernelWriterNeon(AtomicGEMMKernelWriter):
    def __init__(self, f, params):
        super(AtomicGEMMKernelWriterNeon, self).__init__(f, params, None)

    def allocate_vectors(self, t, m, n, k, dtype):
        """Allocate Neon registers for computing *GEMM

        Only single and double types are currently supported. The register
        allocation works as follows.

        Matrix A is always loaded so that the columns of the matrix are
        contiguous. This means that the data in the result matrix C is stored in
        column-contiguous order as well.

        Matrix B is loaded in contiguous order regardless of transpose options.
        This means that the matrix is column major in the case that transb is
        false, and row major in the case that transb is true.

        Below are a couple of examples of the register allocation for A and B with
        varying parameters.

        m=2 n=1 k=3 transpose=NN dtype=double : dim(A)=(m,k), dim(B)=(k,n)

                k
        |----+----+----|
        | v1 | v2 | v3 |
        m |----+----+----|
        | v1 | v2 | v3 |
        |----+----+----|

            n
        |----|
        | v4 |
        |----|
        k | v4 |
        |----|
        | v5 |
        |----|

        m=1 n=2 k=3 transpose=TT dtype=double : dim(A)=(k,m), dim(B)=(n,k)

                k
        |----+----+----|
        m | v1 | v2 | v3 |
        |----+----+----|

                k
        |----+----+----|
        | v4 | v5 | v6 |
        n |----+----+----|
        | v4 | v5 | v6 |
        |----+----+----|

        Both examples use double (64-bit) so a single Neon vector (128-bit) can fit
        two elements per vector register.

        This function allocates registers to each matrix, determines vector lanes,
        and returns a, b, c and vect_lanes. Where a, b and c are n-dimensional
        numpy arrays reflecting the shape of the input matrices, with the caveat
        transa has no effect on the dimensionality of A as discussed above. The
        arrays contain vector indexes, e.g. 1, 2, 3 corresponding to v1, v2, v3.

        vect_lanes is a mapping of vector registers to lanes used. e.g.

            In [1]: vect_lanes[4]
            Out[1]: 2

        """
        # A is always m-contiguous
        a = np.zeros(shape=(m, k), dtype=int)
        b = np.zeros(shape=(n, k) if t.transb else (k, n), dtype=int)
        c = np.zeros(shape=(m, n), dtype=int)

        max_neon_lanes = dtype.max_neon_lanes
        # =====================================================
        # Make matrices A, B, C in which value of <matrix>[i][j] = v,
        # where v is the vector register in which that value is stored.
        vreg = dtype.base_vreg
        lane = 0

        # A
        for j in range(k):
            for i in range(m):
                a[i, j] = vreg
                lane += 1
                if lane == max_neon_lanes or i + 1 >= m:
                    vreg += 1
                    lane = 0

        # B
        if t.transb:
            for j in range(k):
                for i in range(n):
                    b[i, j] = vreg
                    lane += 1
                    if lane == max_neon_lanes or i + 1 >= n:
                        vreg += 1
                        lane = 0
        else:
            for j in range(n):
                for i in range(k):
                    b[i, j] = vreg
                    lane += 1
                    if lane == max_neon_lanes or i + 1 >= k:
                        vreg += 1
                        lane = 0

        # C
        for j in range(n):
            for i in range(m):
                c[i, j] = vreg
                lane += 1
                if lane == max_neon_lanes or i + 1 >= m:
                    vreg += 1
                    lane = 0

        # =====================================================
        # vect_lanes stores a mapping of vectors to the number
        # of lanes used.
        vect_lanes = np.zeros(vreg, dtype=int)
        vreg = dtype.base_vreg

        for j in range(k):
            for i in range(0, m, max_neon_lanes):
                vect_lanes[vreg] = min(m - i, max_neon_lanes)
                vreg += 1

        if not t.transb:
            for i in range(n):
                for j in range(0, k, max_neon_lanes):
                    vect_lanes[vreg] = min(k - j, max_neon_lanes)
                    vreg += 1
        else:
            for j in range(k):
                for i in range(0, n, max_neon_lanes):
                    vect_lanes[vreg] = min(n - i, max_neon_lanes)
                    vreg += 1

        for j in range(n):
            for i in range(0, m, max_neon_lanes):
                vect_lanes[vreg] = min(m - i, max_neon_lanes)
                vreg += 1

        for array in (a, b, c):
            for vector, count in zip(*np.unique(array, return_counts=True)):
                assert vect_lanes[vector] == count

        return a, b, c, vect_lanes

    def complex_allocate_vectors(self, t, m, n, k, dtype):
        a_real = np.zeros(shape=(m, k), dtype=int)
        a_imag = np.zeros(shape=(m, k), dtype=int)
        b = np.zeros(shape=(n, k) if t.transb else (k, n), dtype=int)
        result_real = np.zeros(shape=(m, n), dtype=int)
        result_imag = np.zeros(shape=(m, n), dtype=int)

        lane = 0
        vreg = dtype.base_vreg

        # A
        for j in range(k):
            for i in range(m):
                a_real[i, j] = vreg
                a_imag[i, j] = vreg + 1
                lane += 1
                if lane == dtype.real_max_neon_lanes or i + 1 >= m:
                    vreg += 2
                    lane = 0

        # B
        lane = 0
        vreg = np.max(a_imag) + 1
        if t.transb:
            for j in range(k):
                for i in range(n):
                    b[i, j] = vreg
                    lane += 1
                    if lane == dtype.max_neon_lanes or i + 1 >= n:
                        vreg += 1
                        lane = 0
        else:
            for j in range(n):
                for i in range(k):
                    b[i, j] = vreg
                    lane += 2
                    if lane == dtype.real_max_neon_lanes or i + 1 >= k:
                        vreg += 1
                        lane = 0

        # Load result from last vector downwards
        lane = 0
        vreg = NUM_VREGS - 1
        for j in range(n):
            for i in range(m):
                result_real[i, j] = vreg - 1
                result_imag[i, j] = vreg
                lane += 1
                if lane == dtype.real_max_neon_lanes or i + 1 >= m:
                    vreg -= 2
                    lane = 0

        vect_lanes = [0] * NUM_VREGS
        for j in range(k):
            for i in range(0, m):
                vect_lanes[a_real[i, j]] = np.count_nonzero(a_real == a_real[i, j])
                vect_lanes[a_imag[i, j]] = np.count_nonzero(a_imag == a_imag[i, j])

        if not t.transb:
            for j in range(n):
                for i in range(0, k):
                    vect_lanes[b[i, j]] = np.count_nonzero(b == b[i, j]) * 2
        else:
            for j in range(k):
                for i in range(0, n):
                    vect_lanes[b[i, j]] = np.count_nonzero(b == b[i, j]) * 2

        for j in range(n):
            for i in range(0, m):
                vect_lanes[result_real[i, j]] = np.count_nonzero(
                    result_real == result_real[i, j]
                )
                vect_lanes[result_imag[i, j]] = np.count_nonzero(
                    result_imag == result_imag[i, j]
                )

        return a_real, a_imag, b, result_real, result_imag, vect_lanes

    def get_constant_limits(self, t, k, n, m):
        a_consts = ['A{}'.format(a) for a in range(m if t.transa else k)]
        b_consts = ['B{}'.format(b) for b in range(k if t.transb else n)]
        c_consts = ['C{}'.format(c) for c in range(n)]
        return a_consts + b_consts, c_consts

    def generate_column_constants(self, f, t, k, n, m, undef=False):
        ab_consts, c_consts = self.get_constant_limits(t, k, n, m)
        gpr_num = BASE_GPR
        for constant in ab_consts:
            gpr_num += 1
            # don't use x18 or x29
            if gpr_num == 18 or gpr_num == 29:
                gpr_num += 1
            # for Windows Arm64EC don't use x13, x14, x23, x24 or x28
            if self.params.target_os == "windows_arm64ec" and (gpr_num == 13 or gpr_num == 23 or gpr_num == 28):
                gpr_num += 2
            if undef:
                f.write("#undef {}\n".format(constant))
            else:
                f.write("#define {} x{}\n".format(constant, gpr_num))
        f.write('\n')
        c_gpr_num = BASE_GPR
        for constant in c_consts:
            c_gpr_num += 1
            # don't use x18 or x29
            if c_gpr_num == 18 or c_gpr_num == 29:
                c_gpr_num += 1
            # for Windows Arm64EC don't use x13, x14, x23, x24 or x28
            if self.params.target_os == "windows_arm64ec" and (c_gpr_num == 13 or c_gpr_num == 23 or c_gpr_num == 28):
                c_gpr_num += 2
            if undef:
                f.write("#undef {}\n".format(constant))
            else:
                f.write("#define {} x{}\n".format(constant, c_gpr_num))

    def generate_function_header(self, f, kernel_name):
        if self.params.target_os == "linux":
            # function header
            wi(f, ".global\t{}".format(kernel_name))
            wi(f, ".type\t{}, %function".format(kernel_name))
            # Align function entry on 32-byte boundary, optimal when cache lines
            # are also 32-bytes.
            wi(f, ".p2align 5")
            f.write('\n')
            wl(f, "{}".format(kernel_name))
        elif self.params.target_os == "mac":
            # function header
            wi(f, ".global\t_{}".format(kernel_name))
            # Align function entry on 4-byte boundary, optimal when cache lines
            # are also 32-bytes.
            wi(f, ".p2align 2")
            f.write('\n')
            wl(f, "_{}".format(kernel_name))
        # For "windows" or "windows_arm64ec" targets...
        elif self.params.target_os in ("windows", "windows_arm64ec"):
            # If we're building for Arm64EC prepend # to kernel_name
            # and enclose in double-quotes
            decorated_name = "\"#{}\"".format(kernel_name) if self.params.target_os == "windows_arm64ec" else kernel_name
            # function header
            wi(f, ".global\t{}".format(decorated_name))
            wi(f, ".def\t{};".format(decorated_name))
            wi(f, ".scl 2;")
            wi(f, ".type 32;")
            wi(f, ".endef")
            f.write('\n')
            wl(f, "{}".format(decorated_name))
        else:
            raise Exception("Unrecognized operating system target " + self.params.target_os + " in Neon atomic kernel generation")

    def generate_save_or_restore_registers(self, f, reg_max, vreg_max, restore=False):
        """Save or restore registers to stack.

        Procedure Call Standard for the 64-bit ARM Architecture (AAPCS64) [1]
        states:

            For GPRs:
            * A subroutine invocation must preserve the contents of the registers
            x19-x29 and SP.
            * We do not use x18. As per the PCS: "...portable hand-coded assembler
            should avoid it [x18] entirely. It should not be assumed that treating
            the register as callee-saved will be sufficient to satisfy the
            requirements of the platform."
            * We do not use x29 (the Frame Pointer). As per the PCS: "Conforming
            code shall construct a linked list of stack-frames. Each frame shall
            link to the frame of its caller by means of a frame record of two 64-bit
            values on the stack. The frame record for the innermost frame (belonging
            to the most recent routine invocation) shall be pointed to by the Frame
            Pointer register (FP).
            For SIMD (neon):
            * Registers v8-v15 must be preserved by a callee across subroutine
            calls; the remaining registers (v0-v6, v16-v31) do not need to be
            preserved (or should be preserved by the caller). Additionally, only
            the bottom 64-bits of each value stored in v8-v15 need to be
            preserved; it is the responsibility of the caller to preserve larger
            values.

        The Link Register (x30) is also preserved on the stack, allowing it to be
        used as a GPR [2].

        In addition for Windows Arm64EC ABI we do not use x13, x14, x23, x24 and
        x28:

            We can also see how the registers x13, x14, x23, x24, x28, v16-v31
            have no representation and, thus, cannot be used in Arm64EC [3].

        Note that the restriction of V registers has been relaxed in the versions
        of the Arm64EC ABI that we support.

        [1] https://github.com/ARM-software/abi-aa/tree/main/aapcs64
        [2] https://developer.arm.com/documentation/dui0801/latest/
        [3] https://learn.microsoft.com/en-us/windows/arm/arm64ec-abi

        """
        assert 0 <= reg_max < NUM_GPRS-5 if self.params.target_os == "windows_arm64ec" else NUM_GPRS, (
            'GPRs exhausted, required: {}'.format(reg_max)
        )
        assert 0 <= vreg_max < NUM_VREGS, (
            'Vector registers exhausted, required: {}'.format(vreg_max)
        )

        gpr_base = 19
        vreg_base = 8

        total_gprs_to_preserve = min(max(0, reg_max - (gpr_base - 1)), 12)
        total_vregs_to_preserve = min(max(0, vreg_max - (vreg_base - 1)), 8)
        # + 2 here to allow for storing x29 and x30
        stack_total = 8 * (total_gprs_to_preserve + total_vregs_to_preserve + 2)

        # 16-byte align stack
        stack_total -= stack_total % - 16

        # set up the frame record
        if not restore:
            wi(f, "// vec spills: {}, gpreg spills: {}".format(total_vregs_to_preserve, total_gprs_to_preserve))
            wi(f, "stp x29, x30, [sp, #-{}]!".format(stack_total))
            wi(f, "mov x29, sp")

        # store the registers we need to, besides x29 and x30
        if reg_max > gpr_base - 1 or vreg_max > vreg_base - 1:
            if restore:
                wi(f, "// restore registers from stack")
                op = 'ld'
            else:
                wi(f, "// save current registers state")
                op = 'st'

            offset = 16
            if total_gprs_to_preserve % 2 != 0:
                wi(f, "{}r x{}, [sp, #{}]".format(op, gpr_base, offset))
                offset += 8
                gpr_base += 1
                total_gprs_to_preserve -= 1

            # Preserve remaining general purpose registers in pairs (ldp/stp)
            for i in range(gpr_base, gpr_base + total_gprs_to_preserve, 2):
                wi(f, "{}p x{}, x{}, [sp, #{}]".format(op, i, i+1, offset))
                offset += 16

            if total_vregs_to_preserve % 2 != 0:
                wi(f, "{}r d{}, [sp, #{}]".format(op, vreg_base, offset))
                offset += 8
                vreg_base += 1
                total_vregs_to_preserve -= 1

            # Preserve remaining vector registers in pairs (ldp/stp)
            for i in range(vreg_base, vreg_base + total_vregs_to_preserve, 2):
                wi(f, "{}p d{}, d{}, [sp, #{}]".format(op, i, i+1, offset))
                offset += 16

        if restore:
            # Restore stack pointer
            wi(f, "ldp x29, x30, [sp], #{}".format(stack_total))

    def generate_a_b_pointers(self, f, t, m, n, k):
        f.write('\n')
        wi(f, "// pointers to A {}".format("rows" if t.transa else "columns"))
        wi(f, "mov A0, a_ptr")
        for i in range(1, m if t.transa else k):
            wi(f, "add A{}, A{}, lda".format(i, i-1))

        f.write('\n')
        wi(f, "// pointers to B {}".format("rows" if t.transb else "columns"))
        wi(f, "mov B0, b_ptr")
        for i in range(1, k if t.transb else n):
            wi(f, "add B{}, B{}, ldb".format(i, i-1))

    def generate_c_pointers(self, f, n):
        f.write('\n')
        wi(f, "// pointers to C columns")
        wi(f, "mov C0, c_ptr")
        for i in range(1, n):
            wi(f, "add C{}, C{}, ldc".format(i, i-1))

    def clear_result_vectors(self, f, result_vectors):
        wi(f, "// clear vectors used for result")
        for result_vector in np.unique(result_vectors):
            wi(f, "dup v{}.2d, xzr".format(result_vector))
        f.write('\n')


    def generate_alpha_check(self, f, kernel_name, dtype):
        """If alpha is zero, then we can skip alpha*A*B as the result will be a 0
        matrix, which we have already initialized
        """
        wi(f, "// if alpha is 0 we can skip (alpha * A * B)")
        wi(f, "fcmp {}{}, {}".format(dtype.spec, ALPHA_REAL_REGISTER, FPZERO))
        if dtype in (Type.c, Type.z):
            # If type is complex we need to check real and imaginary parts
            wi(f, "b.ne .L{}_multiply_axb".format(kernel_name))
            wi(f, "fcmp {}{}, {}".format(dtype.spec, ALPHA_IMAG_REGISTER, FPZERO))
            wi(f, "b.eq .L{}_skip_axb".format(kernel_name))
        else:
            wi(f, "b.eq .L{}_check_beta".format(kernel_name))
        f.write('\n')


    def generate_a_load(self, f, t, k, m, dtype, vect_lanes, a):
        wi(f, "// load A")
        if t.transa:
            for i in range(0, m, dtype.max_neon_lanes):
                for j in range(k):
                    for x in range(vect_lanes[a[i, j]]):
                        wi(f, "ld1 {{v{}.{}}}[{}], [A{}], #{}".format(
                            a[i+x, j], dtype.spec, x, i+x, dtype.width))
        else:
            for j in range(k):
                for i in range(0, m, dtype.max_neon_lanes):
                    if vect_lanes[a[i, j]] == dtype.max_neon_lanes:
                        # Four singles or two doubles
                        wi(f, "ldr q{}, [A{}], #16".format(a[i, j], j))
                    elif vect_lanes[a[i, j]] == dtype.max_neon_lanes // 2:
                        # Two singles or one double
                        wi(f, "ldr d{}, [A{}], #8".format(a[i, j], j))
                    elif vect_lanes[a[i, j]] == dtype.max_neon_lanes // 4:
                        # One single
                        wi(f, "ldr s{}, [A{}], #4".format(a[i, j], j))
                    else:
                        # Three singles
                        assert dtype == Type.s, 'Expected three singles'
                        wi(f, "ldr d{}, [A{}], #8".format(a[i, j], j))
                        wi(f, "ld1 {{v{}.s}}[2], [A{}]".format(a[i, j], j))
        f.write('\n')


    def complex_generate_a_load(self, f, t, k, m, dtype, vect_lanes, a_real, a_imag):
        wi(f, "// load A")
        if t.transa:
            for j in range(k):
                for i in range(0, m):
                    real_vector = ("v{0}.{1}".format(
                        a_real[i, j], dtype.spec))
                    imaginary_vector = ("v{0}.{1}".format(
                        a_imag[i, j], dtype.spec))
                    wi(f, "ld2 {{{}, {}}}[{}], [A{}], #{}".format(
                        real_vector, imaginary_vector,
                        i % dtype.real_max_neon_lanes, i, dtype.width))
        else:
            for j in range(k):
                for i in range(0, m, dtype.real_max_neon_lanes):
                    if vect_lanes[a_real[i, j]] == dtype.real_max_neon_lanes:
                        wi(
                            f,
                            "ld2 {{v{2}.{0}{1}, v{3}.{0}{1}}}, [A{4}], #32".format(
                                dtype.real_max_neon_lanes, dtype.spec,
                                a_real[i, j], a_imag[i, j], j
                            )
                        )
                    elif vect_lanes[a_real[i, j]] == dtype.real_max_neon_lanes // 2:
                        vectors = "{{v{}.d, v{}.d}}[0]".format(
                            a_real[i, j], a_imag[i, j])
                        if dtype == Type.c:
                            vectors = "{{v{}.2s, v{}.2s}}".format(
                                a_real[i, j], a_imag[i, j])
                        wi(f, "ld2 {}, [A{}], #16".format(vectors, j))
                    elif vect_lanes[a_real[i, j]] == dtype.real_max_neon_lanes // 4:
                        # One single
                        wi(f, "ld2 {{v{}.s, v{}.s}}[0], [A{}], #8".format(
                            a_real[i, j], a_imag[i, j], j))
                    else:
                        # Three singles
                        wi(f, "ld2 {{v{}.2s, v{}.2s}}, [A{}], #16".format(
                            a_real[i, j], a_imag[i, j], j))
                        wi(f, "ld2 {{v{}.s, v{}.s}}[2], [A{}]".format(
                            a_real[i, j], a_imag[i, j], j))
        f.write('\n')


    def generate_b_load(self, f, t, k, n, dtype, vect_lanes, b):
        wi(f, "// load B")
        if t.transb:
            for i in range(k):
                for j in range(0, n, dtype.max_neon_lanes):
                    if vect_lanes[b[j, i]] == dtype.max_neon_lanes:
                        # Four singles or two doubles
                        wi(f, "ldr q{}, [B{}], #16".format(b[j, i], i))
                    elif vect_lanes[b[j, i]] == dtype.max_neon_lanes // 2:
                        # Two singles or one double
                        wi(f, "ldr d{}, [B{}], #8".format(b[j, i], i))
                    elif vect_lanes[b[j, i]] == dtype.max_neon_lanes // 4:
                        # One single
                        wi(f, "ldr s{}, [B{}], #4".format(b[j, i], i))
                    else:
                        # Three singles
                        # assert dtype == Type.s, 'Expected three singles'
                        wi(f, "ldr d{}, [B{}], #8".format(b[j, i], i))
                        wi(f, "ld1 {{v{}.s}}[2], [B{}]".format(b[j, i], i))
        else:
            for j in range(n):
                for i in range(0, k, dtype.max_neon_lanes):
                    if vect_lanes[b[i, j]] == dtype.max_neon_lanes:
                        # Four singles or two doubles
                        wi(f, "ldr q{}, [B{}], #16".format(b[i, j], j))
                    else:
                        for x in range(vect_lanes[b[i, j]]):
                            wi(f, "ld1 {{v{}.{}}}[{}], [B{}], #{}".format(
                                b[i+x, j], dtype.spec, x, j, dtype.width))
        f.write('\n')


    def complex_generate_b_load(self, f, t, k, n, dtype, vect_lanes, b):
        wi(f, "// load B")
        if t.transb:
            for i in range(k):
                for j in range(0, n, dtype.max_neon_lanes):
                    if vect_lanes[b[j, i]] == dtype.real_max_neon_lanes:
                        wi(f, "ldr q{}, [B{}], #16".format(b[j, i], i))
                    elif vect_lanes[b[j, i]] == dtype.real_max_neon_lanes // 2:
                        wi(f, "ldr d{}, [B{}], #8".format(b[j, i], i))
        else:
            for j in range(n):
                for i in range(0, k, dtype.max_neon_lanes):
                    if vect_lanes[b[i, j]] == dtype.real_max_neon_lanes:
                        # Four singles or two doubles
                        wi(f, "ldr q{}, [B{}], #16".format(b[i, j], j))
                    else:
                        for x in range(dtype.max_neon_lanes // 2):
                            wi(f, "ld1 {{v{}.{}}}[{}], [B{}], #8".format(
                                b[i+x, j], 'd', x, j))
        f.write('\n')


    def generate_a_mult_b(self, f, k, m, n, dtype, a, b, t, c):
        wi(f, "// calculate (a * b)")
        for h in range(k):
            for i in range(0, m, dtype.max_neon_lanes):
                for j in range(n):
                    string_a = 'v{}.{}{}'.format(
                        a[i, h], dtype.max_neon_lanes, dtype.spec)
                    if t.transb:
                        string_b = 'v{}.{}[{}]'.format(
                            b[j, h], dtype.spec,
                            j % dtype.max_neon_lanes
                        )
                    else:
                        string_b = 'v{}.{}[{}]'.format(
                            b[h, j], dtype.spec,
                            h % dtype.max_neon_lanes
                        )
                    string_c = ('v{}.{}{}'.format(
                        c[i, j], dtype.max_neon_lanes, dtype.spec))
                    wi(f, "fmla {}, {}, {}".format(string_c, string_a, string_b))
        f.write('\n')


    def complex_b_lanes(self, n, k, t, dtype):
        b_real_lane = np.zeros(shape=(n, k) if t.transb else (k, n), dtype=int)
        b_imag_lane = np.zeros(shape=(n, k) if t.transb else (k, n), dtype=int)

        for k_ in range(k):
            for n_ in range(n):
                b_row, b_col = (n_, k_) if t.transb else (k_, n_)
                b_real_lane[b_row, b_col], b_imag_lane[b_row, b_col] = (0, 1)
                if dtype == Type.c:
                    b_real_lane[b_row, b_col], b_imag_lane[b_row, b_col] = (
                        (2, 3) if b_row % 2 else (0, 1))
        return b_real_lane, b_imag_lane


    def complex_generate_a_mult_b(self, f, k, m, n, t, a_real, a_imag,
                                b, result_real, result_imag, dtype):
        wi(f, "// calculate (a * b)")
        b_real_lanes, b_imag_lanes = self.complex_b_lanes(n, k, t, dtype)
        for step in range(2):
            for k_ in range(k):
                for m_ in range(0, m, dtype.real_max_neon_lanes):
                    for n_ in range(n):
                        b_row, b_col = (n_, k_) if t.transb else (k_, n_)
                        b_real_string = "v{}.{}[{}]".format(
                            b[b_row, b_col], dtype.spec,
                            b_real_lanes[b_row, b_col])
                        b_imag_string = "v{}.{}[{}]".format(
                            b[b_row, b_col], dtype.spec,
                            b_imag_lanes[b_row, b_col])

                        a_real_string = 'v{}.{}{}'.format(
                            a_real[m_, k_], dtype.real_max_neon_lanes, dtype.spec)
                        a_imag_string = 'v{}.{}{}'.format(
                            a_imag[m_, k_], dtype.real_max_neon_lanes, dtype.spec)

                        result_real_string = 'v{}.{}{}'.format(
                            result_real[m_, n_],
                            dtype.real_max_neon_lanes, dtype.spec)
                        result_imag_string = 'v{}.{}{}'.format(
                            result_imag[m_, n_],
                            dtype.real_max_neon_lanes, dtype.spec)

                        # real result
                        if step == 0:
                            wi(f, 'fmla {}, {}, {}'.format(
                                result_real_string, a_real_string, b_real_string))
                        else:
                            wi(f, 'fml{} {}, {}, {}'.format(
                                'a' if t.conja != t.conjb else 's',
                                result_real_string, a_imag_string, b_imag_string))

                        # imaginary result
                        if step == 0:
                            wi(f, 'fml{} {}, {}, {}'.format(
                                's' if t.conjb else 'a',
                                result_imag_string, a_real_string, b_imag_string))
                        else:
                            wi(f, 'fml{} {}, {}, {}'.format(
                                's' if t.conja else 'a',
                                result_imag_string, a_imag_string, b_real_string))
        f.write('\n')


    def generate_res_mult_alpha(self, f, m, n, c, dtype):
        wi(f, "// multiply with alpha")
        for i in range(0, m, dtype.max_neon_lanes):
            for j in range(n):
                string_c = 'v{}.{}{}'.format(
                    c[i, j], dtype.max_neon_lanes, dtype.spec
                )
                wi(f, "fmul {}, {}, v0.{}[0]".format(
                    string_c, string_c, dtype.spec))
        f.write('\n')


    def complex_generate_mult_alpha(self, f, m, n, result_real, result_imag,
                                    int_res_real, int_res_imag, dtype):
        wi(f, "// multiply with alpha")
        real_alpha_string = 'v{}.{}[0]'.format(ALPHA_REAL_REGISTER, dtype.spec)
        imag_alpha_string = 'v{}.{}[0]'.format(ALPHA_IMAG_REGISTER, dtype.spec)
        for step in range(2):
            for i in range(0, m, dtype.real_max_neon_lanes):
                for j in range(n):
                    result_real_string = 'v{}.{}{}'.format(
                        result_real[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    result_imag_string = 'v{}.{}{}'.format(
                        result_imag[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    int_res_real_string = 'v{}.{}{}'.format(
                        int_res_real[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    int_res_imag_string = 'v{}.{}{}'.format(
                        int_res_imag[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    if step == 0:
                        wi(f, 'fmul {}, {}, {}'.format(
                            int_res_real_string, result_real_string,
                            real_alpha_string))
                        wi(f, 'fmul {}, {}, {}'.format(
                            int_res_imag_string, result_real_string,
                            imag_alpha_string))
                    else:
                        wi(f, 'fmls {}, {}, {}'.format(
                            int_res_real_string, result_imag_string,
                            imag_alpha_string))
                        wi(f, 'fmla {}, {}, {}'.format(
                            int_res_imag_string, result_imag_string,
                            real_alpha_string))
        f.write('\n')


    def generate_beta_check(self, f, kernel_name, dtype):
        wi(f, "// if beta is 0 we can skip (C * beta)")
        if dtype in (Type.c, Type.z):
            # If type is complex we need to check real and imaginary parts
            wi(f, "fcmp {}{}, {}".format(dtype.spec, BETA_REAL_REGISTER, FPZERO))
            wi(f, "b.ne .L{}_multiply_c".format(kernel_name))
            wi(f, "fcmp {}{}, {}".format(dtype.spec, BETA_IMAG_REGISTER, FPZERO))
        else:
            wi(f, "fcmp {}{}, {}".format(dtype.spec, BETA_REGISTER, FPZERO))
        wi(f, "b.eq .L{}_store_c".format(kernel_name))
        f.write('\n')


    def allocate_c_vectors(self, result_vectors, m, n, dtype):
        # We can load C into vectors used for a and b
        # And also any vectors not used after the result
        available_vectors = np.append(
            np.arange(dtype.base_vreg, np.min(result_vectors)),
            np.arange(np.max(result_vectors)+1, NUM_VREGS)
        )
        c_v = np.zeros(shape=(m, n), dtype=int)

        index = 0
        lane = 0
        for j in range(n):
            for i in range(m):
                c_v[i, j] = available_vectors[index]
                lane += 1
                if lane == dtype.max_neon_lanes or i + 1 >= m:
                    index += 1
                    lane = 0
        return c_v


    def complex_allocate_c_vectors(self, result_real, result_imag,
                                   vect_lanes, m, n, dtype):
        # Create a list of all vectors excluding vectors used for the result
        available_vectors = np.append(
            np.arange(dtype.base_vreg, np.min(result_real)),
            np.arange(np.max(result_imag)+1, NUM_VREGS)
        )

        c_real = np.zeros(shape=(m, n), dtype=int)
        c_imag = np.zeros(shape=(m, n), dtype=int)

        index = 0
        lane = 0
        for j in range(n):
            for i in range(m):
                c_real[i, j] = available_vectors[index]
                c_imag[i, j] = available_vectors[index+1]
                lane += 1
                if lane == dtype.real_max_neon_lanes or i + 1 >= m:
                    index += 2
                    lane = 0

        for j in range(n):
            for i in range(0, m):
                vect_lanes[c_real[i, j]] = np.count_nonzero(c_real == c_real[i, j])
                vect_lanes[c_imag[i, j]] = np.count_nonzero(c_imag == c_imag[i, j])

        return c_real, c_imag, vect_lanes


    def generate_c_load(self, f, c_v, m, n, dtype, vect_lanes, result):
        # Make sure that we have pointers to the columns
        self.generate_c_pointers(f, n)
        f.write('\n')
        wi(f, "// load C")
        for i in range(0, m, dtype.max_neon_lanes):
            for j in range(n):
                if vect_lanes[result[i, j]] == dtype.max_neon_lanes:
                    # Four singles or two doubles
                    wi(f, "ldr q{}, [C{}], #16".format(c_v[i, j], j))
                elif vect_lanes[result[i, j]] == dtype.max_neon_lanes // 2:
                    # Two singles or one double
                    wi(f, "ldr d{}, [C{}], #8".format(c_v[i, j], j))
                elif vect_lanes[result[i, j]] == dtype.max_neon_lanes // 4:
                    # One single
                    wi(f, "ldr s{}, [C{}], #4".format(c_v[i, j], j))
                else:
                    # Three singles
                    assert dtype == Type.s, 'Expected three singles'
                    wi(f, "ldr d{}, [C{}], #8".format(c_v[i, j], j))
                    wi(f, "ld1 {{v{}.s}}[2], [C{}]".format(c_v[i, j], j))
        f.write('\n')


    def complex_generate_c_load(self, f, c_real, c_imag, m, n,
                                vect_lanes, result, dtype):
        self.generate_c_pointers(f, n)
        f.write('\n')
        wi(f, "// load C")
        for j in range(n):
            for i in range(0, m, dtype.real_max_neon_lanes):
                if vect_lanes[result[i, j]] == dtype.real_max_neon_lanes:
                    wi(
                        f,
                        "ld2 {{v{2}.{0}{1}, v{3}.{0}{1}}}, [C{4}], #32".format(
                            dtype.real_max_neon_lanes, dtype.spec,
                            c_real[i, j], c_imag[i, j], j
                        )
                    )
                elif vect_lanes[result[i, j]] == dtype.real_max_neon_lanes // 2:
                    vectors = "{{v{}.d, v{}.d}}[0]".format(
                        c_real[i, j], c_imag[i, j])
                    if dtype == Type.c:
                        vectors = "{{v{}.2s, v{}.2s}}".format(
                            c_real[i, j], c_imag[i, j])
                    wi(f, "ld2 {}, [C{}], #16".format(vectors, j))
                elif vect_lanes[result[i, j]] == dtype.real_max_neon_lanes // 4:
                    # One single
                    wi(f, "ld2 {{v{}.s, v{}.s}}[0], [C{}], #8".format(
                        c_real[i, j], c_imag[i, j], j))
                else:
                    # Three singles
                    wi(f, "ld2 {{v{}.2s, v{}.2s}}, [C{}], #16".format(
                        c_real[i, j], c_imag[i, j], j))
                    wi(f, "ld2 {{v{}.s, v{}.s}}[2], [C{}]".format(
                        c_real[i, j], c_imag[i, j], j))
        f.write('\n')


    def generate_result_plus_c_mult_beta(self, f, c_v, result, m, n, dtype):
        wi(f, "// calculate (result + (c * beta))")
        for i in range(0, m, dtype.max_neon_lanes):
            for j in range(n):
                string_c = 'v{}.{}{}'.format(
                    c_v[i, j], dtype.max_neon_lanes, dtype.spec
                )
                result_string = 'v{}.{}{}'.format(
                    result[i, j], dtype.max_neon_lanes, dtype.spec
                )
                wi(f, "fmla {}, {}, v1.{}[0]".format(
                    result_string, string_c, dtype.spec))
        f.write('\n')


    def complex_generate_result_plus_c_mult_beta(self, f, c_real, c_imag, result_real,
                                                result_imag, m, n, dtype):
        wi(f, "// calculate (result + (c * beta))")
        real_beta_string = 'v{}.{}[0]'.format(BETA_REAL_REGISTER, dtype.spec)
        imag_beta_string = 'v{}.{}[0]'.format(BETA_IMAG_REGISTER, dtype.spec)

        for step in range(2):
            for i in range(0, m, dtype.real_max_neon_lanes):
                for j in range(n):
                    real_string_c = 'v{}.{}{}'.format(
                        c_real[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    imag_string_c = 'v{}.{}{}'.format(
                        c_imag[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    real_result_string = 'v{}.{}{}'.format(
                        result_real[i, j], dtype.real_max_neon_lanes, dtype.spec)
                    imag_result_string = 'v{}.{}{}'.format(
                        result_imag[i, j], dtype.real_max_neon_lanes, dtype.spec)

                    if step == 0:
                        wi(f, "fmla {}, {}, {}".format(
                            real_result_string, real_string_c, real_beta_string))
                        wi(f, "fmla {}, {}, {}".format(
                            imag_result_string, real_string_c, imag_beta_string))
                    else:
                        wi(f, "fmls {}, {}, {}".format(
                            real_result_string, imag_string_c, imag_beta_string))
                        wi(f, "fmla {}, {}, {}".format(
                            imag_result_string, imag_string_c, real_beta_string))
        f.write('\n')


    def generate_store_result(self, f, m, dtype, n, c, vect_lanes):
        self.generate_c_pointers(f, n)
        wi(f, "// store result")
        for i in range(0, m, dtype.max_neon_lanes):
            for j in range(n):
                if vect_lanes[c[i, j]] == dtype.max_neon_lanes:
                    # Four singles or two doubles
                    wi(f, "str q{}, [C{}], #16".format(c[i, j], j))
                elif vect_lanes[c[i, j]] == dtype.max_neon_lanes // 2:
                    # Two singles or one double
                    wi(f, "str d{}, [C{}], #8".format(c[i, j], j))
                elif vect_lanes[c[i, j]] == dtype.max_neon_lanes // 4:
                    # One single
                    wi(f, "str s{}, [C{}], #4".format(c[i, j], j))
                else:
                    # Three singles
                    # assert dtype == Type.s, 'Expected three singles'
                    wi(f, "str d{}, [C{}], #8".format(c[i, j], j))
                    wi(f, "st1 {{v{}.s}}[2], [C{}]".format(c[i, j], j))
        f.write('\n')


    def complex_generate_store_result(self, f, m, n, result_real,
                                    result_imag, vect_lanes, dtype):
        self.generate_c_pointers(f, n)
        wi(f, "// store result")
        for i in range(0, m, dtype.real_max_neon_lanes):
            for j in range(n):
                if vect_lanes[result_real[i, j]] == dtype.real_max_neon_lanes:
                    wi(
                        f,
                        "st2 {{v{2}.{0}{1}, v{3}.{0}{1}}}, [C{4}], #32".format(
                            dtype.real_max_neon_lanes, dtype.spec,
                            result_real[i, j], result_imag[i, j], j
                        )
                    )
                elif vect_lanes[result_real[i, j]] == dtype.real_max_neon_lanes // 2:
                    vectors = "{{v{}.d, v{}.d}}[0]".format(
                        result_real[i, j], result_imag[i, j])
                    if dtype == Type.c:
                        vectors = "{{v{}.2s, v{}.2s}}".format(
                            result_real[i, j], result_imag[i, j])
                    wi(f, "st2 {}, [C{}], #16".format(vectors, j))
                elif vect_lanes[result_real[i, j]] == dtype.real_max_neon_lanes // 4:
                    # One single
                    wi(f, "st2 {{v{}.s, v{}.s}}[0], [C{}], #8".format(
                        result_real[i, j], result_imag[i, j], j))
                else:
                    # Three singles
                    wi(f, "st2 {{v{}.2s, v{}.2s}}, [C{}], #16".format(
                        result_real[i, j], result_imag[i, j], j))
                    wi(f, "st2 {{v{}.s, v{}.s}}[2], [C{}]".format(
                        result_real[i, j], result_imag[i, j], j))
        f.write('\n')


    def move_alpha_beta_to_fp_registers(self, f, dtype, reg_max, vreg_max):
        wi(f, "// move alpha and beta from GPRs or load from stack")
        if dtype == Type.c:
            wi(f, "lsr x8, x6, #0x20")
            wi(f, "fmov s0, w6")
            wi(f, "fmov s1, w8")
            wi(f, "lsr x9, x7, #0x20")
            wi(f, "fmov s2, w7")
            wi(f, "fmov s3, w9")
        elif dtype == Type.z:
            gpr_base = 19
            vreg_base = 8

            total_gprs_to_preserve = min(max(0, reg_max - (gpr_base - 1)), 12)
            total_vregs_to_preserve = min(max(0, vreg_max - (vreg_base - 1)), 8)
            # + 2 here to allow for storing x29 and x30
            stack_total = 8 * (total_gprs_to_preserve + total_vregs_to_preserve + 2)

            # 16-byte align stack
            stack_total -= stack_total % - 16

            wi(f, "fmov d0, x6")
            wi(f, "fmov d1, x7")
            wi(f, "ldr d2, [sp, #{}]".format(stack_total))
            wi(f, "ldr d3, [sp, #{}]".format(stack_total+8))


    def generate_complex_routine(self, f, t, m, n, k, dtype, kernel_name):
        ab_consts, c_consts = self.get_constant_limits(t, k, n, m)
        reg_max = BASE_GPR + max(len(ab_consts), len(c_consts))

        if self.params.target_os == "windows_arm64ec":
            # we do not use x13 so we need to increase reg_max to compensate
            if reg_max >= 13:
                reg_max += 1
            # we do not use x14 so we need to increase reg_max to compensate
            if reg_max >= 14:
                reg_max += 1

        # we do not use x18 so we need to increase reg_max to compensate
        if reg_max >= 18:
            reg_max += 1

        if self.params.target_os == "windows_arm64ec":
            # we do not use x23 so we need to increase reg_max to compensate
            if reg_max >= 23:
                reg_max += 1
            # we do not use x24 so we need to increase reg_max to compensate
            if reg_max >= 24:
                reg_max += 1
            # we do not use x28 so we need to increase reg_max to compensate
            if reg_max >= 28:
                reg_max += 1

        # we do not use x29 so we need to increase reg_max to compensate
        if reg_max >= 29:
            reg_max += 1

        self.generate_function_header(f, kernel_name)
        self.generate_save_or_restore_registers(f, reg_max, NUM_VREGS-1)
        # For "windows" or "windows_arm64ec" targets...
        if self.params.target_os in ("windows", "windows_arm64ec"):
            self.move_alpha_beta_to_fp_registers(f, dtype, reg_max, NUM_VREGS-1)
        generate_dimension_shift(f, dtype)
        self.generate_a_b_pointers(f, t, m, n, k)
        a_real, a_imag, b, result_real, result_imag, vect_lanes = (
            self.complex_allocate_vectors(t, m, n, k, dtype))
        self.generate_alpha_check(f, kernel_name, dtype)

        wl(f, ".L{}_multiply_axb".format(kernel_name))
        self.clear_result_vectors(f, np.append(result_real, result_imag))
        self.complex_generate_a_load(f, t, k, m, dtype, vect_lanes, a_real, a_imag)
        self.complex_generate_b_load(f, t, k, n, dtype, vect_lanes, b)
        self.complex_generate_a_mult_b(f, k, m, n, t, a_real,
                                a_imag, b, result_real, result_imag, dtype)

        intermediate_res_real, intermediate_res_imag, vect_lanes = (
            self.complex_allocate_c_vectors(result_real, result_imag,
                                            vect_lanes, m, n, dtype))
        self.complex_generate_mult_alpha(
            f, m, n, result_real, result_imag,
            intermediate_res_real, intermediate_res_imag, dtype)

        wi(f, "b .L{}_check_beta".format(kernel_name))
        f.write('\n')

        wl(f, ".L{}_skip_axb".format(kernel_name))
        self.clear_result_vectors(
            f, np.append(intermediate_res_real, intermediate_res_imag))

        wl(f, ".L{}_check_beta".format(kernel_name))
        self.generate_beta_check(f, kernel_name, dtype)

        wl(f, ".L{}_multiply_c".format(kernel_name))
        c_real, c_imag, vect_lanes = self.complex_allocate_c_vectors(
            intermediate_res_real, intermediate_res_imag,
            vect_lanes, m, n, dtype)
        self.complex_generate_c_load(
            f, c_real, c_imag, m, n, vect_lanes, intermediate_res_real, dtype)

        self.complex_generate_result_plus_c_mult_beta(
            f, c_real, c_imag, intermediate_res_real,
            intermediate_res_imag, m, n, dtype)

        wl(f, ".L{}_store_c".format(kernel_name))
        self.complex_generate_store_result(
            f, m, n, intermediate_res_real,
            intermediate_res_imag, vect_lanes, dtype)

        wl(f, ".L{}_restore".format(kernel_name))
        self.generate_save_or_restore_registers(
            f, reg_max, NUM_VREGS-1, restore=True)
        wi(f, "ret")
        f.write('\n')

        return kernel_name


    def generate_routine(self, f, t, m, n, k, dtype, kernel_name):
        ab_consts, c_consts = self.get_constant_limits(t, k, n, m)
        reg_max = BASE_GPR + max(len(ab_consts), len(c_consts))

        if self.params.target_os == "windows_arm64ec":
            # we do not use x13 so we need to increase reg_max to compensate
            if reg_max >= 13:
                reg_max += 1
            # we do not use x14 so we need to increase reg_max to compensate
            if reg_max >= 14:
                reg_max += 1

        # we do not use x18 so we need to increase reg_max to compensate
        if reg_max >= 18:
            reg_max += 1

        if self.params.target_os == "windows_arm64ec":
            # we do not use x23 so we need to increase reg_max to compensate
            if reg_max >= 23:
                reg_max += 1
            # we do not use x24 so we need to increase reg_max to compensate
            if reg_max >= 24:
                reg_max += 1
            # we do not use x28 so we need to increase reg_max to compensate
            if reg_max >= 28:
                reg_max += 1

        # we do not use x29 so we need to increase reg_max to compensate
        if reg_max >= 29:
            reg_max += 1

        self.generate_function_header(f, kernel_name)

        a, b, result, vect_lanes = self.allocate_vectors(t, m, n, k, dtype)
        c_v = self.allocate_c_vectors(result, m, n, dtype)

        self.generate_save_or_restore_registers(
            f, reg_max, max(np.max(c_v), np.max(result)))
        generate_dimension_shift(f, dtype)
        self.generate_a_b_pointers(f, t, m, n, k)
        self.clear_result_vectors(f, result)

        wl(f, ".L{}_check_alpha".format(kernel_name))
        self.generate_alpha_check(f, kernel_name, dtype)

        wl(f, ".L{}_multiply_axb".format(kernel_name))
        self.generate_a_load(f, t, k, m, dtype, vect_lanes, a)
        self.generate_b_load(f, t, k, n, dtype, vect_lanes, b)
        self.generate_a_mult_b(f, k, m, n, dtype, a, b, t, result)
        self.generate_res_mult_alpha(f, m, n, result, dtype)

        wl(f, ".L{}_check_beta".format(kernel_name))
        self.generate_beta_check(f, kernel_name, dtype)

        wl(f, ".L{}_multiply_c".format(kernel_name))
        self.generate_c_load(f, c_v, m, n, dtype, vect_lanes, result)
        self.generate_result_plus_c_mult_beta(f, c_v, result, m, n, dtype)

        wl(f, ".L{}_store_c".format(kernel_name))
        self.generate_store_result(f, m, dtype, n, result, vect_lanes)

        wl(f, ".L{}_restore".format(kernel_name))
        self.generate_save_or_restore_registers(
            f, reg_max, max(np.max(c_v), np.max(result)), restore=True
        )
        wi(f, "ret")

        return kernel_name

    def write_kernel(self):
        assert self.params.dtype.is_complex or not (self.params.trans.conja or self.params.trans.conjb)
        f = self.f
        self.generate_column_constants(
            f, self.params.trans, self.params.k, self.params.n, self.params.m)

        if self.params.dtype.is_complex:
            self.generate_complex_routine(
                f, self.params.trans, self.params.m, self.params.n,
                self.params.k, self.params.dtype, self.params.routine_name)
        else:
            self.generate_routine(
                f, self.params.trans, self.params.m, self.params.n,
                self.params.k, self.params.dtype, self.params.routine_name)

        self.generate_column_constants(f, self.params.trans, self.params.k,
                                       self.params.n, self.params.m, undef=True)
### End of class AtomicGEMMKernelWriterNeon

def generate_input_parameters_neon(f):
    f.write("// parameters\n")
    parameters = ['a_ptr', 'lda', 'b_ptr', 'ldb', 'c_ptr', 'ldc']
    for i, param in enumerate(parameters):
        f.write("#define {} x{}\n".format(param, i))
    f.write('\n')

def generate_source_files_neon(atom_kern_params_list, source_filenames):
    """Write a source file for each type containing all kernels for that type
    """
    for dtype, source_filename in source_filenames.items():
        with open(source_filename, 'w') as f:
            generate_input_parameters_neon(f)
            dtype_params = [
                params for params in atom_kern_params_list
                if params.dtype == dtype]
            prev_consts = None
            for params in dtype_params:
                # We don't need to generate a kernel in the case that the data
                # type is real, and transpose parameters are complex. We simply need
                # to deal with these cases in the dispatch code
                if not params.dtype.is_complex and (params.trans.conja or params.trans.conjb):
                    continue
                kernel_writer = AtomicGEMMKernelWriterNeon(f, params)
                kernel_writer.write_kernel()

def valid_atomic_kern_params_neon(trans_list, dtypes, target_os):
    valid_dtypes = {Type.s, Type.d, Type.c, Type.z}
    def gen_params_list(dtype, target_os):
        reg_info = AtomicKernelRegInfoNeon(dtype, target_os)
        params_list = [AtomicKernelParamsNeon(m, n, k, trans, dtype, target_os)
                        for trans in trans_list
                        for m, n, k in reg_info.generate_valid_kernel_dims(trans)]
        return params_list
    if dtypes:
        valid_dtypes = valid_dtypes.intersection(dtypes)
        ret_val = []
        for dtype in valid_dtypes:
            ret_val.extend(gen_params_list(dtype, target_os))
    else:
        # If no dtypes are specified then kernels for all valid dtypes are generated by default
        ret_val = [p for dtype in valid_dtypes for p in gen_params_list(dtype, target_os)]
    return ret_val, valid_dtypes
