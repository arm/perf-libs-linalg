# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import sys
from atomic_gemm_common import *


class AtomicKernelRegInfoSVE(AtomicKernelRegInfo):
    def __init__(self, dtype):
        # We have 31 general purpose registers, of which 7 are integer parameters to the function
        # a_ptr, lda, b_ptr, ldb, c_ptr, ldc, m
        # We don't want to use x18 and, x29, so have 29 general purpose register to use
        # There are also parameters alpha and beta, which are constants. In the real case, these
        # take two floating point (vector) registers, otherwise they take four
        self.dtype = dtype
        self.is_complex = dtype.is_complex
        self.max_b_regs = 8 if dtype.spec == "s" else 16
        super(AtomicKernelRegInfoSVE, self).__init__(29, 7, 32, 4 if self.is_complex else 2)

    @staticmethod
    def num_vecs_required(m_vecs, n, k, trans):
        # The number of vector registers required is the maximum between storing the matrices A, B and result matrix,
        # and storing the result matrix and the matrix C
        b_cols = k if trans.transb else n
        return max(m_vecs * (k + n) + b_cols, 2 * m_vecs * n)

    @staticmethod
    def num_gpregs_required(n, k, trans):
        # We need to store pointers to result matrix C, or pointers to columns of A and B
        b_cols = k if trans.transb else n
        # In the case of a transpose matrix we need to know how large the offset
        # between the start of full indexed register loads in a column
        a_col_offset_regs = 1 if trans.transa else 0
        return max(n, k + b_cols + a_col_offset_regs)

    def generate_valid_kernel_dims(self, trans):
        ret_list = [(m_vecs, n, k)
            for m_vecs in self._get_valid_m_vecs(trans)
            for k in self._get_valid_k(m_vecs, trans)
            for n in self._get_valid_n(m_vecs, k, trans)]
        return ret_list

    def _get_valid_m_vecs(self, trans):
        # Returns a list of valid sizes for the number of vectors in a column of A. We work with the
        # constraint that A, B and the result matrix need to fit into registers
        # Minimum size of B is 1 vector x 1 column (or row), and minimum size of the output matrix
        # has the same number of vectors in a column as A, and as many columns as B
        avail_vregs = self.num_vreg - self.base_vreg
        return [m_vecs for m_vecs in range(1, 16) if self.num_vecs_required(m_vecs, 1, 1, trans) <= avail_vregs]

    def _get_valid_k(self, m_vecs, trans):
        # Returns a list of valid sizes for the number of columns in matrix A. We work with the constraint
        # that A, B and the result matrix need to fit into vector registers, and that the pointers to the
        # start of columns of A and B can be stored in general purpose registers. The minimum number of
        # columns in B is one.
        avail_vregs = self.num_vreg - self.base_vreg
        avail_gpregs = self.num_gpreg - self.base_gpreg - (1 if trans.transa else 0)
        # We need to make sure that we can fit B into the first 8 (16) registers for single (double) precision
        # as these are the only registers that are allowed to be used as the second source vector in an indexed FMLA
        b_regs = lambda x: 0 if not trans.transb else x
        b_neon_regs = lambda x: 0 if trans.transb else (x * self.dtype.width / 16)
        return [k for k in range(1, avail_gpregs)
                  if self.num_vecs_required(m_vecs, 1, k, trans) <= avail_vregs and
                     b_regs(k) <= self.max_b_regs and b_neon_regs(k) <= 16]

    def _get_valid_n(self, m_vecs, k, trans):
        # Returns a list of the valid numbers of columns in matrix C. We already know the number of vectors
        # required to store the data in a single column of C, as this is the same as for A
        avail_vregs = self.num_vreg - self.base_vreg
        avail_gpregs = self.num_gpreg - self.base_gpreg - (1 if trans.transa else 0)
        b_regs = lambda x: x if trans.fullname[1] == 'N' else 0
        b_neon_regs = lambda x: (x * self.dtype.width / 16) if trans.fullname[1] != 'N' else 0
        return [n for n in range(1, avail_gpregs)
                  if (self.num_vecs_required(m_vecs, n, k, trans) <= avail_vregs and
                     self.num_gpregs_required(n, k, trans) <= avail_gpregs and
                     b_regs(n) <= self.max_b_regs and b_neon_regs(k) <= 16)]

class AtomicKernelParamsSVE(AtomicGEMMKernelParams):
    def __init__(self, m_vecs, n, k, trans, dtype, target_os):
        """ An atomic SVE GEMM kernel needs to know the number of vectors that are
            in a column of A (and hence in a column of C) and B, and the number of columns
            in each of the matrices.

            With this data we can write a kernel. The kernel is not aware of the numbers
            of registers available, and assumes that the parameters passed to it are
            valid.
        """
        self.m_vecs = m_vecs
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans = trans
        self.target_os = target_os

    def __eq__(self, other):
        return (self.m_vecs == other.m_vecs and
                self.n == other.n and
                self.k == other.k and
                self.trans == other.trans and
                self.target_os == other.target_os)

    def __hash__(self):
        return (self.m_vecs | (self.n << 5) | (self.k << 10) | (self.dtype.value << 15) | (int(self.trans) << 17))

    @property
    def routine_name(self):
        return "kernel_sve_{}gemm_{}_{}_{}_{}".format(self.dtype.name,
            self.m_vecs, self.n, self.k, self.trans.fullname)

    @property
    def c_function_signature(self):
        return ('void {0}(const {1} *a, kernel_inttype lda, '
                'const {1} *b, kernel_inttype ldb, {1} *c, '
                'kernel_inttype ldc, kernel_inttype m, {1} alpha, {1} beta);'.format(
                    self.routine_name, self.dtype.c_type))
### End of class AtomicKernelParamsSVE

class AtomicKernelAssignedRegistersSVE(object):
    def __init__(self, reg_info, params):
        assert type(params) == AtomicKernelParamsSVE
        assert type(reg_info) == AtomicKernelRegInfoSVE

        # NB: The order of the function calls in this method is important.
        #     Behavior will change if you swap the order of the setting of the b
        #     and c registers

        # Pointers to columns of A
        self._set_a_col_ptrs(reg_info, params)

        # Pointers to columns (or rows) of B
        self._set_b_ptrs(reg_info, params)

        # Pointers to columns of C
        self._set_c_col_ptrs(reg_info, params)

        # The order of the assignment of the registers is important. The C
        # matrix must be set last
        # The B matrix gets assigned the lowest numbered registers, as the indexed
        # fmul and fmla instructions only work on the first 8 (16) vector registers
        # for single (double) precision
        # Registers in results matrix
        self._set_ab_res_regs(params)

        # Registers in matrix A
        self._set_mat_a_regs(params)

        # Registers in row of matrix B
        self._set_mat_b_regs(params)

        # Registers in matrix C
        self._set_mat_c_regs(params)

        # We store intermediate results in different registers in the real and complex kernels.
        # Now that we have all our matrix registers assigned, we can set up the aliases for this.
        self._set_mat_reg_aliases(params)

        # As we are going to load B into the first vectors, we need
        # to move alpha and beta to after any other parameters
        self._set_alpha_beta_regs(params)
    ### End of function __init__

    def _get_gpregister_list(self, base, top):
        # Generate a list of valid general-purpose registers (i.e. excluding x18 and x29)
        # starting from base and going to base+top
        return [ x + base if x + base < 18 \
                 else x + base + 1 if x + base < 28 \
                 else x + base + 2 \
                 for x in range(top) ]

    def _set_a_col_ptrs(self, reg_info, params):
        # General purpose registers to store columns of A
        self.a_col_ptrs = self._get_gpregister_list(reg_info.base_gpreg, params.k)
        # In the transpose case we need to know the offset between successive
        # vector loads in a column of A. This variable will only be used
        # if A is transposed in a kernel
        self.a_col_offset = params.k + reg_info.base_gpreg if params.k + reg_info.base_gpreg < 18 \
                            else params.k + reg_info.base_gpreg + 1

    def _set_b_ptrs(self, reg_info, params):
        # For B we need to set the pointers to columns or rows, depending on the transpose options
        curr_reg = reg_info.base_gpreg + params.k + (1 if params.trans.transa else 0)
        num_ptrs = params.k if params.trans.transb else params.n
        self.b_ptrs = self._get_gpregister_list(curr_reg, num_ptrs)
        self.max_assigned_gpreg = max(self.b_ptrs)
        # As well as setting the b_ptrs we make sure that Neon loads from a row / column of B
        # can be done with an offset between -128 and 112 from the current pointer. For this we
        # need to know the number of Neon registers in a row / column of B (depending on transpose option)
        elements_per_neon_vec = 16 / params.dtype.width
        b_dim = params.n if params.trans.transb else params.k
        self.b_num_neon_vecs = int(b_dim / elements_per_neon_vec)

    def _set_c_col_ptrs(self, reg_info, params):
        # General purpose registers to store columns of C
        self.c_col_ptrs = self._get_gpregister_list(reg_info.base_gpreg, params.n)
        self.max_assigned_gpreg = max(self.max_assigned_gpreg, max(self.c_col_ptrs))

    def _set_alpha_beta_regs(self, params):
        # We store alpha and beta after the other registers, so that we can store
        # B in the lowest numbered registers.
        # We need to ensure we don't use any of the input registers storing alpha and beta
        # as intermediates while we manipulate them while moving them -- this is only a
        # problem for the smallest complex kernels where the number of vector registers
        # required to store A, B, and AB is less that the number of registers required to
        # store the separate components of alpha and beta.
        curr_reg = max(self.max_assigned_vreg + 1, 4 if params.dtype.is_complex else 2)
        self.alpha_reg = curr_reg
        self.beta_reg = curr_reg + 1
        self.max_assigned_vreg += 2

    def _set_ab_res_regs(self, params):
        # Sets the registers required to store the result matrix.
        # This is stored in the lowest numbered registers we have access
        # to, as we then know the size of this, and can store both of the
        # A and B matrices, as well as the C matrix starting in registers
        # numbered after this one
        curr_reg = (params.k if params.trans.transb else params.n) + params.m_vecs * params.k
        self.ab_res_regs = []
        for _ in range(params.m_vecs):
            row_regs = [curr_reg + j for j in range(params.n)]
            self.ab_res_regs.append(row_regs)
            curr_reg += params.n
        self.max_assigned_vreg = curr_reg-1

    def _set_mat_a_regs(self, params):
        # Sets the register numbers to store matrix A
        self.mat_a_regs = []
        curr_reg = (params.k if params.trans.transb else params.n)
        for _ in range(params.m_vecs):
            row_regs = [curr_reg + j for j in range(params.k)]
            self.mat_a_regs.append(row_regs)
            curr_reg += params.k

    def _set_mat_b_regs(self, params):
        # Sets the register numbers to store matrix B. This needs to be the lowest numbered registers,
        # as we are only allowed to use a limited set of registers when performing indexed fmla operations
        curr_reg = 0
        num_regs = params.k if params.trans.transb else params.n
        self.mat_b_regs = [curr_reg + x for x in range(num_regs)]

    def _set_mat_c_regs(self, params):
        # Sets the register numbers to store matrix C
        # We start from register zero, as we can overwrite registers used for A and B matrices
        # We need to pay attention not to use registers assigned to the result matrix, though
        self.mat_c_regs = []
        curr_reg = 0
        res_base_reg = self.ab_res_regs[0][0]
        reg_after_res = self.ab_res_regs[-1][-1] + 1
        for _ in range(params.m_vecs):
            if curr_reg < res_base_reg and (curr_reg + params.n) >= res_base_reg:
                row_regs = [j for j in range(curr_reg, res_base_reg)]
                num_assigned_regs = len(row_regs)
                row_regs += [reg_after_res + j for j in range(params.n - num_assigned_regs)]
                curr_reg = reg_after_res + params.n - num_assigned_regs
            else:
                row_regs = [curr_reg + j for j in range(params.n)]
                curr_reg += params.n
            self.mat_c_regs.append(row_regs)
        self.max_assigned_vreg = max(self.max_assigned_vreg, curr_reg - 1)

    def _set_mat_reg_aliases(self, params):
        # Sets the register numbers to store intermediate results for A*B and alpha*A*B
        # These differ for the real and complex kernels because in the complex cases we require
        # additional registers to compute alpha*A*B
        # For complex kernels we store the result of alpha*A*B in what is the C registers in the real kernels
        self.alpha_ab_res_regs = self.mat_c_regs if params.dtype.is_complex else self.ab_res_regs
        # We use the same set of registers to accumulate alpha*A*B+C into
        self.abpc_res_regs = self.alpha_ab_res_regs
        # For complex kernels we store the result of alpha*A*B+C in what is the A*B registers in the real kernels
        self.c_load_regs = self.ab_res_regs if params.dtype.is_complex else self.mat_c_regs
### End of class AtomicKernelAssignedRegistersSVE

class AtomicGEMMKernelWriterSVE(AtomicGEMMKernelWriter):
    def __init__(self, f, params, assigned_regs):
        super(AtomicGEMMKernelWriterSVE, self).__init__(f, params, assigned_regs)
        get_label_str = lambda routine_name, section: ".L_{}_{}".format(routine_name, section)
        self.labels = dict((x, get_label_str(self.params.routine_name, x))
                            for x in ["check_alpha", "zero_res_mat", "load_a", "multiply_axb",
                                      "load_c", "multiply_c", "check_beta", "store_c"])

    def generate_function_header(self, f):
        if self.params.target_os == "linux":
            wi(f, ".global {}".format(self.params.routine_name))
            wi(f, ".type {}, %function".format(self.params.routine_name))
            wi(f, ".p2align 5")
            f.write("{}:\n".format(self.params.routine_name))
        elif self.params.target_os == "mac":
            wi(f, ".global _{}".format(self.params.routine_name))
            wi(f, ".p2align 2")
            f.write("_{}:\n".format(self.params.routine_name))
        # For "windows" or "windows_arm64ec" targets...
        elif self.params.target_os in ("windows", "windows_arm64ec"):
            # If we're building for Arm64EC prepend # to kernel name
            # and enclose in double-quotes
            decorated_name = "\"#{}\"".format(self.params.routine_name) if self.params.target_os == "windows_arm64ec" else self.params.routine_name
            # function header
            wi(f, ".global {}".format(decorated_name))
            wi(f, ".def {};".format(decorated_name))
            wi(f, ".scl 2;")
            wi(f, ".type 32;")
            wi(f, ".endef")
            f.write("{}:\n".format(decorated_name))
        else:
            raise Exception("Unrecognized operating system target " + self.params.target_os + " in SVE atomic kernel generation")

    def write_kernel(self):
        """ Responsible for the writing of a kernel to file
        """
        f = self.f
        # Generate the defines for the different elements required in the
        # GEMM operation
        f.write("// Start of operation for kernel: {}\n".format(self.params.routine_name))
        self.gen_function_defines_sve()

        # Generate function header
        self.generate_function_header(f)

        # Store required registers to satisfy pcs
        wi(f, "// PCS spills")
        self.generate_pcs_spill(True)
        f.write("\n")

        # Generate a true predicate in p_alltrue (usually p0)
        wi(f, "ptrue p_alltrue.{}".format(self.params.dtype.spec))

        # For single-precision complex NC/TC kernels we need to manually calculate
        # conjugates, so create a predicate to access just the imaginary components
        if self.params.dtype==Type.c and not self.params.trans.conja and self.params.trans.conjb:
            wi(f, "pfalse p_imag.b")
            wi(f, "zip1 p_imag.s, p_imag.s, p_alltrue.s")

        # Get a predicate for the load / store to the last vector register in A and C
        # We set m_vecs appropriately in the chooser to take care of complex cases
        wi(f, "// Find out how full the last vector register is for loads / stores to and from A and C")
        if self.params.dtype.is_complex:
            wi(f, "// Need to double the element count in complex kernels")
            wi(f, "lsl m_dim, m_dim, #1")
        if self.params.m_vecs != 1:
            wi(f, "cnt{} curr_vl, ALL, MUL #{}".format(self.params.dtype.suffix, self.params.m_vecs - 1))
        else:
            wi(f, "eor curr_vl, curr_vl, curr_vl")
        wi(f, "whilelt p_lastvec.{}, curr_vl, m_dim".format(self.params.dtype.spec))

        # Set the values for the pointers to the start of columns of A and B
        self.generate_col_ptrs(True, True, False, self.params.dtype)

        # Move alpha and beta to higher numbered registers to allow B to occupy the
        # lowest numbered registers
        self.move_alpha_beta_to_high_registers()

        # Check if alpha is zero, and branch to checking to beta if this is the case
        self.generate_alpha_check_sve()

        # Load matrix A
        self.generate_a_load()

        # Multiplication of A x B
        self.generate_a_mult_b()

        # Check if beta is zero
        self.generate_check_beta()

        # Load matrix C
        self.generate_c_load()

        # Multiply C by beta and add result
        self.generate_multiply_beta_c()

        # Store the result
        self.store_result()

        # Restore registers from the stack
        wi(f, "// Restore PCS spills")
        self.generate_pcs_spill(False)
        wi(f, "ret")

        # Store the size of the function
        if self.params.target_os == "linux":
            f.write(".size {0}, . - {0}\n\n".format(self.params.routine_name))
    ### End of function write_kernel

    def gen_function_defines_sve(self):
        """ Generates the defines used in an SVE kernel. This is for
            convenience only, and makes the kernel code easier to read
        """
        f = self.f
        # Pointers into matrices A, B and C
        # Matrix A
        f.write("// Pointers to columns of A\n")
        for i, num in enumerate(self.assigned_regs.a_col_ptrs):
            f.write("// A{} <- x{}\n".format(i, num))

        # Matrix B
        f.write("// Pointers to a single column (or row in the transpose case) of B\n")
        for i, num in enumerate(self.assigned_regs.b_ptrs):
            f.write("// B{} <- x{}\n".format(i, num))

        # Matrix C
        f.write("// Pointers to matrix C. These can reuse registers for A and B,\n")
        f.write("// as they aren't loaded at the same time\n")
        for i, num in enumerate(self.assigned_regs.c_col_ptrs):
            f.write("// C{} <- x{}\n".format(i, num))
        f.write("\n")

        # Registers to store values from the different matrices
        f.write("// Assign registers to A, B, C and result matrix\n")

        def write_matrix_regs(regs, mat_label):
            for i, row_regs in enumerate(regs):
                for j, reg_num in enumerate(row_regs):
                    f.write("// {}{}_{} <- z{}\n".format(mat_label, i, j, reg_num))

        # Result matrix
        write_matrix_regs(self.assigned_regs.ab_res_regs, "RES")
        f.write("\n")

        # Matrix A
        write_matrix_regs(self.assigned_regs.mat_a_regs, "A")
        f.write("\n")

        # Matrix B (we only have a single number or column here)
        for i, reg_num in enumerate(self.assigned_regs.mat_b_regs):
            f.write("// B_{} <- z{}\n".format(i, reg_num))
        f.write("\n")

        # Matrix C
        write_matrix_regs(self.assigned_regs.mat_c_regs, "C")
        f.write("\n")

        f.write("// Reassign registers to alpha and beta\n")
        f.write("// alpha <- z{}\n".format(self.assigned_regs.alpha_reg))
        f.write("// beta <- z{}\n".format(self.assigned_regs.beta_reg))
        f.write("\n")
    ### End of function gen_function_defines_sve

    def generate_pcs_spill(self, spill):
        f = self.f
        # We need to spill used floating point registers d8 - d15
        # We know that we assign vectors in contiguous order, so we can figure
        # out how many vectors to spill if we find the maximum assigned register
        num_vec_spills = min(8, max(self.assigned_regs.max_assigned_vreg - 7, 0))

        # Spill general purpose regiseters r19-28
        # When x30 is being used as a temporary register we do not need to save/restore
        # it separately because it will always be saved/restored alongside x29
        max_gpreg = 28 if self.assigned_regs.max_assigned_gpreg > 28 \
                    else self.assigned_regs.max_assigned_gpreg
        num_gp_spills = max(max_gpreg - 18, 0)

        # We update the stack pointer in increments of 16 bytes
        sp_increment = (int((num_vec_spills + 1) / 2) + int((num_gp_spills + 1) / 2) + 1) * 16

        op = "st" if spill else "ld"

        if spill:
            wi(f, "// vec spills: {}, gpreg spills: {}".format(num_vec_spills, num_gp_spills))
            wi(f, "stp x29, x30, [sp, #-{}]!".format(sp_increment))
            wi(f, "mov x29, sp")
        # Write the floating point spills / restores
        curr_offset = 16
        for i in range(0, num_vec_spills-1, 2):
            wi(f, "{}p d{}, d{}, [sp, #{}]".format(op, i+8, i+9, curr_offset))
            curr_offset += 16

        if num_vec_spills % 2 != 0:
            wi(f, "{}r d{}, [sp, #{}]".format(op, num_vec_spills + 7, curr_offset))
            curr_offset += 16

        # Write the general purpose spills / restores
        for i in range(0, num_gp_spills-1, 2):
            wi(f, "{}p x{}, x{}, [sp, #{}]".format(op, i + 19, i + 20, curr_offset))
            curr_offset += 16

        if num_gp_spills % 2 != 0:
            wi(f, "{}r x{}, [sp, #{}]".format(op, num_gp_spills + 18, curr_offset))

        if not spill:
            wi(f, "ldp x29, x30, [sp], #{}".format(sp_increment))
    ### End of function generate_pcs_spill

    def generate_col_ptrs(self, shift_lds, a_b, c, dtype):
        """ Writes the loading of column pointers into assigned registers to file.
            shift_lds, a_b and c are booleans. We may or may not need to
            multiply the values of the lda, ldb and ldc parameters to get the
            byte offset, rather than offset in the datatype. This method may
            be called more than once, and we only need to do this
            multiplication once. The parameter shift_lds indicates whether
            this shift is to be emitted or not.

            Boolean a_b indicates that the column pointers for matrices A and
            B are to be generated, and boolean c indicates whether the
            pointers for matrix C are to be generated
        """
        f = self.f
        if shift_lds:
            generate_dimension_shift(f, dtype)
            f.write("\n")

        def write_col_ptrs(regs, label, ldstr):
            wi(f, "mov x{}, {}".format(regs[0], label))
            for curr, prev in zip(regs[1:], regs):
                wi(f, "add x{}, x{}, {}".format(curr, prev, ldstr))

        if a_b:
            wi(f, "// Store memory location of start of columns of A")
            a_col_strd = "#{}".format(self.params.dtype.width) if self.params.trans.transa else "lda"
            if self.params.m_vecs > 8:
                # If there are more than 8 vector registers, we cannot index into them using
                # offsets from the base pointer in the range [-8, 7]. Update the base pointer
                # in this case. In the case that A is transposed, we don't have this restriction,
                # so we don't do anything in this case
                if not self.params.trans.transa:
                    wi(f, "incb a_ptr, ALL, MUL #8")
                wi(f, "incb c_ptr, ALL, MUL #8")
            write_col_ptrs(self.assigned_regs.a_col_ptrs, "a_ptr", a_col_strd)
            f.write("\n")

            # We want to make sure that in ld1rq instructions we have an index
            # offset of fewer than 112 bytes.
            wi(f, "// Store memory location of start of columns of B")
            if self.assigned_regs.b_num_neon_vecs > 7:
                # Update the base pointer for the matrix B
                # The maximum offset is 112, so we need the column pointers to be within 112 bytes of the data to load
                # If not, we can go to a minimum offset of -128, so we simply add 128 to the pointers for the columns of B
                wi(f, "// Update the pointers to B so that loads from B have an offset in the range [-128, 112]")
                wi(f, "add b_ptr, b_ptr, #128")
            write_col_ptrs(self.assigned_regs.b_ptrs, "b_ptr", "ldb")
            f.write("\n")

        if c:
            wi(f, "// Store memory location of start of columns of C")
            write_col_ptrs(self.assigned_regs.c_col_ptrs, "c_ptr", "ldc")
            f.write("\n")
    ### End of function generate_col_ptrs

    def move_alpha_beta_to_high_registers(self):
        f = self.f
        wi(f, "// Move alpha and beta to higher numbered registers to free space")
        wi(f, "// in lowest numbered registers for B")
        if self.params.dtype.is_complex:
            # alpha.re will be in z0.x[0]
            # alpha.im will be in z1.x[0]
            #  beta.re will be in z2.x[0]
            #  beta.im will be in z3.x[0]
            # We want to store:
            #  zA.x = [ alpha.re, alpha.im, alpha.re, alpha.im .... ]
            wi(f, "ins v0.{0}[1], v1.{0}[0]".format(self.params.dtype.spec))
            wi(f, "ins v2.{0}[1], v3.{0}[0]".format(self.params.dtype.spec))
            wi(f, "dup z{0}.{1}, z0.{1}[0]".format(self.assigned_regs.alpha_reg, self.params.dtype.spec2x))
            wi(f, "dup z{0}.{1}, z2.{1}[0]\n".format(self.assigned_regs.beta_reg, self.params.dtype.spec2x))
        else:
            wi(f, "dup z{0}.{1}, z0.{1}[0]".format(self.assigned_regs.alpha_reg, self.params.dtype.spec))
            wi(f, "dup z{0}.{1}, z1.{1}[0]\n".format(self.assigned_regs.beta_reg, self.params.dtype.spec))

    def generate_alpha_check_sve(self):
        f = self.f
        wl(f, self.labels["check_alpha"])
        wi(f, "// If alpha is 0 we can skip (alpha * A * B)")
        if self.params.dtype.is_complex:
            wi(f, "fcmne p_comp.{0}, p_alltrue/z, z{1}.{0}, #0.0".format(self.params.dtype.spec, self.assigned_regs.alpha_reg))
            wi(f, "ptest p_alltrue, p_comp.b")
        else:
            wi(f, "fcmp {}{}, #0.0".format(self.params.dtype.spec, self.assigned_regs.alpha_reg))
        wi(f, "b.ne {}".format(self.labels["load_a"]))

        # Zero the matrix in the case that alpha was zero
        f.write("\n")
        wl(f, self.labels["zero_res_mat"])
        wi(f, "// Zero result matrix to deal with the case that alpha=0")
        for row in self.assigned_regs.alpha_ab_res_regs:
            for reg_num in row:
                wi(f, "dup z{}.{}, #0".format(reg_num, self.params.dtype.spec))
        wi(f, "b {}".format(self.labels["check_beta"]))
        f.write("\n")

    def _write_matrix_load(self, which_mat):
        f = self.f
        stride_reg = None
        if which_mat.upper() == "A":
            mat_regs = self.assigned_regs.mat_a_regs
            col_ptrs = self.assigned_regs.a_col_ptrs
            if self.params.trans.transa:
                # In the case that we are loading A, if A is transposed, we need to
                # have a register store the strided index. For A, this can be a vector
                # register which will be used to store matrix B in the future
                stride_reg = self.assigned_regs.mat_b_regs[0]
                # Set up the strided indexed loads
                wi(f, "// Get the offsets for strided loads")
                # We know that lda is the second parameter passed in, so lives in w/x1. We need
                # to use w for single, and x for double precision offset register in the index
                # command
                wi(f, "index z{}.{}, #0, {}1".format(stride_reg, self.params.dtype.spec, "w" if self.params.dtype.spec == "s" else "x"))
                if self.params.dtype.is_complex:
                    # Need a second register to temporarily store a second index for the imaginary parts
                    # We won't need this again after the final index is constructed
                    im_stride_reg = self.assigned_regs.mat_a_regs[0][0]
                    wi(f, "index z{}.{}, #{}, {}1".format(im_stride_reg, self.params.dtype.spec, "4" if self.params.dtype.spec == "s" else "8", "w" if self.params.dtype.spec == "s" else "x"))
                    wi(f, "zip1 z{0}.{1}, z{0}.{1}, z{2}.{1}".format(stride_reg, self.params.dtype.spec, im_stride_reg))
                    # We need to know how far to step within a column to load one
                    # full vector of strided elements
                    next_dtype = self.params.dtype.spec2x
                    if next_dtype == "q":
                        # Need to divide cnt by 2 because that's how many complex elements there are
                        # in a vector.
                        wi(f, "cntd x{}".format(self.assigned_regs.a_col_offset))
                        wi(f, "lsr x{0}, x{0}, #1".format(self.assigned_regs.a_col_offset))
                    else:
                        wi(f, "cnt{} x{}".format(next_dtype, self.assigned_regs.a_col_offset))
                else:
                    wi(f, "cnt{} x{}".format(self.params.dtype.suffix, self.assigned_regs.a_col_offset))
                wi(f, "mul x{0}, x{0}, lda".format(self.assigned_regs.a_col_offset))
        elif which_mat.upper() == "C":
            mat_regs = self.assigned_regs.c_load_regs
            col_ptrs = self.assigned_regs.c_col_ptrs

        num_mat_regs = len(mat_regs)
        assert(self.params.m_vecs == len(mat_regs))

        col_offset = 0
        # In the case that we are loading / storing into a column major matrix we need to make sure that we
        # don't use an immediate offset for the ld1 instruction which is larger than 8
        if num_mat_regs > 8 and (which_mat.upper() == "C" or (which_mat.upper() == "A" and not self.params.trans.transa)):
            col_offset = 8

        if which_mat.upper() == "C" or (which_mat.upper() == "A" and not self.params.trans.transa):
            for i, row in enumerate(mat_regs[:-1]):
                for j, reg in enumerate(row):
                    wi(f, "ld1{0} {{z{1}.{2}}}, p_alltrue/z, [x{3}, #{4}, MUL VL]".format(self.params.dtype.suffix, reg, self.params.dtype.spec, col_ptrs[j], i - col_offset))
        else:
            assert which_mat.upper() == "A" and self.params.trans.transa
            # In the case that we have A transposed, we need to update the column pointer between
            # loads
            sign_extend = ", sxtw" if self.params.dtype.spec == "s" else ""
            for row in mat_regs[:-1]:
                for j, reg in enumerate(row):
                    wi(f, "ld1{0} {{z{1}.{2}}}, p_alltrue/z, [x{3}, z{4}.{2}{5}]".format(
                        self.params.dtype.suffix, reg, self.params.dtype.spec, col_ptrs[j], stride_reg, sign_extend))
                    wi(f, "add x{0}, x{0}, x{1}".format(col_ptrs[j], self.assigned_regs.a_col_offset))
        f.write("\n")

        wi(f, "// Partial register load must be predicated")
        # We can only use offsets in the range -8 to 7. Use of col_offset makes
        # sure that we are within these bounds, assuming that a valid set of
        # parameters has been passed
        if which_mat.upper() == "C" or (which_mat.upper() == "A" and not self.params.trans.transa):
            for j, reg in enumerate(mat_regs[-1]):
                wi(f, "ld1{0} {{z{1}.{2}}}, p_lastvec/z, [x{3}, #{4}, MUL VL]".format(self.params.dtype.suffix, reg, self.params.dtype.spec, col_ptrs[j], num_mat_regs - 1 - col_offset))
        else:
            sign_extend = ", sxtw" if self.params.dtype.spec == "s" else ""
            assert which_mat.upper() == "A" and self.params.trans.transa
            for j, reg in enumerate(mat_regs[-1]):
                wi(f, "ld1{0} {{z{1}.{2}}}, p_lastvec/z, [x{3}, z{4}.{2}{5}]".format(
                    self.params.dtype.suffix, reg, self.params.dtype.spec, col_ptrs[j], stride_reg, sign_extend))
        f.write("\n")

    def generate_a_load(self):
        f = self.f
        wl(f, self.labels["load_a"])
        # For complex types, zero the results registers because we will be using FCMLAs
        if self.params.dtype.is_complex:
            wi(f, "// Zero the results registers")
            for row in self.assigned_regs.ab_res_regs:
                for reg_num in row:
                    wi(f, "dup z{}.{}, #0".format(reg_num, self.params.dtype.spec))
            f.write("\n")
        wi(f, "// Load matrix A")
        self._write_matrix_load("A")

    def generate_c_load(self):
        f = self.f
        wl(f, self.labels["load_c"])
        wi(f, "// Load matrix C")
        self._write_matrix_load("C")

    def generate_a_mult_b(self):
        # For Neon-sized row of B: (very first row uses fmul instead of fmla)
            # Load row of B
            # Multiply row by A
        # Multiply result by alpha
        f = self.f
        wl(f, self.labels["multiply_axb"])
        # How many Neon vectors fit into a column or row (depending on transpose) of B
        b_dim = self.params.n if self.params.trans.transb else self.params.k
        elements_per_neon_vec = int(16 / self.params.dtype.width)
        last_vec_lanes = (b_dim * self.params.dtype.width) % 16
        assert last_vec_lanes % self.params.dtype.width == 0
        last_vec_lanes = int(last_vec_lanes / self.params.dtype.width)

        # Load full vectors from B and perform multiplication
        if not self.params.trans.transb:
            self._a_mult_b_no_transpose(last_vec_lanes, elements_per_neon_vec)
        else:
            self._a_mult_b_transpose(last_vec_lanes, elements_per_neon_vec)

        # For real kernels only, check if alpha == 1.0 and don't perform multiplication if it is.
        # We can only check against floating point zero or another register, though.
        # It is safe to use the first floating point registers used for A and B to compare with alpha here
        if not self.params.dtype.is_complex:
            wi(f, "// If alpha is 1, we can skip (alpha * A * B)")
            wi(f, "fmov {}{}, #1.0".format(self.params.dtype.spec, self.assigned_regs.mat_a_regs[0][0]))
            wi(f, "fcmp {0}{1}, {0}{2}".format(self.params.dtype.spec, self.assigned_regs.alpha_reg, self.assigned_regs.mat_a_regs[0][0]))
            wi(f, "b.eq {}\n".format(self.labels["check_beta"]))

        # Multiply the result by alpha
        wi(f, "// Multiply the result by alpha")
        # By this point the registers used to store A and B (starting from z0) will be available
        # for reuse as temporary registers as we have finished doing AxB
        if self.params.dtype.is_complex:
            for row in self.assigned_regs.alpha_ab_res_regs:
                for reg in row:
                    wi(f, "dup z{}.{}, #0".format(reg, self.params.dtype.spec))
            for i, row in enumerate(self.assigned_regs.alpha_ab_res_regs):
                for j, reg in enumerate(row):
                    res_reg = self.assigned_regs.ab_res_regs[i][j]
                    # We do the conjugation here in the case of CC kernels because the FCMLA instruction
                    # doesn't support conjugating both input vectors
                    write_fcmla(self.f, reg, self.params.dtype.spec, res_reg, self.assigned_regs.alpha_reg,
                                conj_ab=(self.params.trans.conja and self.params.trans.conjb))
        else:
            for row in self.assigned_regs.alpha_ab_res_regs:
                for res_reg in row:
                    wi(f, "fmul z{0}.{1}, z{0}.{1}, z{2}.{1}".format(res_reg, self.params.dtype.spec, self.assigned_regs.alpha_reg))
        f.write("\n")

    def _a_mult_b_no_transpose(self, last_vec_lanes, elements_per_neon_vec):
        a_col_ind = 0
        # 112 is the largest offset allowed in an ld1rq operation. We need to see if we need to update the pointer
        # or not. If we need more than 7 Neon vectors to store a column / row of B, then we do need to update.
        base_offset = 0 if (self.assigned_regs.b_num_neon_vecs <= 7) else -8
        offsets = [base_offset + y for y in range(self.assigned_regs.b_num_neon_vecs)]
        for v in offsets:
            for j, reg in enumerate(self.assigned_regs.mat_b_regs):
                wi(self.f, "ld1rq{} {{z{}.{}}}, p_alltrue/z, [x{}, #{}]".format(self.params.dtype.suffix, reg, self.params.dtype.spec,
                                                                                self.assigned_regs.b_ptrs[j], v*16))
            self.f.write("\n")

            # For each element in the Neon vector, multiply by the appropriate column of A
            for i, row in enumerate(self.assigned_regs.ab_res_regs):
                wi(self.f, "// Neon row {} of B, SVE row {} of result matrix".format(v, i))
                for neon_lane in range(0, elements_per_neon_vec):
                    # Multiply the row of B by all of A
                    for j, res_reg in enumerate(row):
                        if self.params.dtype.is_complex:
                            write_fcmla(self.f, res_reg, self.params.dtype.spec, self.assigned_regs.mat_a_regs[i][a_col_ind + neon_lane],
                                        self.assigned_regs.mat_b_regs[j], conj_a=self.params.trans.conja,
                                        lane=None if self.params.dtype==Type.z else neon_lane)
                        else:
                            op = "fmul" if (neon_lane == 0 and v == base_offset) else "fmla"
                            wi(self.f, "{0} z{1}.{2}, z{3}.{2}, z{4}.{2}[{5}]".format(op, res_reg, self.params.dtype.spec,
                                                                                      self.assigned_regs.mat_a_regs[i][a_col_ind + neon_lane],
                                                                                      self.assigned_regs.mat_b_regs[j],
                                                                                      neon_lane))
                    self.f.write("\n")

            a_col_ind += elements_per_neon_vec

        # If the number of neon vectors divides exactly into the number of elements in a column (or row)
        # of B, there is nothing left to do. Otherwise last_vec_lanes will be non-zero and we have to do something
        if last_vec_lanes != 0:
            # We have a 'tail' to deal with. Set up a predicate with the correct number of lanes
            # We have to double the constant for complex cases as there are twice as many active elements
            wi(self.f, "// Get the predicate with the correct number of Neon lanes")
            predicate_lanes = 2*last_vec_lanes if self.params.dtype.is_complex else last_vec_lanes
            wi(self.f, "ptrue p_tail.{}, vl{}".format(self.params.dtype.spec, predicate_lanes))

            wi(self.f, "// Load Neon row {} of B".format(self.assigned_regs.b_num_neon_vecs))
            for j, reg in enumerate(self.assigned_regs.mat_b_regs):
                wi(self.f, "ld1rq{} {{z{}.{}}}, p_tail/z, [x{}, #{}]".format(self.params.dtype.suffix, reg, self.params.dtype.spec,
                                                                             self.assigned_regs.b_ptrs[j], (self.assigned_regs.b_num_neon_vecs + base_offset) * 16))
            self.f.write("\n")

            # Now perform the multiplication
            for i, row in enumerate(self.assigned_regs.ab_res_regs):
                wi(self.f, "// Neon row {} of B, SVE row {} of result matrix".format(self.assigned_regs.b_num_neon_vecs, i))
                for neon_lane in range(0, last_vec_lanes):
                    # Multiply the row of B by all of A
                    for j, res_reg in enumerate(row):
                        if self.params.dtype.is_complex:
                            write_fcmla(self.f, res_reg, self.params.dtype.spec, self.assigned_regs.mat_a_regs[i][a_col_ind + neon_lane],
                                        self.assigned_regs.mat_b_regs[j], conj_a=self.params.trans.conja, lane=neon_lane)
                        else:
                            op = "fmul" if (self.assigned_regs.b_num_neon_vecs == 0 and neon_lane == 0) else "fmla"
                            wi(self.f, "{0} z{1}.{2}, z{3}.{2}, z{4}.{2}[{5}]".format(op, res_reg, self.params.dtype.spec,
                                                                                      self.assigned_regs.mat_a_regs[i][a_col_ind + neon_lane],
                                                                                      self.assigned_regs.mat_b_regs[j],
                                                                                      neon_lane))
                    self.f.write("\n")

    def _a_mult_b_transpose(self, last_vec_lanes, elements_per_neon_vec):
        # The column index into the output matrix
        c_col_ind = 0
        # 112 is the largest offset allowed in an ld1rq operation. We need to see if we need to update the pointer
        # or not. If we need more than 7 Neon vectors to store a column / row of B, then we do need to update.
        base_offset = 0 if (self.assigned_regs.b_num_neon_vecs <= 7) else -8
        offsets = [base_offset + y for y in range(self.assigned_regs.b_num_neon_vecs)]
        # Loop over the number of Neon columns there are in B
        for v in offsets:
            # Load the Neon registers from a column of B
            for j, reg in enumerate(self.assigned_regs.mat_b_regs):
                wi(self.f, "ld1rq{} {{z{}.{}}}, p_alltrue/z, [x{}, #{}]".format(self.params.dtype.suffix, reg, self.params.dtype.spec,
                                                                                self.assigned_regs.b_ptrs[j], v * 16))
            self.f.write("\n")

            # For the case where we are doing single-precision complex NC/TC kernels we need to
            # conjugate the values from B as we cannot do it as part of the FCMLA instruction
            if self.params.dtype==Type.c and not self.params.trans.conja and self.params.trans.conjb:
                for reg in self.assigned_regs.mat_b_regs:
                    wi(self.f, "fneg z{0}.s, p_imag/m, z{0}.s".format(reg))
                self.f.write("\n")

            # Now we've loaded the column from B, we multiply by the appropriate column of A
            # Loop over the rows in A
            for i, row in enumerate(self.assigned_regs.mat_a_regs):
                # For each Neon lane we do
                wi(self.f, "// Neon column {} of B, SVE row {} of result matrix".format(v, i))
                for neon_lane in range(elements_per_neon_vec):
                    # For each register in the current row of A, multiply by the appropriate entry in B
                    for j, a_reg in enumerate(row):
                        if self.params.dtype.is_complex:
                            write_fcmla(self.f, self.assigned_regs.ab_res_regs[i][c_col_ind + neon_lane], self.params.dtype.spec, a_reg,
                                        self.assigned_regs.mat_b_regs[j], conj_a=self.params.trans.conja, conj_b=self.params.trans.conjb,
                                        lane=None if self.params.dtype==Type.z else neon_lane)
                        else:
                            op = "fmul" if j == 0 else "fmla"
                            wi(self.f, "{0} z{1}.{2}, z{3}.{2}, z{4}.{2}[{5}]".format(op, self.assigned_regs.ab_res_regs[i][c_col_ind + neon_lane],
                                                                                      self.params.dtype.spec, a_reg, self.assigned_regs.mat_b_regs[j], neon_lane))
            self.f.write("\n")
            c_col_ind += elements_per_neon_vec

        # If the number of neon vectors divides exactly into the number of elements in a column (or row)
        # of B, there is nothing left to do. Otherwise last_vec_lanes will be non-zero and we have to do something
        if last_vec_lanes != 0:
            # We have a 'tail' to deal with. Set up a predicate with the correct number of lanes
            # We have to double the constant for complex cases as there are twice as many active elements
            wi(self.f, "// Get the predicate with the correct number of Neon lanes")
            predicate_lanes = 2*last_vec_lanes if self.params.dtype.is_complex else last_vec_lanes
            wi(self.f, "ptrue p_tail.{}, vl{}".format(self.params.dtype.spec, predicate_lanes))

            wi(self.f, "// Load Neon column {} of B".format(self.assigned_regs.b_num_neon_vecs))
            for j, reg in enumerate(self.assigned_regs.mat_b_regs):
                wi(self.f, "ld1rq{} {{z{}.{}}}, p_tail/z, [x{}, #{}]".format(self.params.dtype.suffix, reg, self.params.dtype.spec,
                                                                             self.assigned_regs.b_ptrs[j], (self.assigned_regs.b_num_neon_vecs + base_offset) * 16))
            self.f.write("\n")

            # For the case where we are doing single-precision complex NC/TC kernels we need to
            # conjugate the values from B as we cannot do it as part of the FCMLA instruction
            if self.params.dtype==Type.c and not self.params.trans.conja and self.params.trans.conjb:
                for reg in self.assigned_regs.mat_b_regs:
                    wi(self.f, "fneg z{0}.s, p_imag/m, z{0}.s".format(reg))
                self.f.write("\n")

            # Now perform the multiplication
            for i, row in enumerate(self.assigned_regs.mat_a_regs):
                wi(self.f, "// Neon column {} of B, SVE row {} of result matrix".format(self.assigned_regs.b_num_neon_vecs, i))
                for neon_lane in range(last_vec_lanes):
                    # Multiply the row of B by all of A
                    for j, a_reg in enumerate(row):
                        if self.params.dtype.is_complex:
                            write_fcmla(self.f, self.assigned_regs.ab_res_regs[i][c_col_ind + neon_lane], self.params.dtype.spec, a_reg,
                                        self.assigned_regs.mat_b_regs[j], conj_a=self.params.trans.conja, conj_b=self.params.trans.conjb,
                                        lane=neon_lane)
                        else:
                            op = "fmul" if j == 0 else "fmla"
                            wi(self.f, "{0} z{1}.{2}, z{3}.{2}, z{4}.{2}[{5}]".format(op, self.assigned_regs.ab_res_regs[i][c_col_ind + neon_lane],
                                                                                      self.params.dtype.spec, a_reg, self.assigned_regs.mat_b_regs[j], neon_lane))
                    self.f.write("\n")

    def generate_check_beta(self):
        f = self.f
        wl(f, self.labels["check_beta"])
        # At this point we can load the memory addresses of the columns of C, as we are done with A and B,
        # and all code paths need to go through the beta check
        self.generate_col_ptrs(False, False, True, self.params.dtype)

        # If beta is zero, branch to storing C
        wi(f, "// If beta is 0, we can skip (C * beta)")
        if self.params.dtype.is_complex:
            wi(f, "fcmne p_comp.{0}, p_alltrue/z, z{1}.{0}, #0.0".format(self.params.dtype.spec, self.assigned_regs.beta_reg))
            wi(f, "ptest p_alltrue, p_comp.b")
        else:
            wi(f, "fcmp {}{}, #0.0".format(self.params.dtype.spec, self.assigned_regs.beta_reg))
        wi(f, "b.eq {}\n".format(self.labels["store_c"]))

    def generate_multiply_beta_c(self):
        f = self.f
        wl(f, self.labels["multiply_c"])
        for i, row in enumerate(self.assigned_regs.abpc_res_regs):
            for j, res_reg in enumerate(row):
                if self.params.dtype.is_complex:
                    write_fcmla(self.f, res_reg, self.params.dtype.spec, self.assigned_regs.c_load_regs[i][j], self.assigned_regs.beta_reg)
                else:
                    wi(f, "fmla z{0}.{1}, p_alltrue/m, z{2}.{1}, z{3}.{1}".format(res_reg, self.params.dtype.spec, self.assigned_regs.c_load_regs[i][j], self.assigned_regs.beta_reg))
            f.write("\n")

    def store_result(self):
        f = self.f
        wl(f, self.labels["store_c"])

        # We can't use an offset of greater than 7 (multiples of the vector length).
        num_mat_regs = len(self.assigned_regs.abpc_res_regs)
        col_offset = 0 if num_mat_regs <= 8 else 8

        # We know that we have the pointers to columns C set up correctly at this point
        # Store the full vectors
        for i, row in enumerate(self.assigned_regs.abpc_res_regs[:-1]):
            for j, reg in enumerate(row):
                wi(f, "st1{0} z{1}.{2}, p_alltrue, [x{3}, #{4}, MUL VL]".format(self.params.dtype.suffix, reg, self.params.dtype.spec,
                                                                                self.assigned_regs.c_col_ptrs[j], i - col_offset))
            f.write("\n")

        # The last vector is predicated, and we know that p_lastvec stores the correct predicate
        wi(f, "// The last row may need predication")

        for j, reg in enumerate(self.assigned_regs.abpc_res_regs[-1]):
            wi(f, "st1{0} z{1}.{2}, p_lastvec, [x{3}, #{4}, MUL VL]".format(self.params.dtype.suffix, reg, self.params.dtype.spec,
                                                                            self.assigned_regs.c_col_ptrs[j], len(self.assigned_regs.abpc_res_regs) - 1 - col_offset))
        f.write("\n")
### End of class AtomicGEMMKernelWriterSVE

def write_base_gpregs(f, dtype):
    """ Base registers are those which are invariant for all of the methods
        in the file to be generated for the datatype passed. For real-valued
        methods, we have the parameters passed to the function, and we also want
        to read a multiple of the current vector length to be able to work out
        how full the final vector is
    """
    f.write("// parameters and shared variable names\n")
    gp_names = ['a_ptr', 'lda', 'b_ptr', 'ldb', 'c_ptr', 'ldc', 'm_dim', 'curr_vl']
    for i, name in enumerate(gp_names):
        f.write("#define {} x{}\n".format(name, i))

def write_predregs(f, dtype):
    """ Give some meaningful names to the predicate registers we will be using
    """
    pred_names = [ 'p_alltrue', 'p_lastvec', 'p_tail', 'p_comp', 'p_imag' ]
    for i, name in enumerate(pred_names):
        f.write("#define {} p{}\n".format(name, i))

def write_fcmla(f, res_reg, spec, first_src_reg, second_src_reg,
                conj_a=False, conj_b=False, conj_ab=False, lane=None):
    """ Write out the pairs of FCMLA instructions required to implement a single complex fused multiply add
        operation. In cases that require taking the conjugate of the second source register we must swap the order of
        the source registers in the FCMLA instruction as this only supports conjugating its first source register.
        However this swapping of registers does not work with indexed FCMLAs because we can only specify individual
        lanes for the second register; for indexed FCMLAs we must instead explicitly conjugate the values stored in
        the second register before calling this function.
    """
    if (conj_a and not conj_b) or conj_ab:
        # CN/CT kernels or alpha*conj(A*B)
        input_reg = [ first_src_reg, second_src_reg ]
        rotations = [ "0", "270" ]
    elif lane is None and conj_b and not conj_a:
        # NC/TC kernels with non-indexed FCMLA
        input_reg = [ second_src_reg, first_src_reg ]
        rotations = [ "0", "270" ]
    else:
        # The only options left are:
        #   Indexed NC/TC kernels
        #   Non-indexed CC/NN/NT/TN/TT kernels
        # In the CC case we do a non-conjugate A*B multiplication first
        # and then do alpha*conj(A*B) later
        assert (lane is not None and conj_b and not conj_a) or (conj_a==conj_b)
        input_reg = [ first_src_reg, second_src_reg ]
        rotations = [ "0", "90" ]

    for r in rotations:
        if lane is None:
            wi(f, "fcmla z{0}.{1}, p_alltrue/m, z{2}.{1}, z{3}.{1}, #{4}".format(res_reg, spec, input_reg[0], input_reg[1], r))
        else:
            # write an indexed FCMLA when a lane is provided
            wi(f, "fcmla z{0}.{1}, z{2}.{1}, z{3}.{1}[{4}], #{5}".format(res_reg, spec, input_reg[0], input_reg[1], lane, r))

def generate_source_files_sve(atom_kern_params_list_sve, source_filenames):
    for dtype, source_filename in source_filenames.items():
        with open(source_filename, 'w') as f:
            # Filter the parameters for the datatype
            dtype_params = [
                params for params in atom_kern_params_list_sve
                if params.dtype == dtype]

            # Get the register info for the datatype. This tells us how many
            # registers are available in total, and how many are reserved for
            # things like parameters and variables common to all routines.
            reg_info = AtomicKernelRegInfoSVE(dtype)
            # First thing we write in the file is definitions of the registers
            # which are not going to change
            write_base_gpregs(f, dtype)
            f.write("\n")
            write_predregs(f, dtype)
            f.write("\n\n")

            for params in dtype_params:
                # Figure out which registers are to be used to store which part
                # of which matrix
                assigned_regs = AtomicKernelAssignedRegistersSVE(reg_info, params)
                # Write the kernel
                kernel_writer = AtomicGEMMKernelWriterSVE(f, params, assigned_regs)
                kernel_writer.write_kernel()

def valid_atomic_kern_params_sve(dtypes, target_os):
    valid_dtypes = {Type.s, Type.d, Type.c, Type.z}
    def gen_params_list(dtype, target_os):
        reg_info = AtomicKernelRegInfoSVE(dtype)
        params_t = AtomicKernelParamsSVE
        transposes = Transpose.complex_transposes() if dtype.is_complex else Transpose.real_transposes()
        params_list = [params_t(m_vecs, n, k, trans, dtype, target_os)
                        for trans in transposes
                        for m_vecs, n, k in reg_info.generate_valid_kernel_dims(trans)]
        return params_list

    if (dtypes):
        valid_dtypes = valid_dtypes.intersection(dtypes)
        sve_params = []
        for dtype in valid_dtypes:
            sve_params.extend(gen_params_list(dtype, target_os))
        return sve_params, valid_dtypes
    else:
        # If no dtypes are specified then kernels for all valid dtypes are generated by default
        return [p for dtype in valid_dtypes for p in gen_params_list(dtype, target_os)], valid_dtypes
