# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import math

from enum import Enum, unique

@unique
class Transpose(Enum):
    NN = 0, 'NN'
    NT = 1, 'NT'
    TN = 2, 'TN'
    TT = 3, 'TT'
    CN = 4, 'CN'
    CT = 5, 'CT'
    NC = 6, 'NC'
    TC = 7, 'TC'
    CC = 8, 'CC'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        member.transa = name[0] != 'N'
        member.conjb = name[1] == 'C'
        member.transb = name[1] != 'N'
        member.conja = name[0] == 'C'
        return member

    def __int__(self):
        return self.value

    def __lt__(self, other):
        return self.value < other.value

    @property
    def perflibs_constants(self):
        constants = {
            Transpose.NN: ("perflibs::linalg::PERFLIBS_NOTRANS", "perflibs::linalg::PERFLIBS_NOTRANS"),
            Transpose.NT: ("perflibs::linalg::PERFLIBS_NOTRANS", "perflibs::linalg::PERFLIBS_TRANS"),
            Transpose.TN: ("perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_NOTRANS"),
            Transpose.TT: ("perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_TRANS"),
            Transpose.CN: ("perflibs::linalg::PERFLIBS_CONJTRANS", "perflibs::linalg::PERFLIBS_NOTRANS"),
            Transpose.CT: ("perflibs::linalg::PERFLIBS_CONJTRANS", "perflibs::linalg::PERFLIBS_TRANS"),
            Transpose.NC: ("perflibs::linalg::PERFLIBS_NOTRANS", "perflibs::linalg::PERFLIBS_CONJTRANS"),
            Transpose.TC: ("perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_CONJTRANS"),
            Transpose.CC: ("perflibs::linalg::PERFLIBS_CONJTRANS", "perflibs::linalg::PERFLIBS_CONJTRANS"),
        }
        return constants[self]

    @property
    def equiv_tran(self):
        constants = {
            Transpose.NC: Transpose.NT,
            Transpose.CN: Transpose.TN,
            Transpose.CC: Transpose.TT,
            Transpose.CT: Transpose.TT,
            Transpose.TC: Transpose.TT
        }
        if self not in constants.keys():
            raise TypeError(
                "Equivalent trans is only applicable to conj types.")
        return constants[self]

    @staticmethod
    def complex_transposes():
        return Transpose.__members__.values()

    @staticmethod
    def real_transposes():
        return [Transpose.NN, Transpose.TN, Transpose.NT, Transpose.TT]


@unique
class Type(Enum):
    """Type enum representing supported types.

    Types
    =====
    s=single 32-bit (float)
    d=double 64-bit
    c=complex single 64-bit (float)
    z=complex double 128-bit

    """
    s, d, c, z = range(4)

    @property
    def width(self):
        """Return type width in bytes."""
        if self.value == 0:
            return 4
        elif self.value < 3:
            return 8
        else:
            return 16

    @property
    def max_neon_lanes(self):
        """Returns the number of lanes the vector is split into for the
        given type.
        """
        return 16 // self.width

    @property
    def is_complex(self):
        return self in (Type.c, Type.z)

    @property
    def real_max_neon_lanes(self):
        """Returns the number of lanes the vector is split into for the real
        and imaginary parts only i.e. for single complex, max lanes is 2
        (given each complex size is 64 bits), however each 64 bit complex
        number is split up into 2 x 32bit real and imaginary components hence
        we can fit double the number of elements when dealing with
        real/imaginary components on their own
        """
        if not self.is_complex:
            raise TypeError(
                "Real max lanes only applies to complex types")
        return 2 * self.max_neon_lanes

    @property
    def c_type(self):
        if self.value == 0:
            return 'float'
        elif self.value == 1:
            return 'double'
        elif self.value == 2:
            return 'std::complex<float>'
        else:
            return 'std::complex<double>'

    @property
    def base_vreg(self):
        """Returns the index of the first unused vector register.

        Alpha and beta parameters are stored in v0 and v1 if type is single or
        double, therefore v2 is first available vector register.  For complex
        types alpha and beta parameters use v0-v3, the vectors required is
        doubled to the store real and imaginary components.
        """
        if not self.is_complex:
            return 2
        else:
            return 4

    @property
    def spec(self):
        """Type specifier"""
        if self is Type.c:
            return 's'
        elif self is Type.z:
            return 'd'
        else:
            return self.name

    @property
    def spec2x(self):
        """Next largest type specifier"""
        if self is Type.s or self is Type.c:
            return 'd'
        elif self is Type.d or self is Type.z:
            return 'q'

    @property
    def suffix(self):
        """ The suffix for assembly instructions for the datatype """
        if self in [Type.c, Type.s]:
            return 'w'
        else:
            return 'd'

class AtomicKernelRegInfo(object):
    """ Stores constant data about the number of registers available, as well
        as the number of base registers used for kernels for a target (currently one of Neon or SVE)
    """

    def __init__(self, num_gpreg, base_gpreg, num_vreg, base_vreg):
        self.num_gpreg = num_gpreg
        self.base_gpreg = base_gpreg
        self.num_vreg = num_vreg
        self.base_vreg = base_vreg

class AtomicGEMMKernelWriter(object):
    def __init__(self, f, params, assigned_regs):
        self.f = f
        self.params = params
        self.assigned_regs = assigned_regs

    def write_kernel(self):
        raise NotImplementedError("Not implemented write_kernel function for this data type")
#### End of class AtomicGEMMKernelWriter

class AtomicGEMMKernelParams(object):
    def __init__(self):
        self.dtype = Type.s
        # Nothing to do in here
        return

    @property
    def routine_name(self):
        raise NotImplementedError("Need to override the routine_name property")

    @property
    def c_function_signature(self):
        return ('void {0}(const {1} *a, kernel_inttype lda, '
                'const {1} *b, kernel_inttype ldb, {1} *c, '
                'kernel_inttype ldc, {1} alpha, {1} beta);'.format(
                    self.routine_name, self.dtype.c_type))


    @property
    def function_signature(self):
        return 'extern "C" {}'.format(self.c_function_signature)

# Helper functions for writing to a file
def wi(f, string):
    # Write indent - write line to file (f) with leading tab space
    f.write("\t{}\n".format(string))

def wl(f, string):
    # Write label - write line to file (f) with trailing colon
    f.write("{}:\n".format(string))

def generate_dimension_shift(f, dtype):
    # multiply lda, ldb, ldc by data type width for indexing
    f.write('\n')
    wi(f, "// convert to bytes")
    ld_shift = int(math.log2(dtype.width))
    for dimension in ('lda', 'ldb', 'ldc'):
        wi(f, "lsl {0}, {0}, #{1}".format(dimension, ld_shift))
