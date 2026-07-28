# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import sys
import json
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class GemmKernelType:
	kernel_type: str
	types: Tuple[Any, ...]

	def get_type_list(self):
		return ", ".join(self.types)

@dataclass
class MatrixInterleaveSpec:
	datatype: str
	cntg_interleave: int | str
	strd_interleave: int | str
	split_factor: int
	strd_unroll: int
	strd_interleave_step: int
	cntg_interleave_step: int

	def get_cntg_interleave(self):
		if isinstance(self.cntg_interleave, int):
			return f"intval<{self.cntg_interleave}>"
		return f"{self.cntg_interleave}"

	def get_strd_interleave(self):
		if isinstance(self.strd_interleave, int):
			return f"intval<{self.strd_interleave}>"
		return f"{self.strd_interleave}"

	@property
	def matrix_req(self):
		return "matrix_requirement::{}_one".format("strd" if self.strd_interleave_step == 1 else "cntg")

	def to_cpp_spec(self):
		return f"matrix_interleave_spec<{self.datatype}> {{ {self.get_cntg_interleave()}, {self.get_strd_interleave()}, {self.cntg_interleave_step}_ki, {self.strd_interleave_step}_ki, {self.matrix_req}, {self.split_factor}_ki, {self.strd_unroll}_ki }}"

@dataclass
class GemmKernelSpec:
	kernel_type: str
	symbol: str
	c_type: str
	cntg_unroll: int


	a_interleave_spec: MatrixInterleaveSpec
	b_interleave_spec: MatrixInterleaveSpec

	apply_beta: str
	extensions: str

	def get_full_kernel_type(self) -> GemmKernelType:
		a_interleave_spec =	MatrixInterleaveSpec( **self.a_interleave_spec )
		b_interleave_spec =	MatrixInterleaveSpec( **self.b_interleave_spec )

		return GemmKernelType( self.kernel_type, ( a_interleave_spec.datatype, b_interleave_spec.datatype, self.c_type ) )

	def to_cpp_spec(self):
		a_interleave_spec =	MatrixInterleaveSpec( **self.a_interleave_spec )
		b_interleave_spec =	MatrixInterleaveSpec( **self.b_interleave_spec )

		a_interleave_spec_cpp = a_interleave_spec.to_cpp_spec()
		b_interleave_spec_cpp = b_interleave_spec.to_cpp_spec()

		return f"interleave_matmul_kernel_spec<{a_interleave_spec.datatype}, {b_interleave_spec.datatype}, {self.c_type}> {{ &{self.symbol}, {self.cntg_unroll}_ki, {a_interleave_spec_cpp}, {b_interleave_spec_cpp}, false, value_support::{self.apply_beta}, extensions::{self.extensions} }}"

	def to_function_fwd_decl(self):

		a_type = MatrixInterleaveSpec( **self.a_interleave_spec ).datatype
		b_type = MatrixInterleaveSpec( **self.b_interleave_spec ).datatype
		return f"perflibs::linalg::interleave_matmul_kernel<{a_type}, {b_type}, {self.c_type}> {self.symbol};"

with open(sys.argv[1]) as f:
	j = json.load(f)

kernels = {}
symbol_decl_list_cpp=set()

for interleave_matmul_kernel_spec in ( GemmKernelSpec(**e) for e in j ):
	kernel_type = interleave_matmul_kernel_spec.get_full_kernel_type()
	kernels.setdefault(kernel_type, []).append(interleave_matmul_kernel_spec)

	symbol_decl_list_cpp.add(interleave_matmul_kernel_spec.to_function_fwd_decl())


symbol_decl_list_cpp = "\n\t".join(sorted(symbol_decl_list_cpp))

kernel_spec_list_cpp=[]
for kernel_type, kernel_specs in kernels.items():
	cpp_kernel_specs     = ",\n\t".join( ( x.to_cpp_spec() for x in kernel_specs ) )

	cpp_kernel_indices = []

	for i, kernel_spec in enumerate(kernel_specs):
		cpp_kernel_indices.append(
			f"{kernel_spec.symbol}_idx = {i}zu")

	cpp_kernel_indices = ", ".join(cpp_kernel_indices)

	kernel_spec_list_cpp.append(f"""
template<typename ArchitectureSpec>
const auto interleave_matmul_kernel_specs<interleave_matmul_kernel_spec<{kernel_type.get_type_list()}>, ArchitectureSpec> = std::array {{
	{cpp_kernel_specs}
}};
constexpr std::size_t {cpp_kernel_indices};""")

kernel_spec_list_cpp= "\n".join(kernel_spec_list_cpp)


print(f"""
#ifndef PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_HPP
#define PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_HPP

#include "matmul3_kernels_pre.hpp"

#include <array>

extern "C" {{
	using perflibs::bf16;
	using perflibs::r16;
	using perflibs::r32;
	using perflibs::r64;
	using perflibs::c32;
	using perflibs::c64;

	{symbol_decl_list_cpp}
}} // extern "C"

namespace perflibs::linalg {{
{kernel_spec_list_cpp}

}} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_FRAMEWORK_MATMUL3_KERNELS_HPP
""")
