#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple


InterleaveValue = int | dict[str, int]


@dataclass(frozen=True)
class GemmKernelType:
	kernel_type: str
	types: Tuple[Any, ...]

	def get_type_list(self):
		return ", ".join(self.types)

@dataclass
class MatrixInterleaveSpec:
	datatype: str
	cntg_interleave: InterleaveValue
	strd_interleave: InterleaveValue
	split_factor: int
	strd_unroll: int
	strd_interleave_step: int
	cntg_interleave_step: int

	def get_interleave(self, value: InterleaveValue, extension: str):
		if isinstance(value, int):
			return f"intval<{value}>"
		multiplier = value["vector_length_multiplier"]
		return f"vl<extensions::{extension}, {self.datatype}, {multiplier}>"

	@property
	def matrix_req(self):
		return "matrix_requirement::{}_one".format("strd" if self.strd_interleave_step == 1 else "cntg")

	def to_cpp_spec(self, extension: str):
		cntg_interleave = self.get_interleave(self.cntg_interleave, extension)
		strd_interleave = self.get_interleave(self.strd_interleave, extension)
		return (
			f"matrix_interleave_spec<{self.datatype}> {{ "
			f"{cntg_interleave}, {strd_interleave}, "
			f"{self.cntg_interleave_step}_ki, {self.strd_interleave_step}_ki, "
			f"{self.matrix_req}, {self.split_factor}_ki, "
			f"{self.strd_unroll}_ki }}"
		)

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

		a_interleave_spec_cpp = a_interleave_spec.to_cpp_spec(self.extensions)
		b_interleave_spec_cpp = b_interleave_spec.to_cpp_spec(self.extensions)

		return f"interleave_matmul_kernel_spec<{a_interleave_spec.datatype}, {b_interleave_spec.datatype}, {self.c_type}> {{ &{self.symbol}, {self.cntg_unroll}_ki, {a_interleave_spec_cpp}, {b_interleave_spec_cpp}, false, value_support::{self.apply_beta}, extensions::{self.extensions} }}"

	def to_function_fwd_decl(self):

		a_type = MatrixInterleaveSpec( **self.a_interleave_spec ).datatype
		b_type = MatrixInterleaveSpec( **self.b_interleave_spec ).datatype
		return f"perflibs::linalg::interleave_matmul_kernel<{a_type}, {b_type}, {self.c_type}> {self.symbol};"

parser = argparse.ArgumentParser()
parser.add_argument("input_json", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

with args.input_json.open(encoding="utf-8") as f:
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


generated_header = f"""
#ifndef PERFLIBS_LINALG_KERNEL_SPECS_MATMUL3_KERNELS_HPP
#define PERFLIBS_LINALG_KERNEL_SPECS_MATMUL3_KERNELS_HPP

#include "kernel_specs/matmul3_kernel_spec.hpp"

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

#endif //PERFLIBS_LINALG_KERNEL_SPECS_MATMUL3_KERNELS_HPP
"""

args.output.write_text(generated_header, encoding="utf-8")
