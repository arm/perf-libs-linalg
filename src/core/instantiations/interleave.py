# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


def make_flag(flag_str):
	return [ x.removeprefix("interleave_flags::") for x  in flag_str.split("|") ]

data_type_interleave_factors = {
	"r16": [ 8, 16, 24, 32 ],
	"r32": [ 6, 8, 9, 12, 16, 32 ],
	"r64": [ 2, 6, 8, 9, 20 ],
	"c32": [ 4, 10, 8, 16 ],
	"c64": [ 4, 5 ]
}

#we should really build up this list by reading the gemm kernel specs
#and find out which interleave factors we need for each dattype
#and only build that
data_type_interleave_factors = {
	"bf16": [ 2, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32 ],
	"r16": [ 2, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32 ],
	"r32": [ 2, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32 ],
	"r64": [ 2, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32, 64 ],
	"c32": [ 2, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32 ],
	"c64": [ 2, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32 ]
}

flags = {
	"interleave_flags::DiagVirt|interleave_flags::DiagReal|interleave_flags::LowerVirt":  [ "c32", "c64" ],
	"interleave_flags::DiagVirt|interleave_flags::DiagReal|interleave_flags::UpperVirt" : [ "c32", "c64" ],
	"interleave_flags::DiagVirt|interleave_flags::DiagUnit|interleave_flags::LowerVirt|interleave_flags::VirtZero": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::DiagVirt|interleave_flags::DiagUnit|interleave_flags::UpperVirt|interleave_flags::VirtZero": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::DiagVirt|interleave_flags::LowerVirt": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::DiagVirt|interleave_flags::UpperVirt": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::IsConj": [ "c32", "c64" ],
	"interleave_flags::IsConj|interleave_flags::DiagVirt|interleave_flags::LowerVirt": [ "c32", "c64" ],
	"interleave_flags::IsConj|interleave_flags::DiagVirt|interleave_flags::UpperVirt": [ "c32", "c64" ],
	"interleave_flags::LowerVirt": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::LowerVirt|interleave_flags::VirtZero": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::None": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::UpperVirt": [ "bf16", "r16", "r32", "r64", "c32", "c64" ],
	"interleave_flags::UpperVirt|interleave_flags::VirtZero": [ "bf16", "r16", "r32", "r64", "c32", "c64" ]
}

functions = [
	( "n_cpp_interleave", "template void n_cpp_interleave<{interleave_factor}, kernel_inttype({flag}), {src_data_type}, {dst_data_type}, ARCHITECTURE_SPEC>(std::size_t, std::size_t, {src_data_type} const*, std::size_t, std::size_t, std::size_t, std::size_t, {dst_data_type}*, std::size_t, kernel_inttype, kernel_inttype);" ),
	( "t_cpp_interleave", "template void t_cpp_interleave<{interleave_factor}, kernel_inttype({flag}), {src_data_type}, {dst_data_type}, ARCHITECTURE_SPEC>(std::size_t, std::size_t, {src_data_type} const*, std::size_t, std::size_t, std::size_t, std::size_t, {dst_data_type}*, std::size_t, kernel_inttype, kernel_inttype);" )
]

def make_flag_string(flags):
	return "_".join( ( x.lower() for x in flags ) )

def make_macro_name(func_name, flags):
	flag_string = make_flag_string(flags)
	return f"LINALG_INTERLEAVE_{func_name}_{flag_string}"

print("""
#ifndef PERFLIBS_LINALG_INSTANTIATIONS_INTERLEAVE_HPP
#define PERFLIBS_LINALG_INSTANTIATIONS_INTERLEAVE_HPP


#include "perflibs_float.hpp"
#include "perflibs_complex.hpp"

#define PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_ACTIVELY_REQUESTED =1
#include "framework/interleave_fallback_kernel.hpp"
""")

for function_name, function_proto in functions:
	for flag, data_types in flags.items():
		functions = []
		macro_name = make_macro_name(function_name, make_flag(flag))
		for data_type in data_types:
			src_data_type = data_type
			dst_data_type = data_type
			for interleave_factor in data_type_interleave_factors[ data_type ]:
				functions.append( function_proto.format(**locals()) )
		bs = "\\"
		bsnl = "\\\n"

		print(f"""
#define {macro_name}(ARCHITECTURE_SPEC){bs}
namespace perflibs::linalg {{ {bs}
{bsnl.join(functions)} {bs}
}} //namespace perflibs::linalg
""")

print("#endif //PERFLIBS_LINALG_INSTANTIATIONS_INTERLEAVE_HPP")
