# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import json
import sys

import routine_signature

def add_to_namespace(namespaces, content):
	out=content
	for ns in reversed(namespaces):
		out=f"""namespace {ns} {{
{out}
}}"""
	return out

def make_chooser_cases(archs, routine_name, template_actual_parameters):
	return "\n".join( [
		f"case architectures::{arch}: return {routine_name}<{template_actual_parameters}, linalg::spec::{arch}_architecture_spec>;"
			for arch in archs ])

def make_architecture_spec_declarations(archs):
	return "\n".join(f"struct {arch}_architecture_spec;" for arch in archs)

def generate_chooser(json_file_path, architecture_json_file_path, output):
	with open(json_file_path) as f:
		j = json.load(f)
	template_formal_args = routine_signature.make_template_formal_args(j["template_parameters"])
	template_actual_args = routine_signature.make_template_actual_args(j["template_parameters"])

	function_params = routine_signature.make_function_parameters_unresolved(j["parameters"],
		[
			t["name"] for t
			in j["template_parameters"]
			if t["name"] not in [ "IntType", "ParamCheck" ]
		])

	with open(architecture_json_file_path) as arch_json_file:
		archs = json.load(arch_json_file)

	qualified_name = "::".join(j["namespaces"] + [j["name"]])

	chooser_cases = make_chooser_cases(archs["architectures"], qualified_name, template_actual_args)
	architecture_spec_declarations = make_architecture_spec_declarations(archs["architectures"])

	if "includes" in j:
		additional_includes="\n".join([f'#include "{inc}"' for inc in j["includes"]])
	else:
		additional_includes=""


	function_declaration = add_to_namespace(j["namespaces"],
		f"""
template<{template_formal_args}, typename ArchSpec>
{j["return_type"]} {j["name"]}({function_params});
""")


	chooser_code = f"""

#include "get_architecture.hpp"

#include "detect/system.hpp"

//this should be the only LINALG include
//and linalg_util should not include any
//other linalg headers
#include "framework/linalg_util.hpp"

#include "perflibs_float.hpp"
#include "perflibs_complex.hpp"
#include "perflibs_ftn_symb.hpp"

#include "pl_linalg_complex.h"

{additional_includes}

{function_declaration}

namespace perflibs {{
namespace linalg {{


namespace spec {{

{architecture_spec_declarations}

}} //namespace spec

}} //linalg

namespace chooser {{


template<{template_formal_args}>
static auto {j["name"]}_set_fptr() {{
	auto architecture = get_architecture();

	switch(architecture) {{
		default:
		{chooser_cases}
	}}
}}

template<{template_formal_args}>
static auto {j["name"]}_fptr = {j["name"]}_set_fptr<{template_actual_args}>();

}} //namespace chooser
}} //namespace perflibs
"""

	output.write(chooser_code)


def make_function_parameter(param_data, template_types, template_values):
	out = ""
	if param_data["is_const"]:
		out += "const "

	if param_data["type"] in template_values:
		param_type = template_values[ param_data["type"] ]
	else:
		param_type = param_data["type"]

	out += f"{param_type} "

	if  param_data["is_pointer"]:
		out += "*"

	if "is_reference" in param_data and param_data["is_reference"]:
		out += "&"

	out += param_data["name"]

	return out


def make_function_parameters(params_data, template_types, template_values):
	out = []
	for param in params_data:
		out.append(make_function_parameter(param, template_types, template_values))

	return ", ".join(out)

def make_actual_function_parameters(params_data):
	return ", ".join([ x["name"] for x in params_data ])

def make_function_stub(routine_name, j, param_pack_idx):
	routine_data = j["instantiations"][routine_name]

	if j["return_type"] in [ x["name"] for x in j["template_parameters"] ]:
		return_type = routine_data["template_values"][ j["return_type"] ]
	else:
		return_type = j["return_type"]


	if routine_data["symbol_type"] == "fortran":
		symbol_name = f"{routine_name}_"
	else:
		symbol_name = f"{routine_name}"

	template_values = routine_data["template_values"][param_pack_idx] if isinstance(routine_data["template_values"], list) else routine_data["template_values"]

	template_actual_args = routine_signature.make_resolved_template_actual_args(
		template_values, j["template_parameters"])

	actual_function_args = make_actual_function_parameters(j["parameters"])

	parameters = make_function_parameters(
		j["parameters"],
		j["template_parameters"],
		routine_data["template_values"])

	cplx_return_map = {
		"c16" : "perflibs_halfcomplex_t",
		"c32" : "pl_linalg_singlecomplex_t",
		"c64" : "pl_linalg_doublecomplex_t"
	}

	# GNU-style complex return ABI (return by value, not by parameter pointer).
	if return_type in cplx_return_map:
		return_type = cplx_return_map[ return_type ]
		return f"""
{return_type} {symbol_name}({parameters}) {{
	auto ret = perflibs::chooser::{j["name"]}_fptr<{template_actual_args}>({actual_function_args});
	return {{ ret.real(), ret.imag() }};
}}
"""

	return f"""
{return_type} {symbol_name}({parameters}) {{
	return perflibs::chooser::{j["name"]}_fptr<{template_actual_args}>({actual_function_args});
}}
"""


def make_template_instantiation(name, j):
	template_formal_args = routine_signature.make_template_formal_args(j["template_parameters"])
	template_actual_args = routine_signature.make_template_actual_args(j["template_parameters"])
	actual_function_args = routine_signature.make_actual_function_parameters(j["parameters"])

	function_params_unresolved = routine_signature.make_function_parameters_unresolved(j["parameters"],
		[
			t["name"] for t
			in j["template_parameters"]
			if t["name"] not in [ "IntType", "ParamCheck" ]
		])



	out = f"""
using perflibs::bf16, perflibs::r16, perflibs::r32, perflibs::r64, perflibs::c16, perflibs::c32, perflibs::c64;

template<{template_formal_args}>
{j["return_type"]} {name}({function_params_unresolved}) {{
	return perflibs::chooser::{j["name"]}_fptr<{template_actual_args}>({actual_function_args});
}}
"""
	for tv in j["instantiations"][name]["template_values"]:
		function_params_resolved = routine_signature.make_function_parameters(j["parameters"],
			[
				t for t
				in j["template_parameters"]
				if t["name"] not in [ "IntType", "ParamCheck" ]
			], tv)

		template_values = []
		for tp in j["template_parameters"]:
			template_values.append(tv[tp["name"]])
		template_values = ", ".join(template_values)

		out +=  f"template {j['return_type']} {name}<{template_values}>({function_params_resolved});\n"

	return add_to_namespace(j["namespaces"], out)



def make_function_stubs(routine_json_file_path, output):
	with open(routine_json_file_path) as f:
		j = json.load(f)

	c_instantiations = ""
	cxx_instantiations = ""

	for name, data in j["instantiations"].items():
		if data["symbol_type"] == "c++template":
			 cxx_instantiations += make_template_instantiation(name, j)
		elif isinstance(data["template_values"], list):
			for i in range(0, len(data["template_values"])):
				c_instantiations += make_function_stub(name, j, i)
		else:
			c_instantiations += make_function_stub(name, j, 0)

	out = ""
	if cxx_instantiations:
		out += cxx_instantiations
	if c_instantiations:
		out = f"""
extern "C" {{
using perflibs::bf16, perflibs::r16, perflibs::r32, perflibs::r64, perflibs::c16, perflibs::c32, perflibs::c64;

{c_instantiations}

}} //extern "C"
"""
	output.write(out)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--routine-json-file-path")
	parser.add_argument("--architectures-json-file-path")
	parser.add_argument("--output-file-path")

	args = parser.parse_args()

	if args.output_file_path is not None:
		with open(args.output_file_path, "w") as f:
			generate_chooser(args.routine_json_file_path, args.architectures_json_file_path, f)
			make_function_stubs(args.routine_json_file_path, f)
	else:
		generate_chooser(args.routine_json_file_path, args.architectures_json_file_path, sys.stdout)
		make_function_stubs(args.routine_json_file_path, sys.stdout)

if __name__ == "__main__":
	main()
