# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import json
import sys
import os

import routine_signature

def generate_instantiation(routines_json_directory, instantiations_json_file_path, architecture_name, output):
	with open(instantiations_json_file_path) as f:
		instantiations = json.load(f)

	out = []
	interface_includes = set()
	for instantiation in instantiations:

		routine_json_file_path = os.path.join(routines_json_directory, f"{instantiation['routine']}.json")

		with open(routine_json_file_path) as f:
			routine_data = json.load(f)

			return_type = routine_data["return_type"]
			return_type = routine_signature.resolve_template(
				routine_data["return_type"],
				routine_data["template_parameters"],
				routine_data["instantiations"][instantiation['symbol']]["template_values"])

			if routine_data["instantiations"][instantiation['symbol']]['symbol_type'] == "c++template":
				template_values = instantiation["template_values"]
			else:
				template_values = routine_data["instantiations"][instantiation['symbol']]["template_values"]

			actual_template_args = routine_signature.make_resolved_template_actual_args(
				template_values, routine_data["template_parameters"])

			parameters = routine_signature.make_function_parameters(
				routine_data["parameters"], routine_data["template_parameters"], template_values)

			qualified_routine_name = "::".join(routine_data['namespaces'] + [ instantiation['routine'] ] )

			out.append(
				f"template {return_type} {qualified_routine_name}<{actual_template_args}, perflibs::linalg::spec::{architecture_name}_architecture_spec>({parameters});")

			interface_includes.add(f'#include "{routine_data["interface_header_include"]}"')


	out = "\n".join(out)
	includes = "\n".join(interface_includes)

	actively_request_macro = "\n".join(set( [
			f'#define PERFLIBS_LINALG_{x["routine"].upper()}_ROUTINE_ACTIVELY_REQUESTED = 1'
				for x in instantiations ] ))

	code = f"""
{actively_request_macro}

#include "linalg/specs.hpp"

{includes}

#include "perflibs_float.hpp"
#include "perflibs_complex.hpp"

using perflibs::bf16, perflibs::r16, perflibs::r32, perflibs::r64, perflibs::c16, perflibs::c32, perflibs::c64;

{out}
"""

	output.write(code)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--instantiation-json-file-path")
	parser.add_argument("--routine-json-dir")
	parser.add_argument("--architecture-name")
	parser.add_argument("--output-file-path")
	args = parser.parse_args()

	if args.output_file_path is not None:
		with open(args.output_file_path, "w") as f:
			generate_instantiation(args.routine_json_dir, args.instantiation_json_file_path, args.architecture_name, f)
	else:
		generate_instantiation(args.routine_json_dir, args.instantiation_json_file_path, args.architecture_name, sys.stdout)

if __name__ == "__main__":
	main()
