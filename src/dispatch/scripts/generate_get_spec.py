# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import json


def make_get_spec(architecture, header_guard, config_json_path, tuned_routine_spec, kernels, kernel_specs):
	with open(config_json_path) as f:
		j = json.load(f)

	def make_switch(function_name):
		cases = []

		for system, info in j["architectures"][architecture]["systems"].items():
			if system == j["architectures"][architecture]["default_system"]:
				cases.append("default:")

			sys_return = info.get("inherit", system)

			cases.append(f"case machine::system::{system}: return {function_name}(strategy_tag, pctx, system_t<machine::system::{sys_return}>{{}});")

		return "\n\t\t".join(cases)


	def make_getter_function(function_name, has_strategy_tag=False, has_constraint=False):
		switch = make_switch(f"{function_name}_system")

		strategy_tag_template_parameter = "typename StrategyTag,"
		strategy_tag_function_parameter = "StrategyTag strategy_tag, "

		problem_context_template_parameters = "typename ProblemContextBase"
		problem_context_function_parameter = f"problem_context<ProblemContextBase, {architecture}_architecture_spec>"

		constraint = "" if not has_constraint else f" requires has_{function_name}_system_{architecture}<StrategyTag, problem_context<ProblemContextBase, {architecture}_architecture_spec>>"

		return f"""
template<{strategy_tag_template_parameter}{problem_context_template_parameters}>{constraint}
auto {function_name}({strategy_tag_function_parameter}const {problem_context_function_parameter}& pctx) {{
	switch(pctx.architecture_spec.system) {{
		{switch}
	}}
}}
"""

	includes = []

	for system, info in j["architectures"][architecture]["systems"].items():
		for package in [ "blas", "factorization", "strategy_selection" ]:
			if "inherit" not in info:
				includes.append(f'#include "{package}/{system}/tuned_specs.hpp"')

	includes = "\n".join(includes)


	functions = []

	for kernel in kernels:
		is_axpby = kernel == "axpby"
		functions.append(make_getter_function(f"get_{kernel}_kernel", not is_axpby, is_axpby))

	for kernel_spec in kernel_specs:
		functions.append(make_getter_function(f"get_{kernel_spec}_kernel_spec"))

	if tuned_routine_spec:
		functions.append(make_getter_function("get_spec", True, True))


	functions = "\n\n".join(functions)

	get_spec_system_calls = []

	for system, info in j["architectures"][architecture]["systems"].items():
		if "inherit" not in info:
			get_spec_system_calls.append(f"get_spec_system(tag, pctx, system_t<machine::system::{system}>{{}});")

	get_spec_system_calls = "\n\t".join(get_spec_system_calls)

	out=f"""
#ifndef {header_guard}
#define {header_guard}

#include "machine_spec.hpp"
#include "spec/problem_context.hpp"
#include "spec/specs_system.hpp"

{includes}


#include "detect/system.hpp"

namespace perflibs::linalg::spec {{
template<typename StrategyTag, typename ProblemContext>
concept has_get_spec_system_{architecture} = requires(StrategyTag tag, const ProblemContext& pctx) {{
	{get_spec_system_calls}
}};

{functions}
}} //namespace perflibs::linalg::spec

#ifdef PL_LINALG_LIVE_TARGET
#include "blas/live_targets/tuned_specs.hpp"
#include "factorization/live_targets/tuned_specs.hpp"
#include "strategy_selection/live_targets/tuned_specs.hpp"
#endif

#endif //{header_guard}
"""
	return out

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--config-json-path")
	parser.add_argument("--output-file-path")
	parser.add_argument("--tuned-routine-spec", default=False, action="store_true")
	parser.add_argument("--kernel", default='')
	parser.add_argument("--kernel-spec", default='')
	parser.add_argument("--architecture")

	args = parser.parse_args()

	guard_tokens=[]
	if args.tuned_routine_spec:
		guard_tokens.append("get_spec".upper())

	kernels = [] if not args.kernel else args.kernel.split(',')
	kernel_specs = [] if not args.kernel_spec else args.kernel_spec.split(',')

	for kernel in kernels:
		guard_tokens.append(f"get_{kernel}".upper())

	for kernel_spec in kernel_specs:
		guard_tokens.append(f"get_{kernel_spec}_spec".upper())

	guard_tokens="_".join(guard_tokens)

	header_guard = f"PERFLIBS_LINALG_{args.architecture.upper()}_{guard_tokens}_HPP"

	out = make_get_spec(args.architecture, header_guard, args.config_json_path, args.tuned_routine_spec, kernels, kernel_specs)

	if args.output_file_path is not None:
		with open(args.output_file_path, 'w') as f:
			f.write(out)
	else:
		print(out)

if __name__ == "__main__":
	main()
