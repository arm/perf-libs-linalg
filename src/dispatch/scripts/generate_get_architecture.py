# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import json

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--config-json-path")
	parser.add_argument("--output-file-path")

	args = parser.parse_args()

	with open(args.config_json_path) as f:
		j = json.load(f)

	header_guard = f"PERFLIBS_CHOOSER_ARCHITECTURES_{j['build_name'].upper()}_HPP"

	architectures = ", ".join(j["architectures"])

	arch_cases = []
	for arch, arch_data in j["architectures"].items():
		sys_cases=[]

		if arch == j["default_architecture"]:
			sys_cases.append("default:")

		for system in arch_data["systems"]:
			sys_cases.append(f"case machine::system::{system}:")

		sys_cases = "\n".join(sys_cases)

		arch_cases.append(f"{sys_cases}\nreturn architectures::{arch};")

	cases = "\n".join(arch_cases)

	out = f"""
#ifndef {header_guard}
#define {header_guard}

#include "detect/system.hpp"

namespace perflibs::chooser {{

enum class architectures {{
	{architectures}
}};

inline
auto get_architecture(machine::system system = machine::get_system()) {{
	switch(system) {{
		{cases}
	}}
}}

}} //namespace perflibs::chooser

#endif //{header_guard}
"""

	if args.output_file_path is not None:
		with open(args.output_file_path, 'w') as f:
			f.write(out)
	else:
		print(out)

if __name__ == "__main__":
	main()
