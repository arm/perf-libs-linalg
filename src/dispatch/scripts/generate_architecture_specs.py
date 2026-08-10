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

	header_guard = f"PERFLIBS_ARCHITECTURES_{j['build_name'].upper()}_SPECS_HPP"


	includes = [ '#include "spec/specs_system.hpp"' ]

	for a in j["architectures"]:
		includes.append(f'#include "{a}/linalg/machine_spec.hpp"')

	for a in j["architectures"]:
		includes.append(f'#include "{a}/linalg/get_kernel_spec.hpp"')

	includes = "\n".join(includes)

	out = f"""#ifndef {header_guard}
#define {header_guard}


{includes}

#endif //{header_guard}"""

	if args.output_file_path is not None:
		with open(args.output_file_path, 'w') as f:
			f.write(out)
	else:
		print(out)

if __name__ == "__main__":
	main()
