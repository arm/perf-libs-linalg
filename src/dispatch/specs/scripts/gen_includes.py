#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


import sys
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("package", type=str, help="package")
parser.add_argument("system", type=str, help="system i.e. n1_c6g")
parser.add_argument("json_specs_dir", type=str, help="json spec directory")

parser.add_argument(
        "output",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Optional output file (defaults to stdout)")

args = parser.parse_args()

specs_with_kernels=[]
specs_wout_kernels=[]

json_specs_dir = Path(args.json_specs_dir)
# live_targets headers are generated from package defaults, not per-system targets.
specs_dir = (
	json_specs_dir.parent / "defaults"
	if args.system == "live_targets"
	else json_specs_dir / args.system
)

for filepath in sorted(specs_dir.glob("*.json")):
	with open(filepath) as f:
		j = json.load(f)

	header_filename = Path(filepath).with_suffix(".hpp").name

	if "kernel_spec" in j[0]:
		specs_with_kernels.append( header_filename )
	else:
		specs_wout_kernels.append( header_filename )

header_guard=f"PERFLIBS_LINALG_{args.system.upper()}_{args.package.upper()}_TUNED_SPECS_HPP"

includes = "\n".join( [ f'#include "{x}"' for x in  specs_with_kernels ] )
includes += "\n\n"
includes += "\n".join( [ f'#include "{x}"' for x in  specs_wout_kernels ] )

args.output.write(f"""
#ifndef {header_guard}
#define {header_guard}

{includes}

#endif //{header_guard}
""")
