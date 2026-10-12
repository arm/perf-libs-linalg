#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


import argparse
from pathlib import Path

import specs

parser = argparse.ArgumentParser()
parser.add_argument("package", type=str, help="package")
parser.add_argument("system", type=str, help="system i.e. n1_c6g")

parser.add_argument(
        "output",
        type=argparse.FileType("w"),
        help="output header")

parser.add_argument("--templates-dir", required=True, type=Path)
parser.add_argument("--defaults-dir", type=Path)
parser.add_argument("--target-dir", type=Path)

args = parser.parse_args()

try:
    selected = specs.list_specs(args.templates_dir, args.defaults_dir, args.target_dir)
except ValueError as error:
    parser.error(str(error))

specs_with_kernels=[]
specs_wout_kernels=[]

# Kernel selectors must be declared before the routines that use them.
for key, path in selected.items():
    try:
        routine = specs.load_spec(path, key)
    except ValueError as error:
        parser.error(str(error))

    header_filename = key + ".hpp"
    has_kernel_spec = "kernel_spec" in routine
    if args.system == "live_targets":
        has_kernel_spec = routine["routine"] == "matmul3_vector_scalar"

    if has_kernel_spec:
        specs_with_kernels.append(header_filename)
    else:
        specs_wout_kernels.append(header_filename)

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
