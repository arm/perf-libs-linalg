#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

"""
Regenerate C++ interleave_kernel_specs from a JSON file.

Usage:
    python interleave_kernels.py interleave_kernels.json interleave_kernels.hpp

If output_file is omitted, generated C++ is written to stdout.
"""
import json
import re
import sys
import argparse

def sanitize_key_for_template(key):
    if key.startswith("primary<"):
        return ("primary", None)
    m = re.match(r"specialization<(.+)>", key)
    if m:
        return ("specialization", m.group(1).strip())
    return ("specialization", key)

def generate_cpp_from_data(data):
    keys = list(data.keys())
    keys_sorted = []
    for k in keys:
        if k.startswith("primary<"):
            keys_sorted.append(k)
            break
    for k in keys:
        if not k.startswith("primary<"):
            keys_sorted.append(k)

    lines = []

    for key in keys_sorted:
        kind, inner = sanitize_key_for_template(key)
        entries = data[key]
        if kind == "primary":
            header = ("template<kernel_inttype Flags, typename SrcDataType, typename DstDataType, typename ArchitectureSpec>\n"
                      "inline\n"
                      "constexpr auto interleave_kernel_specs = std::array {\n")
        else:
            header = f"template<typename ArchitectureSpec>\ninline\nconstexpr auto interleave_kernel_specs<{inner}> = std::array {{\n"
        lines.append(header)
        for e in entries:
            cntg_interleave = e.get("cntg_interleave", 1)

            cntg_interleave_step = e.get("cntg_interleave_step", 1)
            strd_interleave = e.get("strd_interleave", 0)
            strd_interleave_step = e.get("strd_interleave_step", 1)
            split_factor = e.get("split_factor", 0)
            vector_size_bits = e.get("vector_size_bits", 0)
            matrix_req = e.get("matrix_req", "matrix_requirement::cntg_one")
            kernel = e.get("kernel", "nullptr")
            kstr = kernel.strip() if isinstance(kernel, str) else "nullptr"
            if isinstance(kernel, str) and kstr.startswith("&"):
                kernel_repr = kstr
            else:
                kernel_repr = f"&{kstr}" if kstr != "nullptr" else "nullptr"
            init = f"    interleave_kernel_spec {{ {cntg_interleave}_ki, {strd_interleave}_ki, {cntg_interleave_step}_ki, {strd_interleave_step}_ki, {split_factor}_ki, {vector_size_bits}_ki, {matrix_req}, {kernel_repr} }},\n"
            lines.append(init)
        lines.append("};\n\n")
    return f"""
#ifndef PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP
#define PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP

#include "interleave_kernels_pre.hpp"

namespace perflibs::linalg {{

{"".join(lines)}
}} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP
"""

def main():
    parser = argparse.ArgumentParser(description='Generate C++ interleave_kernel_specs from a JSON description.')
    parser.add_argument('input_json', help='Path to input JSON file')
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help='Output C++ file path (optional). Defaults to stdout.')
    args = parser.parse_args()

    with open(args.input_json, 'r') as fh:
        data = json.load(fh)

    cpp = generate_cpp_from_data(data)

    # args.output is a file-like object (sys.stdout or an open file); write to it.
    args.output.write(cpp)

    # If argparse opened a file for us (not stdout), close it to flush to disk.
    if args.output is not sys.stdout:
        args.output.close()
        print(f"Wrote {args.output.name}", file=sys.stderr)

if __name__ == '__main__':
    main()
