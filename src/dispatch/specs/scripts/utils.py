# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import itertools
import json
import re

cpp_name = {
    "bf16": "bfloat16_t",
    "r16": "half",
    "r32": "float",
    "r64": "double",
    "c16": "complex_half",
    "c32": "complex_float",
    "c64": "complex_double",
}


def get_matrix_types(name, count=3):
    """Split next-gen datatype suffix into AType, BType, CType"""
    base_regex = "(bf16|r16|r32|r64|c16|c32|c64)"
    if match := re.match(rf"^{base_regex}$", name):
        # Single type defines all types (AType==BType==CType)
        return (name,) * count
    elif match := re.match(rf"^{base_regex}{base_regex}$", name):
        # ATypeBType (e.g. r32r32)
        return match.group(1), match.group(2), match.group(2)
    elif match := re.match(rf"^{base_regex}{base_regex}{base_regex}$", name):
        # ATypeBTypeCType (e.g. r32r32r64)
        return match.group(1), match.group(2), match.group(3)


def to_cpp_typename(typename):
    a_type, b_type, c_type = get_matrix_types(typename)
    return f"{cpp_name[a_type]}, {cpp_name[b_type]}, {cpp_name[c_type]}"


def get_problem_context(typename, problem_type, machine_spec, pctx_type):
    """Get problem context typename based on datatypes & pctx_type"""
    # Get unique types in (AType, BType, CType)
    unique_types = map(cpp_name.get, dict.fromkeys(get_matrix_types(typename)).keys())

    # Cycle through unique types to fill problem_context template arguments
    types = ",".join(itertools.islice(itertools.cycle(unique_types), pctx_type))

    if pctx_type == 1:
        return f"problem_context<{types}, {problem_type}, {machine_spec}>"
    else:
        return f"problem_context<std::tuple<{types}>, {problem_type}, {machine_spec}>"

def read_json(input_path):
    with open(input_path) as infile:
        return json.load(infile)


def write_json(json_input, output_path, **kwargs):
    with open(output_path, "w") as outfile:
        json.dump(json_input, outfile, **kwargs)
