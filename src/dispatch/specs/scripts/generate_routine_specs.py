#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


import collections
import argparse
import string
import json
import re

from json2cpp import JSONReader, generate_cpp
from json2cpp.ast_nodes import *


def generate_tuned_spec(routine, is_live):
    class Reader(JSONReader):
        def statement(self, s):
            """For generating routine spec structs, treat each value
            as a variable declaration"""
            key, value = s

            a_type = "std::remove_cv_t<typename problem_context_type::a_matrix_type::value_type>"
            b_type = "std::remove_cv_t<typename problem_context_type::b_matrix_type::value_type>"
            c_type = "std::remove_cv_t<typename problem_context_type::c_matrix_type::value_type>"
            scalar_type = "typename problem_context_type::scalar_type"

            match value:
                case {"type": "convert", "tag": tag}:
                    return Decl(key, f"{tag}_convert_type<{value['matrix']}_matrix>")
                case {"type": "convert"}:
                    return Decl(key, f"convert_type<{value['matrix']}_matrix>")
                case {"type": "kernel", "kernel_name": "gemv"}:
                    return Decl(key, f"gemv_kernel_spec<{a_type}, {b_type}, {c_type}, {scalar_type}>")
                case {"type": "kernel", "kernel_name": "rot"}:
                    return Decl(key, f"rot_kernel_t<typename problem_context_type::vector_type, typename problem_context_type::scalar_type>*")
                case {"type": "kernel", "kernel_name": "rotg"}:
                    return Decl(key, f"rotg_kernel_t<typename problem_context_type::scalar_type>*")
                case {"type": "kernel", "kernel_name": "rotm"}:
                    return Decl(key, f"rotm_kernel_t<typename problem_context_type::scalar_type>*")
                case {"type": "kernel", "kernel_name": "rotmg"}:
                    return Decl(key, f"rotmg_kernel_t<typename problem_context_type::scalar_type>*")
                case {"type": "kernel", "kernel_name": "swap"}:
                    return Decl(key, f"swap_kernel_t<typename problem_context_type::scalar_type>*")
                case {"type": "kernel", "kernel_name": kernel_name, "num_types": 1}:
                    return Decl(key, f"{kernel_name}_kernel_t<{a_type}>*")
                case {"type": "kernel", "kernel_name": kernel_name, "num_types": 2}:
                    return Decl(key, f"{kernel_name}_kernel_t<{a_type}, {b_type}>*")
                case {"type": "kernel", "kernel_name": kernel_name}:
                    return Decl(key, f"{kernel_name}_kernel_t<{a_type}, {b_type}, {c_type}>*")
                case {"type": "kernel_spec", "kernel_name": "gemm"}:
                    kernel_type = f"interleave_matmul_kernel_spec<{a_type}, {b_type}, {c_type}>"
                    return Decl(key, kernel_type)
                case {"type": "typedef", "value": definition}:
                    return TypeDef(key, definition)
                case {"type": value_type, "value": {"cases": _}} if is_live:
                    return [
                        Decl(f"{key}_cases", f"std::vector<std::pair<kernel_inttype, {value_type}>>"),
                        Decl(f"{key}_default", value_type),
                        Decl(key, value_type),
                    ]
                case {"type": value_type}:
                    return Decl(key, value_type)
                case _:
                    raise RuntimeError("When type expected...")

    assert "tuned_routine_spec" in routine

    spec = routine["tuned_routine_spec"]
    package = routine["package"]
    problem_context_base = f"{package}::{routine['problem_context_base']}"
    problem_context = f"problem_context<{problem_context_base}, ArchitectureSpec>"

    if routine["strategy_tag"] == "strategy_selection_tag":
        strategy_tag = "strategy_selection_tag"
    else:
        strategy_tag = "strategy_tag<{}::{}>".format(
            routine["package"], routine["strategy_tag"]
        )

    # get all of the Type placeholders in the ProblemContext
    problem_context_base_type_placeholders = list(
        collections.OrderedDict.fromkeys(
            [
                f"T{v[1]}"
                for v in string.Formatter().parse(problem_context_base)
                if v[1] is not None
            ]
        )
    )

    template_params = [ f"{x['type']} {x['name']}" for x in routine.get("template_parameters", []) ]
    template_types = ", ".join(
        template_params + [f"typename {x}" for x in problem_context_base_type_placeholders]
    )
    problem_context = problem_context.format(
        *list(problem_context_base_type_placeholders)
    )

    extra_using = []
    matrix_using = ""

    for k, v in spec.items():
          if "type" in v and v["type"] == "kernel_spec":
               tag = f"{v['tag']}_" if "tag" in v else ""
               matrix_using = """
	using a_matrix_type = typename problem_context_type::a_matrix_type;
	using b_matrix_type = typename problem_context_type::b_matrix_type;
	using c_matrix_type = typename problem_context_type::c_matrix_type;

	using a_data_type = std::remove_cv_t<typename a_matrix_type::value_type>;
	using b_data_type = std::remove_cv_t<typename b_matrix_type::value_type>;
	using c_data_type = std::remove_cv_t<typename c_matrix_type::value_type>;"""

               extra_using.append( f"""
	using gemm_{tag}kernel_spec_type = interleave_matmul_kernel_spec<a_data_type, b_data_type, c_data_type>;
	template<auto WhichMatrix>
	using {tag}convert_type = decltype( get_spec_system( std::declval<convert_object_tag<WhichMatrix, gemm_{k}_type>>(), std::declval<problem_context_type>(), system ) );
""" )

    extra_using = "\n".join(extra_using)
    spec = generate_cpp(spec, reader=Reader())

    return f"""
template<{template_types}, typename ArchitectureSpec>
struct spec<{strategy_tag}, {problem_context}> {{
    using strategy_tag_type      = {strategy_tag};
    using problem_context_type   = {problem_context};
    using architecture_spec_type = ArchitectureSpec;

	{matrix_using}

	{extra_using}

    {spec}
}};
    """.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--package", required=True)
    parser.add_argument("-i", "--input-json-file", required=True, nargs="+")
    parser.add_argument("--is-live", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    template = """
#ifndef PERFLIBS_LINALG_{package}_TUNED_ROUTINE_SPECS_HPP
#define PERFLIBS_LINALG_{package}_TUNED_ROUTINE_SPECS_HPP

{stdlib_includes}

#include <framework/waxpby_kernels.hpp>
#include <framework/axpby_kernels.hpp>
#include <framework/find_index_kernels.hpp>
#include <framework/l1_norm_kernels.hpp>
#include <framework/l2_norm_kernels.hpp>
#include <framework/copy_kernels.hpp>
#include <framework/scal_kernels.hpp>
#include <framework/trsv_kernels.hpp>
#include <kernel_specs/matmul3_kernels.hpp>
#include <framework/gemv_kernels.hpp>
#include <framework/trsm_kernels.hpp>
#include <framework/dot_kernels.hpp>
#include <framework/rot_kernels.hpp>
#include <framework/swap_kernels.hpp>
#include <framework/inplace_vector.hpp>

{fwd_includes}

namespace perflibs::linalg::spec {{


template<typename StrategyTag, typename ProblemContext>
struct spec;

{spec_decls}

}} // perflibs::linalg::spec

#endif // PERFLIBS_LINALG_TUNED_ROUTINE_SPECS_HPP
""".lstrip()

    # Iterate over files
    spec_decls = []
    fwd_includes = set()

    for in_file in sorted(args.input_json_file):
        with open(in_file) as f:
            input_data = json.load(f)

        for entry in input_data:
            spec_decls.append(generate_tuned_spec(entry, args.is_live))
            fwd_includes.add(f"#include \"packages/{entry['package']}/fwd.hpp\"")

    fwd_includes = "\n".join(fwd_includes)

    stdlib_includes = "#include <utility>\n#include <vector>" if args.is_live else ""

    print(
        template.format(
            spec_decls="\n\n".join(spec_decls),
            package=args.package.upper(),
            fwd_includes=fwd_includes,
            stdlib_includes=stdlib_includes,
        )
    )


if __name__ == "__main__":
    main()
