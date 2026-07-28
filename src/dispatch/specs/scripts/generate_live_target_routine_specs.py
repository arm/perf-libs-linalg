#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import json

from importlib import import_module
from pathlib import Path

import utils

from json2cpp import CPPWriter, JSONReader, generate_cpp
from json2cpp.ast_nodes import *
from json2cpp.passes import *


def generate_live_target_routine_source(routine, pkg):
    class Reader(JSONReader):
        def __init__(self, prefix="spec->"):
            self.prefix = prefix

        def statement(self, s):
            """For generating get_tuned_routine_specs functions,
            treat each value as an assignment to a spec struct"""
            key, value = s
            return Assign(f"{self.prefix}{key}", self.expr(value), value["type"])

    class Writer(CPPWriter):
        def expr(self, e):
            match e:
                case Kernel("gemv"):
                    return f"get_gemv_kernel(strat_tag, pctx, {self.prefix}max_threads)"
                case Kernel("axpby"):
                    return f"get_spec(axpby_kernel_tag {{ strat_tag }}, pctx)"
                case Kernel(kernel_name):
                    return f"get_{kernel_name}_kernel(strat_tag, pctx)"
                case KernelSpec("gemm", params):
                    tag_params =  f", {params}" if params else ""
                    return f"get_spec(gemm_kernel_tag {{ interleaved, strat_tag{tag_params} }}, pctx)"
                case KernelSpec(kernel_name):
                    return f"get_{kernel_name}_kernel_spec(strat_tag, pctx)"
                case ConvertMatrixObject(matrix):
                    return f"get_spec(convert_object_tag {{ {matrix}_matrix, live_spec->kernel_spec }}, pctx)"
                case _:
                    return super().expr(e)

    assert "tuned_routine_spec" in routine
    spec = routine["tuned_routine_spec"]
    matrix_datatypes = utils.get_matrix_types(routine["datatype"])
    scalar_type = "promote_t<{}, {}, {}>".format(*matrix_datatypes)
    architecture_spec = "ArchitectureSpec"
    problem_context_base = routine["problem_context_base"].format(
        *matrix_datatypes, scalar_type
    )
    problem_context = f"problem_context<{routine['package']}::{problem_context_base}, {architecture_spec}>"

    if routine["strategy_tag"] == "strategy_selection_tag":
        strategy_tag = "strategy_selection_tag"
    else:
        strategy_tag = "strategy_tag<{}::{}>".format(
            routine["package"], routine["strategy_tag"]
        )

    func_name = "load_tuned_routine_spec_from_json"
    cpp_type = utils.to_cpp_typename(routine["datatype"])

    spec_type = f"spec<{strategy_tag}, {problem_context}>"

    file_name = f"{routine['routine']}_{routine['datatype']}"

    read_json_passes = [LiveTargetPass(), RemoveKernelsPass()]
    read_json_spec = generate_cpp(spec, reader=Reader(), passes=read_json_passes)

    extra_initializers = generate_cpp(
        spec,
        reader=Reader("live_spec->"),
        writer=Writer("live_spec->"),
        passes=[FilterKernels()],
    )

    # Initializers that reference `pctx` must be set in get_spec_live.
    # Additionally, initializers that reference `spec.` must be updated
    # to reference `live_spec->` instead.
    extra_initializers += generate_cpp(
        spec,
        reader=Reader("live_spec->"),
        writer=Writer("live_spec->"),
        passes=[
            FilterIfUses("pctx"),
            FindAndReplaceString("spec.", "live_spec->")],
    )

    extra_initializers += generate_cpp(
        spec,
        reader=Reader("live_spec->"),
        writer=Writer("live_spec->"),
        passes=[
            FilterLiveCases(),
            LiftLiveCases("live_spec->"),
        ]
    )

    template_list = [*(routine.get("template_parameters", [])), {"type": "typename", "name": "ArchitectureSpec"}]
    template_args = ", ".join([ x["name"] for x in template_list ])
    template_params = ", ".join([ f"{x['type']} {x['name']}" for x in template_list ])
    template_params = f"template<{template_params}>"

    template = """
namespace live {{
    {template_params}
    inline auto read_tuned_routine_spec_{file_name}(const std::filesystem::path& json_dir, const verbosity_level_t verbosity_level) -> std::optional<{spec_type}> {{
        static std::optional<{spec_type}> spec;
        static bool searched = false;
        if (spec) return spec;
        if (searched) return std::nullopt;
        searched = true;

        const auto json_file_path = json_dir / "{file_name}.json";

        if(std::ifstream json_file_stream {{ json_file_path }}; json_file_stream) {{
            if(const auto json = nlohmann::json::parse(json_file_stream); ! json.is_discarded()) {{
                for(const auto& entry : json) {{
                    if(entry["routine"] == "{routine}" && entry["datatype"] == "{datatype}") {{
                        spec.emplace();
                        {read_json_spec}
                        return spec;
                    }}
                }}
                print(verbosity_level, 3, "{routine}", "{datatype}",
                    "could not find array entry matching routine and datatype");
            }}
            else {{
                print(verbosity_level, 2, "{routine}", "{datatype}", "failed to parse json file, using default parameters");
            }}
        }}
        else {{
            print(verbosity_level, 4, "{routine}", "{datatype}", "could not open json file for reading");
        }}

        print(verbosity_level, 5, "{routine}", "{datatype}", "using default parameters");

        return std::nullopt;
    }}

{template_params}
static auto live_tuned_routine_spec<{strategy_tag}, {problem_context}> = read_tuned_routine_spec_{file_name}<{template_args}>(
        get_live_target_dir(),
        get_verbosity_level());

}} //namespace live

{pkg_source}

{template_params}
inline std::optional<{spec_type}> get_spec_live({strategy_tag} strat_tag, const {problem_context}& pctx) {{
    auto live_spec = live::live_tuned_routine_spec<{strategy_tag}, {problem_context}>;
    if (live_spec) {{
        {extra_initializers}
        return live_spec;
    }}
    // No live spec found
    return std::nullopt;
}}
"""

    pkg_source = pkg.live_target.source(routine)

    return template.format(
        datatype=routine["datatype"],
        cpp_type=cpp_type,
        strategy_tag=strategy_tag,
        problem_context=problem_context,
        architecture_spec=architecture_spec,
        file_name=file_name,
        read_json_spec=read_json_spec.strip(),
        extra_initializers=extra_initializers.strip(),
        spec_type=spec_type,
        pkg_source=pkg_source,
        template_params=template_params,
        template_args=template_args,
        routine=routine["routine"],
    ).strip()


def generate_live_target_routine_header(routine, pkg):
    template = """ #ifndef {header_guard}
#define {header_guard}

#include "{pkg.name}/tuned_routine_specs.hpp"

#include "blas/kernels/axpby_kernels.hpp"

#include "packages/{package}/fwd.hpp"
#include "spec/problem_context.hpp"

#include "live_target.hpp"

#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <optional>

namespace perflibs::linalg::spec {{

{pkg_header}
{live_source}

}} // namespace perflibs::linalg::spec

#endif //{header_guard}
"""

    pkg_header = pkg.live_target.header(routine)

    cpp_type = utils.to_cpp_typename(routine["datatype"])
    problem_type = f"problem_type::{routine['routine']}"
    architecture_spec = "ArchitectureSpec"

    matrix_datatypes = utils.get_matrix_types(routine["datatype"])
    scalar_type = "promote_t<{}, {}, {}>".format(*matrix_datatypes)
    problem_context_base = routine["problem_context_base"].format(
        *matrix_datatypes, scalar_type
    )
    problem_context = f"problem_context<{routine['package']}::{problem_context_base}, {architecture_spec}>"
    routine_spec = f"spec<{cpp_type}, {problem_type}, {architecture_spec}>"

    header_guard = (
        f"PERFLIBS_LINALG_LIVE_TARGET_{routine['routine']}_{routine['datatype']}_HPP".upper()
    )

    live_source = generate_live_target_routine_source(routine, pkg)

    return template.format(
        problem_context=problem_context,
        routine_spec=routine_spec,
        header_guard=header_guard,
        pkg_header=pkg_header,
        pkg=pkg,
        live_source=live_source,
        package=routine["package"],
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, choices=["cpp", "hpp"])
    parser.add_argument("-i", "--input-json-file", required=True)
    parser.add_argument("-p", "--package")

    args = parser.parse_args()

    pkg = import_module(args.package)

    input_data = json.loads(Path(args.input_json_file).read_text())

    for entry in input_data:
        if "tuned_routine_spec" in entry:
            if args.type == "cpp":
                file_content = generate_live_target_routine_source(entry, pkg)
            if args.type == "hpp":
                file_content = generate_live_target_routine_header(entry, pkg)

            print(file_content)


if __name__ == "__main__":
    main()
