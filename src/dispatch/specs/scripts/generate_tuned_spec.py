#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse
import itertools
import json
import re

from importlib import import_module
from pathlib import Path

import utils

from json2cpp import JSONReader, generate_cpp
from json2cpp.ast_nodes import *
from json2cpp.passes import *


def generate_tuned_spec(
    routine, system, untuned, pkg_name, pkg, live_target_spec_name=None
):
    class Reader(JSONReader):
        def statement(self, s):
            """For generating get_specs functions,
            treat each value as an assignment to a spec struct"""
            key, value = s
            match value:
                case {"type": "typedef"}:
                    pass
                case {"type": value_type}:
                    return Assign(f"spec.{key}", self.expr(value), value_type)
                case _:
                    return Assign(f"spec.{key}", self.expr(value), None)

    assert "tuned_routine_spec" in routine
    spec = routine["tuned_routine_spec"]
    is_generic = routine["target"] == "generic"

    header_guard = "PERFLIBS_LINALG_{target}_{routine}_{datatype}_HPP".format(
        target=routine["target"],
        routine=routine["routine"],
        datatype=routine["datatype"],
    ).upper()

    headers = [
        f"packages/{routine['package']}/fwd.hpp",
        "spec/specs_system.hpp",
        "spec/routine_specs.hpp",
        f"{pkg_name}/tuned_routine_specs.hpp",
        "detect/omp.hpp",
    ]
    if not untuned:
        headers.append("spec/config.hpp")
    if "kernel_spec" in routine:
        kernel_name = routine["tuned_routine_spec"]["kernel"]["kernel_name"]
        headers.append(f"{pkg_name}/kernels/{kernel_name}_kernels.hpp")

    includes = "\n".join(f'#include "{header}"' for header in sorted(headers))
    live_target_spec_name = (
        live_target_spec_name or f"{routine['routine']}_{routine['datatype']}"
    )
    live_target_header = f"{pkg_name}/live_targets/{live_target_spec_name}.hpp"
    live_target_guard = f'defined(PL_LINALG_LIVE_TARGET) && __has_include("{live_target_header}")'
    includes += f"""
#if {live_target_guard}
#include "{live_target_header}"
#endif
"""
    routine = {**routine, "_live_target_guard": live_target_guard}

    template = """
#ifndef {header_guard}
#define {header_guard}

{includes}
#include <optional>

namespace perflibs::linalg::spec {{

{pkg_spec}

{live_target_fwd}

{template_params}
PERFLIBS_LINALG_INLINE
static auto {func_name}({strategy_tag} strat_tag, const {problem_context}& pctx, system_t<machine::system::{system}> system) {{
#if {live_target_guard}
    if constexpr(is_live()) {{
       if (auto live_spec = get_spec_live(strat_tag, pctx)) {{
         return *live_spec;
       }}
    }}
#endif
    ::perflibs::linalg::spec::spec<{strategy_tag}, {problem_context}> spec;
    {spec}
    return spec;
}}

}} // namespace perflibs::linalg::spec

#endif // {header_guard}
    """
    matrix_datatypes = utils.get_matrix_types(routine["datatype"])

    arch = system["architecture"]
    arch_spec = f"{arch}_architecture_spec"
    func_name = "get_spec_system"
    problem_type = f"problem_type::{routine['routine']}"
    cpp_type = utils.to_cpp_typename(routine["datatype"])

    template_params = [ f"{x['type']} {x['name']}" for x in routine.get("template_parameters", []) ]
    if is_generic:
        # We need a few alterations if generating for generic_aarch64
        arch_spec = "ArchitectureSpec"
        template_params.append( "typename ArchitectureSpec" )

    template_params.append("typename=int");
    template_params = f"template<{', '.join(template_params)}>"

    scalar_type = "promote_t<{}, {}, {}>".format(*matrix_datatypes)

    problem_context_base = routine["problem_context_base"].format(
        *matrix_datatypes, scalar_type
    )
    problem_context = (
        f"problem_context<{routine['package']}::{problem_context_base}, {arch_spec}>"
    )
    live_problem_context = (
        f"problem_context<{routine['package']}::{problem_context_base}, ArchitectureSpec>"
    )

    if routine["strategy_tag"] == "strategy_selection_tag":
        strategy_tag = "strategy_selection_tag"
    else:
        strategy_tag = "strategy_tag<{}::{}>".format(
            routine["package"], routine["strategy_tag"]
        )

    passes = [
        AddScaleInit(),
        ExpandMaxThreads(),
        ReorderMaxThreads(),
        ReplaceOMPCalls(),
        AddCntgInit(),
        # AddNumberThreadInit is temporarily added for default cholesky configs
        AddAvailThreadInit(),
        AddBlockSizes(
            *map(utils.cpp_name.get, matrix_datatypes),
            arch_spec,
        ),
    ]
    if untuned:
        # Change max_threads assignments to get_max_threads
        passes.insert(0, UntunedPass())

    pkg_spec = pkg.tuned_specs.spec(routine)
    live_template_params = [
        f"{x['type']} {x['name']}" for x in routine.get("template_parameters", [])
    ]
    live_template_params.append("typename ArchitectureSpec")
    live_template_params = f"template<{', '.join(live_template_params)}>"

    live_target_fwd = f"""#if {live_target_guard}
{live_template_params}
inline auto get_spec_live({strategy_tag} strat_tag, const {live_problem_context}& pctx)
    -> std::optional<spec<{strategy_tag}, {live_problem_context}>>;
#endif"""

    return template.format(
        cpp_type=cpp_type,
        problem_type=problem_type,
        arch_spec=arch_spec,
        problem_context=problem_context,
        spec=generate_cpp(spec, reader=Reader(), passes=passes),
        func_name=func_name,
        template_params=template_params,
        header_guard=header_guard,
        includes=includes,
        pkg_spec=pkg_spec,
        live_target_fwd=live_target_fwd,
        system=routine["target"],
        strategy_tag=strategy_tag,
        live_target_guard=live_target_guard,
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-json-file", required=True)
    parser.add_argument("--untuned", action="store_true")
    parser.add_argument("-p", "--package")

    args = parser.parse_args()

    pkg = import_module(args.package)

    with open(args.input_json_file) as f:
        input_data = json.load(f)
    live_target_spec_name = Path(args.input_json_file).with_suffix("").name
    for entry in input_data:
        if "tuned_routine_spec" in entry:
            ROOT_DIR = Path(__file__).parent
            systems = utils.read_json(ROOT_DIR / "systems.json")
            system = systems[entry["target"]]
            spec = generate_tuned_spec(
                entry, system, args.untuned, args.package, pkg, live_target_spec_name
            )
            print(spec)

if __name__ == "__main__":
    main()
