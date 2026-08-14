#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


import argparse
import json

from pathlib import Path

CPP_TYPE_NAMES = {
    "bf16": "bfloat16_t",
    "r16": "half",
    "r32": "float",
    "r64": "double",
    "c16": "complex_half",
    "c32": "complex_float",
    "c64": "complex_double",
}


def cpp_types(entry):
    return tuple(
        CPP_TYPE_NAMES[entry[name]] for name in ("a_type", "b_type", "c_type")
    )


def is_fallback(entry):
    return entry.get("implementation", {}).get("kind") == "fallback"


def fallback_specialization(entry, architecture_spec):
    a_type, b_type, c_type = cpp_types(entry)
    scalar_type = f"promote_t<{a_type}, {b_type}, {c_type}>"
    return (
        f"axpby_fallback<{str(entry.get('conj_x', False)).lower()}, "
        f"{architecture_spec}, {a_type}, {c_type}, {scalar_type}, "
        f"zero_mode::{entry['alpha_zero_mode']}, "
        f"zero_mode::{entry['beta_zero_mode']}>"
    )


def generate_spec_entry(entry, cpp_type):
    kernel = entry["kernel_name"]
    if is_fallback(entry):
        kernel = fallback_specialization(entry, "ArchitectureSpec")

    return f""" std::pair{{
        "{entry["kernel_name"]}"sv,
        spec::axpby_kernel_spec<{cpp_type}> {{
            {kernel},
            value_support::{entry["alpha"]},
            value_support::{entry["beta"]},
            value_support::{entry["incx"]},
            value_support::{entry["incy"]},
            {str(entry["is_sve"]).lower()},
            {str(entry.get("conj_x", False)).lower()},
            zero_mode::{entry["alpha_zero_mode"]},
            zero_mode::{entry["beta_zero_mode"]},
        }}
    }}
    """


def generate_catalogue(data):
    out = """
    #pragma once

    #include "framework/axpby_fallback_kernel_fwd.hpp"
    #include "framework/axpby_kernels.hpp"
    #include "framework/linalg_util.hpp"

    #include "perflibs_float.hpp"
    #include "perflibs_complex.hpp"

    #include <array>
    #include <optional>
    #include <string_view>

    namespace perflibs::linalg {

    namespace spec {
    template<typename AType, typename BType, typename CType>
    struct axpby_kernel_spec {
        axpby_kernel_t<AType, BType, CType>* kernel;
        value_support alpha_support;
        value_support beta_support;
        value_support incx_support;
        value_support incy_support;
        bool is_sve;
        bool conj_x;
        zero_mode alpha_zero_mode;
        zero_mode beta_zero_mode;
    };
    }

    namespace {
    using namespace std::literals::string_view_literals;

    template<typename ArchitectureSpec, typename AType, typename BType, typename CType>
    constexpr std::array<std::pair<std::string_view, spec::axpby_kernel_spec<AType, BType, CType>>, 0> axpby_kernels {{}};

    template<typename ArchitectureSpec, typename AType, typename BType = AType, typename CType = AType>
    std::optional<linalg::spec::axpby_kernel_spec<AType, BType, CType>> try_get_kernel(const std::string_view& name) {
        for (const auto& [kname, spec] : axpby_kernels<ArchitectureSpec, AType, BType, CType>) {
            if (name == kname) {
                return spec;
            }
        }
        return std::nullopt;
    }
    """

    datatypes = set()
    for d in data:
        datatypes.add((d["a_type"], d["b_type"], d["c_type"]))

    for dt_a, dt_b, dt_c in sorted(datatypes):
        cpp_type = ", ".join(
            CPP_TYPE_NAMES[datatype] for datatype in (dt_a, dt_b, dt_c)
        )

        content = ",\n".join(
            generate_spec_entry(x, cpp_type)
            for x in data
            if x["a_type"] == dt_a and x["b_type"] == dt_b and x["c_type"] == dt_c
        )

        out += f"""template<typename ArchitectureSpec>
const auto axpby_kernels<ArchitectureSpec, {cpp_type}> = std::array {{
{content}
}};\n
"""
    out += "} } //namespace perflibs::linalg"

    return out.strip()


def generate_fallback_instantiations(data, architecture):
    architecture_spec = f"spec::{architecture}_architecture_spec"
    instantiations = []

    for entry in data:
        if not is_fallback(entry):
            continue

        specialization = fallback_specialization(entry, architecture_spec)
        instantiations.append(
            f"template decltype({specialization}) {specialization};"
        )

    instantiations = "\n".join(instantiations)

    return f"""
#include "{architecture}/linalg/machine_spec.hpp"
#include "framework/axpby_fallback_kernel.hpp"

namespace perflibs::linalg {{

{instantiations}

}} // namespace perflibs::linalg
""".strip()


def main():
    parser = argparse.ArgumentParser("cpp_kernels_from_json")
    parser.add_argument("--fallback-instantiations", metavar="ARCHITECTURE")
    parser.add_argument("json_filepath", nargs="+")
    args = parser.parse_args()
    data = []

    for path in args.json_filepath:
        data += json.loads(Path(path).read_text())

    if args.fallback_instantiations:
        output = generate_fallback_instantiations(
            data, args.fallback_instantiations
        )
    else:
        output = generate_catalogue(data)

    print(output)


if __name__ == "__main__":
    main()
