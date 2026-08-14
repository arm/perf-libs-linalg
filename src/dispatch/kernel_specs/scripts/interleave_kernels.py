#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

"""
Generate C++ interleave kernel specs or fallback instantiations from JSON.

Usage:
    python interleave_kernels.py interleave_kernels.json interleave_kernels.hpp
    python interleave_kernels.py --print-fallback-data-types interleave_kernels.json
    python interleave_kernels.py --fallback-instantiations ARCHITECTURE DATA_TYPE interleave_kernels.json fallback_instantiations.cpp
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_PARAMETERS = {
    "Flags": "kernel_inttype Flags",
    "SrcDataType": "typename SrcDataType",
    "DstDataType": "typename DstDataType",
    "ArchitectureSpec": "typename ArchitectureSpec",
}

FALLBACK_INSTANTIATIONS_KEY = "fallback_instantiations"


@dataclass
class InterleaveKernelSpec:
    strd_interleave: int
    cntg_interleave: int = 1
    cntg_interleave_step: int = 1
    strd_interleave_step: int = 1
    split_factor: int = 0
    vector_size_bits: int = 0
    matrix_req: str = "matrix_requirement::cntg_one"
    kernel: str = "nullptr"
    implementation: str | None = None

    def implementation_function(self) -> str:
        return {
            ("fallback", "matrix_requirement::cntg_one"): "n_cpp_interleave",
            ("fallback", "matrix_requirement::strd_one"): "t_cpp_interleave",
        }[(self.implementation, self.matrix_req)]

    def is_fallback(self) -> bool:
        return self.implementation == "fallback"

    def to_cpp(self, arguments: tuple[str, ...]) -> str:
        kernel = self.kernel
        if self.implementation is not None:
            kernel = (
                f"{self.implementation_function()}<"
                f"{self.strd_interleave}, {', '.join(arguments)}>"
            )
        kernel = kernel if kernel == "nullptr" else f"&{kernel}"

        values = (
            f"{self.cntg_interleave}_ki",
            f"{self.strd_interleave}_ki",
            f"{self.cntg_interleave_step}_ki",
            f"{self.strd_interleave_step}_ki",
            f"{self.split_factor}_ki",
            f"{self.vector_size_bits}_ki",
            self.matrix_req,
            kernel,
        )
        return f"    interleave_kernel_spec {{ {', '.join(values)} }},"

    def fallback_instantiation(self, arguments: tuple[str, ...]) -> str:
        src_data_type = arguments[1]
        dst_data_type = arguments[2]
        return f"""template void {self.implementation_function()}<{self.strd_interleave}, {', '.join(arguments)}>(
    std::size_t, std::size_t, const {src_data_type}*, std::size_t, std::size_t,
    std::size_t, std::size_t, {dst_data_type}*, std::size_t,
    kernel_inttype, kernel_inttype);"""


@dataclass
class InterleaveKernelTable:
    kind: str
    arguments: tuple[str, ...]
    src_data_type: str
    dst_data_type: str
    template_parameters: tuple[str, ...]
    entries: tuple[InterleaveKernelSpec, ...]

    @classmethod
    def from_json(
        cls, key: str, entries: list[dict[str, object]]
    ) -> "InterleaveKernelTable":
        kind, arguments_string = key.split("<", maxsplit=1)
        arguments_string = arguments_string.removesuffix(">")
        arguments = tuple(argument.strip() for argument in arguments_string.split(","))
        template_parameters = tuple(
            TEMPLATE_PARAMETERS[argument]
            for argument in arguments
            if argument in TEMPLATE_PARAMETERS
        )

        return cls(
            kind=kind,
            arguments=arguments,
            src_data_type=arguments[1],
            dst_data_type=arguments[2],
            template_parameters=template_parameters,
            entries=tuple(InterleaveKernelSpec(**entry) for entry in entries),
        )

    def to_cpp(self) -> str:
        template_declaration = ", ".join(self.template_parameters)
        arguments = ", ".join(self.arguments)
        target = (
            "interleave_kernel_specs"
            if self.kind == "primary"
            else f"interleave_kernel_specs<{arguments}>"
        )
        declaration = (
            f"template<{template_declaration}>\n"
            "inline\n"
            f"constexpr auto {target} = "
        )

        if not self.entries:
            return (
                f"{declaration}std::array<interleave_kernel_spec<"
                f"{self.src_data_type}, {self.dst_data_type}>, 0> {{ }};"
            )

        entries = "\n".join(entry.to_cpp(self.arguments) for entry in self.entries)
        return f"{declaration}std::array {{\n{entries}\n}};"


def parse_tables(data: dict[str, object]) -> list[InterleaveKernelTable]:
    tables = [
        InterleaveKernelTable.from_json(key, entries)
        for key, entries in data.items()
        if key != FALLBACK_INSTANTIATIONS_KEY
    ]
    tables.sort(key=lambda table: table.kind != "primary")
    return tables


def generate_cpp_from_data(data: dict[str, object]) -> str:
    tables = parse_tables(data)

    generated_tables = "\n\n".join(table.to_cpp() for table in tables)

    return f"""#ifndef PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP
#define PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP

#include "kernel_specs/interleave_kernel_spec.hpp"

namespace perflibs::linalg {{

{generated_tables}

template<kernel_inttype Flags, typename ProblemContext, typename System, typename... Types>
PERFLIBS_LINALG_INLINE
auto get_specs(interleave_kernel_specs_tag<Flags, Types...>, const ProblemContext&, System) {{
    return interleave_kernel_specs<Flags, Types..., typename ProblemContext::architecture_spec_type>;
}}

}} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_KERNEL_SPECS_INTERLEAVE_KERNELS_HPP
"""


def flags_expression(flags: list[str]) -> str:
    flags_cpp = " | ".join(f"interleave_flags::{flag}" for flag in flags)
    return f"kernel_inttype({flags_cpp or 'interleave_flags::None'})"


def generate_fallback_instantiations(
    data: dict[str, object], architecture: str, data_type: str
) -> str:
    tables = parse_tables(data)
    primary_table = next(table for table in tables if table.kind == "primary")
    architecture_spec = f"spec::{architecture}_architecture_spec"
    instantiations = []

    for request in data[FALLBACK_INSTANTIATIONS_KEY]:
        if data_type not in request["data_types"]:
            continue

        flags = flags_expression(request["flags"])
        specialization_arguments = (flags, data_type, data_type, "ArchitectureSpec")
        table = next(
            (
                table
                for table in tables
                if table.kind == "specialization"
                and table.arguments == specialization_arguments
            ),
            primary_table,
        )
        arguments = (flags, data_type, data_type, architecture_spec)
        instantiations.extend(
            entry.fallback_instantiation(arguments)
            for entry in table.entries
            if entry.is_fallback()
        )

    generated_instantiations = "\n\n".join(instantiations)
    return f"""#include "{architecture}/linalg/machine_spec.hpp"

#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"

#define PERFLIBS_LINALG_INTERLEAVE_FALLBACK_KERNEL_ACTIVELY_REQUESTED 1
#include "framework/interleave_fallback_kernel.hpp"

namespace perflibs::linalg {{

{generated_instantiations}

}} // namespace perflibs::linalg
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-fallback-data-types", action="store_true")
    parser.add_argument(
        "--fallback-instantiations",
        nargs=2,
        metavar=("ARCHITECTURE", "DATA_TYPE"),
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if not args.print_fallback_data_types and args.output is None:
        parser.error("output is required unless --print-fallback-data-types is used")

    with args.input_json.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    if args.print_fallback_data_types:
        data_types = dict.fromkeys(
            data_type
            for request in data[FALLBACK_INSTANTIATIONS_KEY]
            for data_type in request["data_types"]
        )
        print(*data_types)
        return

    if args.fallback_instantiations is None:
        generated_cpp = generate_cpp_from_data(data)
    else:
        generated_cpp = generate_fallback_instantiations(
            data, *args.fallback_instantiations
        )

    args.output.write_text(generated_cpp, encoding="utf-8")


if __name__ == "__main__":
    main()
