#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

"""
Regenerate C++ interleave_kernel_specs from a JSON file.

Usage:
    python interleave_kernels.py interleave_kernels.json interleave_kernels.hpp
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


def generate_cpp_from_data(data: dict[str, list[dict[str, object]]]) -> str:
    tables = [
        InterleaveKernelTable.from_json(key, entries)
        for key, entries in data.items()
    ]
    tables.sort(key=lambda table: table.kind != "primary")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.input_json.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    args.output.write_text(generate_cpp_from_data(data), encoding="utf-8")


if __name__ == "__main__":
    main()
