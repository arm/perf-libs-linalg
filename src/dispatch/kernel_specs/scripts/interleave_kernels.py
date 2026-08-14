#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

"""
Generate C++ interleave kernel specs or fallback instantiations from JSON.

Usage:
    python interleave_kernels.py --matmul-kernels matmul3_kernels.json interleave_kernels.json interleave_kernels.hpp
    python interleave_kernels.py --print-fallback-data-types interleave_kernels.json
    python interleave_kernels.py --matmul-kernels matmul3_kernels.json --fallback-instantiations ARCHITECTURE DATA_TYPE interleave_kernels.json fallback_instantiations.cpp
"""
import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

TEMPLATE_PARAMETERS = {
    "Flags": "kernel_inttype Flags",
    "SrcDataType": "typename SrcDataType",
    "DstDataType": "typename DstDataType",
    "ArchitectureSpec": "typename ArchitectureSpec",
}

FALLBACK_INSTANTIATIONS_KEY = "fallback_instantiations"

# Our SVE and SME kernels support power-of-two vector lengths from 128 to 2048 bits.
VECTOR_LENGTH_BITS = (128, 256, 512, 1024, 2048)

DATA_TYPE_BITS = {
    "bf16": 16,
    "r16": 16,
    "r32": 32,
    "r64": 64,
    "c32": 64,
    "c64": 128,
}

InterleaveValue = int | dict[str, int]


def resolve_interleave(
    value: InterleaveValue, data_type: str, vector_length_bits: int
) -> int:
    if isinstance(value, int):
        return value
    return (
        vector_length_bits
        // DATA_TYPE_BITS[data_type]
        * value["vector_length_multiplier"]
    )


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

    def matches_shape(self, data_type: str, requirement: dict[str, object]) -> bool:
        if (
            requirement["datatype"] != data_type
            or self.split_factor != requirement["split_factor"]
        ):
            return False

        # Source matrix orientation is only known at runtime.
        for vector_length_bits in VECTOR_LENGTH_BITS:
            if self.vector_size_bits not in (0, vector_length_bits):
                continue

            cntg_interleave = resolve_interleave(
                requirement["cntg_interleave"], data_type, vector_length_bits
            )
            strd_interleave = resolve_interleave(
                requirement["strd_interleave"], data_type, vector_length_bits
            )
            if (
                self.cntg_interleave == cntg_interleave
                and self.strd_interleave == strd_interleave
            ):
                return True

        return False

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
    entries: tuple[InterleaveKernelSpec, ...]

    @property
    def src_data_type(self) -> str:
        return self.arguments[1]

    @property
    def dst_data_type(self) -> str:
        return self.arguments[2]

    @property
    def template_parameters(self) -> tuple[str, ...]:
        return tuple(
            TEMPLATE_PARAMETERS[argument]
            for argument in self.arguments
            if argument in TEMPLATE_PARAMETERS
        )

    @classmethod
    def from_json(
        cls, key: str, entries: list[dict[str, object]]
    ) -> "InterleaveKernelTable":
        kind, arguments_string = key.split("<", maxsplit=1)
        arguments_string = arguments_string.removesuffix(">")
        arguments = tuple(argument.strip() for argument in arguments_string.split(","))
        return cls(
            kind=kind,
            arguments=arguments,
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


def pruned_tables(
    data: dict[str, object],
    matmul_data: list[dict[str, object]],
) -> tuple[InterleaveKernelTable, ...]:
    """Return the interleave tables reachable by any matmul kernel."""
    tables = parse_tables(data)
    primary_table = next(table for table in tables if table.kind == "primary")
    specializations = {
        table.arguments: table for table in tables if table.kind == "specialization"
    }
    result = [replace(primary_table, entries=())]
    requirements = tuple(
        kernel[key]
        for kernel in matmul_data
        for key in ("a_interleave_spec", "b_interleave_spec")
    )

    for request in data[FALLBACK_INSTANTIATIONS_KEY]:
        flags = flags_expression(request["flags"])

        for data_type in request["data_types"]:
            arguments = (flags, data_type, data_type, "ArchitectureSpec")
            source_table = specializations.get(arguments, primary_table)
            entries = tuple(
                entry
                for entry in source_table.entries
                if any(
                    entry.matches_shape(data_type, requirement)
                    for requirement in requirements
                )
            )
            if entries:
                result.append(
                    replace(
                        source_table,
                        kind="specialization",
                        arguments=arguments,
                        entries=entries,
                    )
                )

    return tuple(result)


def generate_cpp_from_data(
    data: dict[str, object],
    matmul_data: list[dict[str, object]],
) -> str:
    generated_tables = "\n\n".join(
        table.to_cpp() for table in pruned_tables(data, matmul_data)
    )

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
    data: dict[str, object],
    matmul_data: list[dict[str, object]],
    architecture: str,
    data_type: str,
) -> str:
    architecture_spec = f"spec::{architecture}_architecture_spec"
    instantiations = (
        entry.fallback_instantiation(
            tuple(
                architecture_spec if argument == "ArchitectureSpec" else argument
                for argument in table.arguments
            )
        )
        for table in pruned_tables(data, matmul_data)
        if table.src_data_type == data_type
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
    parser.add_argument("--matmul-kernels", type=Path)
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
    if not args.print_fallback_data_types and args.matmul_kernels is None:
        parser.error("--matmul-kernels is required for generation")

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

    with args.matmul_kernels.open(encoding="utf-8") as input_file:
        matmul_data = json.load(input_file)

    if args.fallback_instantiations is None:
        generated_cpp = generate_cpp_from_data(data, matmul_data)
    else:
        generated_cpp = generate_fallback_instantiations(
            data, matmul_data, *args.fallback_instantiations
        )

    args.output.write_text(generated_cpp, encoding="utf-8")


if __name__ == "__main__":
    main()
