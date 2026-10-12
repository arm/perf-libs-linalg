# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import argparse

from pathlib import Path
from typing import TypedDict

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()

functions=[
	"template<> std::string_view get_strategy_name<{pctx}>(const {pctx}& pctx, std::size_t idx);",
	"template<> std::size_t get_n_strategies<{pctx}>(const {pctx}& pctx);",
	"template<> bool compute_index<{pctx}>(const {pctx}& pctx, std::size_t idx);",
]


class ContextSpec(TypedDict):
    operation: str
    pctx: str
    types: list[str | tuple[str, ...]]


problem_contexts: list[ContextSpec] = [
	{
		"operation": "matmul2",
		"pctx":
			"""matmul::matmul2<
				triangular_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "bf16", "r16", "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {1}>>,
				general_matrix<matrix_base<{2}>>, {3}>""",
		"types": [ ("bf16", "bf16", "r32", "r32") ]
	},
	{
		"operation": "matadd3",
		"pctx":
			"""matmul::matadd3<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				symmetric_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				general_matrix<matrix_base<const {0}>>,
				symmetric_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				symmetric_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				hermitian_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				general_matrix<matrix_base<const {0}>>,
				hermitian_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "c32", "c64" ]
	},
	{
		"operation": "matmul3",
		"pctx":
			"""matmul::matmul3<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				hermitian_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "c32", "c64" ]
	},
	{
		"operation": "rank_update_2k",
		"pctx":
			"""matmul::rank_update_2k<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				symmetric_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "rank_update_2k",
		"pctx":
			"""matmul::rank_update_2k<
				general_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<const {0}>>,
				hermitian_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "c32", "c64" ]
	},
	{
		"operation": "solve",
		"pctx":
			"""solve::solve<
				triangular_matrix<matrix_base<const {0}>>,
				general_matrix<matrix_base<{0}>>, {0}>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "lu_factorization",
		"pctx":
			"""factorization::lu_factorization<
				general_matrix<matrix_base<{0}>>, pl_linalg_int_t>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "cholesky_factorization",
		"pctx":
			"""factorization::cholesky_factorization<
				hermitian_matrix<matrix_base<{0}>>,
				pl_linalg_int_t>""",
		"types": [ "c32", "c64" ]
	},
	{
		"operation": "cholesky_factorization",
		"pctx":
			"""factorization::cholesky_factorization<
				symmetric_matrix<matrix_base<{0}>>,
				pl_linalg_int_t>""",
		"types": [ "r32", "r64" ]
	},
	{
		"operation": "qr_factorization",
		"pctx":
			"""factorization::qr_factorization<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "bidiagonalization",
		"pctx":
			"""factorization::bidiagonalization<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "bidiagonalization_block",
		"pctx":
			"""factorization::bidiagonalization_block<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "tridiagonalization",
		"pctx":
			"""factorization::tridiagonalization<
				hermitian_matrix<matrix_base<{0}>>>""",
		"types": [ "c32", "c64" ]
	},
	{
		"operation": "tridiagonalization",
		"pctx":
			"""factorization::tridiagonalization<
				symmetric_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64" ]
	},
	{
		"operation": "apply_q_from_qr",
		"pctx":
			"""factorization::apply_q_from_qr<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "generate_q_from_qr",
		"pctx":
			"""factorization::generate_q_from_qr<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "apply_q_from_lq",
		"pctx":
			"""factorization::apply_q_from_lq<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	},
	{
		"operation": "generate_q_from_lq",
		"pctx":
			"""factorization::generate_q_from_lq<
				general_matrix<matrix_base<{0}>>>""",
		"types": [ "r32", "r64", "c32", "c64" ]
	}
]

contexts_by_operation: dict[str, list[str]] = {}
for spec in problem_contexts:
    for data_types in spec["types"]:
        substitutions = (data_types,) if isinstance(data_types, str) else data_types
        context = spec["pctx"].format(*substitutions)
        contexts_by_operation.setdefault(spec["operation"], []).append(context)

# Both declarations and the tool dispatch catalog use these concrete types.
declaration_str = "\n".join(
    function.format(pctx=context)
    for function in functions
    for contexts in contexts_by_operation.values()
    for context in contexts
)

context_lists = []
for operation, contexts in contexts_by_operation.items():
    context_arguments = ",\n".join(contexts)
    context_lists.append(
        f"using {operation}_contexts = std::type_identity<std::tuple<\n"
        f"{context_arguments}\n>>;"
    )
context_list_str = "\n\n".join(context_lists)

header = f"""
#ifndef LINALG_PROBLEM_CONTEXT_BENCH_SUPPORTED_COMPUTE_HPP
#define LINALG_PROBLEM_CONTEXT_BENCH_SUPPORTED_COMPUTE_HPP

#include "perflibs_complex.hpp"
#include "perflibs_float.hpp"
#include "pl_linalg_int.h"
#include "matrix/matrix.hpp"
#include "packages/factorization/problem_context_bases.hpp"
#include "packages/matmul/problem_context_bases.hpp"
#include "packages/solve/problem_context_bases.hpp"

#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace perflibs::linalg {{

template<typename ProblemContext>
std::string_view get_strategy_name(const ProblemContext& pctx, std::size_t idx);

template<typename ProblemContext>
bool compute_index(const ProblemContext& pctx, std::size_t idx);

template<typename ProblemContext>
std::size_t get_n_strategies(const ProblemContext& pctx);

{declaration_str}

{context_list_str}

}} // namespace perflibs:linalg

#endif //LINALG_PROBLEM_CONTEXT_BENCH_SUPPORTED_COMPUTE_HPP
"""

if args.output is None:
	print(header)
else:
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(header + "\n", encoding="utf-8")
