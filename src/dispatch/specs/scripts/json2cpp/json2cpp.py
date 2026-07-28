# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

from json2cpp.ast_nodes import *
from json2cpp.passes import *


class JSONReader:
    """
    This class takes in a JSON object and parses it into an AST.
    """

    def cases(self, cases, default, is_live, ty=None):
        """Parse a cases construct."""
        parsed_cases = [
            (Condition(c["condition"]), self.expr(c["value"], ty=ty))
            for c in cases
            if "condition" in c
        ]
        return Cases(parsed_cases, self.expr(default, ty=ty), is_live)

    def jump_table(self, cases, ty=None):
        parsed_cases = [
            (c["condition"], Return(self.expr(c["value"], ty=ty)), c.get("comment"))
            for c in cases
            if "condition" in c
        ]
        return JumpTable(parsed_cases)

    def expr(self, e, ty=None):
        """Top-level expression parser"""
        match e:
            case { "type": "convert" }:
                return ConvertMatrixObject(e["matrix"], e.get("tag")) #None if not found
            case {"cases": _, "default": _}:
                return self.cases(e["cases"], e["default"], e.get("live", False), ty=ty)
            case {"kernel_name": kernel_name, "type": "kernel"}:
                return Kernel(kernel_name)
            case {"kernel_name": kernel_name, "type": "kernel_spec"}:
                return KernelSpec(kernel_name, e.get("param"))
            case {"value": value, "type": value_type}:
                return self.expr(value, value_type)
            case {"default": default, "type": value_type}:
                return self.expr(default, value_type)
            case {"value": value}:
                return self.expr(value, ty=ty)
            case {"default": default}:
                return self.expr(default, ty=ty)
            case list():
                return Array(e, ty=ty)
            case _:
                return Value(e, ty=ty)

    def statement(self, s):
        """Top-level statement parser"""
        key, value = s
        match (key, value):
            case (_, {"type": ty}):
                return Assign(key, self.expr(value, ty=ty), ty)
            case (_, {"jump_table": cases}):
                return self.jump_table(cases)

    def __call__(self, json):
        return [self.statement(stmt) for stmt in json.items()]


class CPPWriter:
    """
    Transform an AST into C++ output.
    """

    def __init__(self, prefix="spec."):
        self.prefix = prefix

    def jump_table(self, cases):
        def condition(c):
            return " | ".join(k + "_" + str(v).lower() for (k, v) in c.items())

        out = "switch(val) {"
        for cond, value, comment in cases:
            out += f"case {condition(cond)}:\n"
            if comment:
                out += f"// {comment}\n"
            out += self.statement(value) + "\n"
        out += "default: PERFLIBS_ASSERT(false); return nullptr;}"
        return out

    def cases(self, cases, default):
        if len(cases) == 0:
            return self.expr(default)
        else:
            condition0, value0 = cases[0]
            out = f"{self.expr(condition0)} ? {self.expr(value0)}"
            for condition, value in cases[1:]:
                out += f" : {self.expr(condition)} ? {self.expr(value)}"
            out += f" : {self.expr(default)}"
            return out

    def expr(self, e):
        """Top level expression writer"""
        match e:
            case Array(values, str(ty)) if "inplace_vector" in ty:
                value_t = ty.removeprefix("inplace_vector<").removesuffix(">").split(",")[0]
                return (
                    f"std::initializer_list<{value_t}>{{{','.join(map(str, values))}}}"
                )
            case Array(values):
                return f"{{{','.join(map(str, values))}}}"
            case Condition(value):
                return value
            case Cases(cases, default, _):
                return self.cases(cases, default)
            case KernelSpec("gemm", params):
                 tag_params =  f", {params}" if params else ""
                 return f"get_spec_system(gemm_kernel_tag {{ interleaved, strat_tag{tag_params} }}, pctx, system)"
            case Kernel("axpby"):
                 return f"get_spec_system(axpby_kernel_tag {{ strat_tag }}, pctx, system)"
            case Kernel("gemv"):
                return f"get_gemv_kernel(strat_tag, pctx, {self.prefix}max_threads)"
            case Kernel(kernel_name):
                return f"get_{kernel_name}_kernel_system(strat_tag, pctx, system)"
            case KernelSpec(kernel_name):
                return f"get_{kernel_name}_kernel_spec_system(strat_tag, pctx, system)"
            case ConvertMatrixObject(matrix, tag):
                kernel_spec_name =  f"{tag}_kernel_spec" if tag else "kernel_spec"
                return f"get_spec_system(convert_object_tag {{ {matrix}_matrix, spec.{kernel_spec_name} }}, pctx, system)"
            case Value(bool(v)):
                return str(v).lower()
            case Value(v):
                return v
            case Empty():
                return ""

    def assignment(self, key, value):
        return f"{key} = {self.expr(value)};"

    def if_else(self, condition, if_branch, else_branch):
        else_clause = f" else {{ {self(else_branch)} }}" if else_branch else ""
        return f"if {self.expr(condition)} {{ {self(if_branch)} }}{else_clause}"

    def statement(self, s):
        """Top level statement writer"""
        match s:
            case list():
                return self(s)
            case Return(Expression() as e):
                return f"return {self.expr(e)};"
            case Assign(lhs, Expression() as rhs):
                return self.assignment(lhs, rhs)
            case JumpTable(cases):
                return self.jump_table(cases)
            case TypeDef(lhs, rhs):
                return f"using {lhs} = {rhs};"
            case IfElse(condition, if_branch, else_branch):
                return self.if_else(condition, if_branch, else_branch)
            case Decl(var_name, value_type, None):
                return f"{value_type} {var_name};"
            case Decl(var_name, value_type, init):
                return f"{value_type} {var_name} = {self.expr(init)};"
            case For(decl, init, body):
                stmt = self.statement(decl).rstrip(";")
                return f"for ({stmt} : {self.expr(init)}) {{{self(body)}}}"
            case Break():
                return "break;"
            case _:
                return ""

    def __call__(self, stmts):
        return "\n".join(
            [self.statement(stmt) for stmt in stmts if not isinstance(stmt, Empty)]
        )


def generate_cpp(
    json,
    reader=JSONReader(),
    writer=CPPWriter(),
    passes=[AddScaleInit(), ExpandMaxThreads(), ReorderMaxThreads(), ReplaceOMPCalls()],
):
    # Parse JSON into AST
    parsed_input = reader(json)

    # Run through transformation passes
    [parsed_input := visitor(parsed_input) for visitor in passes]

    # Convert AST to C++
    return writer(parsed_input)
