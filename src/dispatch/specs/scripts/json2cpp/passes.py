# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

from dataclasses import dataclass
from json2cpp.ast_nodes import *


class Visitor:
    """Abstract Visitor class that traverses a given AST.

       Subclasses of this class can override the statement &
       expr functions and transform the tree.
    """
    def statement(self, s):
        match s:
            case Assign(key, value, value_type):
                return Assign(key, self(value), value_type)
            case IfElse(condition, if_body, else_body):
                return IfElse(condition, self(if_body), self(else_body))
            case _:
                return s

    def expr(self, e):
        match e:
            case Cases(cases, default, is_live):
                return Cases([(self(c), self(v)) for (c, v) in cases], self(default), is_live)
            case _:
                return e

    def visit(self, node):
        match node:
            case Statement():
                return self.statement(node)
            case Expression():
                return self.expr(node)
            case _:
                return node

    def __call__(self, node):
        if isinstance(node, list):
            return [self(n) for n in node]
        return self.visit(node)

class ExpandMaxThreads(Visitor):
    """Surround a max_threads assignment with an is_mp check.

       Places the assignment inside an IfElse node, assigning
       1 thread if constexpr(omp::is_mp) evaluates to false.
    """

    def statement(self, node):
        match node:
            case Assign(key, Value(1), _) if "max_threads" in key:
                return super().statement(node)
            case Assign(key, value, _) if "max_threads" in key:
                return IfElse(
                    Condition("constexpr(omp::is_mp)"),
                    [IfElse(
                        Condition("(thread_throttle())"),
                        [node],
                        [Assign(key, Value("MAX"))]
                    )] if value != Value("MAX") else [node],
                    [Assign(key, Value(1), "kernel_inttype")],
                )
            case _:
                return super().statement(node)


class ReplaceOMPCalls(Visitor):
    """ Replace max_threads values with appropriate omp:: calls.

        Integral values above 1 are placed inside an
        omp::bounded_get_max_threads() call. "MAX" is
        replaced with omp::get_max_threads().
    """
    class OMPExpressionVisitor(Visitor):
        def expr(self, e):
            match e:
                case Value(1):
                    return e
                case Value("MAX"):
                    return Value("omp::get_max_threads()")
                case Value(v):
                    return Value(f"omp::bounded_get_max_threads({v})")
                case _:
                    return super().expr(e)

    def statement(self, s):
        match s:
            case Assign(key, value, value_t) if "max_threads" in key:
                return Assign(key, self.OMPExpressionVisitor()(value), value_t)
            case _:
                return super().statement(s)


class CheckOMPInConditions(Visitor):
    """Check if omp::get_max_threads() is used in conditions."""

    def __init__(self):
        self.found_omp_usage = False

    def expr(self, e):
        match e:
            case Condition(condition) if "omp::get_max_threads()" in condition:
                self.found_omp_usage = True
                return e
            case _:
                return super().expr(e)

class ReplaceOMPInConditions(Visitor):
    """Replace omp::get_max_threads() with spec.max_threads in conditions."""

    def expr(self, e):
        match e:
            case Condition(condition):
                new_condition = condition.replace(
                    "omp::get_max_threads()", "spec.max_threads")
                return Condition(new_condition)
            case _:
                return super().expr(e)
class ReorderMaxThreads(Visitor):
    """Reorder statements so max_threads is assigned before other assignments
       that use omp::get_max_threads() in conditions. Replace omp::get_max_threads()
       with spec.max_threads in those conditions.
    """

    def __call__(self, stmts):
        # Separate statements into categories
        scale_init_stmts = []
        max_threads_stmts = []
        dependent_stmts = []
        other_stmts = []

        for stmt in stmts:
            if self._is_scale_initialization(stmt):
                scale_init_stmts.append(stmt)
            elif self._is_max_threads_assignment(stmt):
                max_threads_stmts.append(stmt)
            elif self._uses_omp_get_max_threads_in_condition(stmt):
                # Replace omp::get_max_threads() with spec.max_threads in this statement
                replaced_stmt = ReplaceOMPInConditions()(stmt)
                dependent_stmts.append(replaced_stmt)
            else:
                other_stmts.append(stmt)

        # Reorder: scale first, then max_threads, then dependent statements, then others
        return scale_init_stmts + max_threads_stmts + dependent_stmts + other_stmts

    def _is_scale_initialization(self, stmt):
        """Check if this statement initializes scale"""
        match stmt:
            case Assign(key, _, _) if "scale" in key and "const auto" in key:
                return True
            case _:
                return False

    def _is_max_threads_assignment(self, stmt):
        """Check if this statement assigns to max_threads"""
        match stmt:
            case Assign(key, _, _) if "max_threads" in key:
                return True
            case IfElse(_, if_body, else_body):
                # Check nested statements in if-else (for ExpandMaxThreads output)
                for s in if_body + else_body:
                    if self._is_max_threads_assignment(s):
                        return True
                return False
            case _:
                return False

    def _uses_omp_get_max_threads_in_condition(self, stmt):
        """Check if this statement uses omp::get_max_threads() in any condition"""
        checker = CheckOMPInConditions()
        checker(stmt)
        return checker.found_omp_usage

class FilterKernels(Visitor):
    def statement(self, s):
        match s:
            case Assign(key, value, value_type) if value_type in ["kernel", "kernel_spec", "convert"]:
                return super().statement(s)
            case _:
                return []

class FilterLiveCases(Visitor):
    def statement(self, s):
        match s:
            case Assign(_, Cases(_, _, is_live=True)):
                return super().statement(s)
            case _:
                return []

@dataclass
class FilterIfUses(Visitor):
    string: str

    def statement(self, s):
        match s:
            case Assign(_, rhs, _):
                is_used = IsUsed(self.string)
                is_used(rhs)
                if is_used:
                    return super().statement(s)
                else:
                    return []
            case _:
                return []

class RemoveKernelsPass(Visitor):
    def statement(self, s):
        match s:
            case Assign(key, value, value_type) if value_type in ["kernel", "kernel_spec", "convert"]:
                return []
            case _:
                return super().statement(s)

@dataclass
class IsUsed(Visitor):
    """Determine whether variable `name` is used."""

    name: str
    yes: bool = False

    def __bool__(self):
        return self.yes

    def expr(self, e):
        match e:
            case Condition(x) if self.name in x:
                self.yes = True
                return e
            case _:
                return super().expr(e)


class AddScaleInit(Visitor):
    """Initialize scale variable once at the top if any cases use it."""

    def __call__(self, stmts):
        # First pass: check if any statement uses scale
        scale_needed = False
        for stmt in stmts:
            scale_is_used = IsUsed("scale")
            scale_is_used(stmt)
            if scale_is_used:
                scale_needed = True
                break

        # If scale is needed, add declaration at the top
        if scale_needed:
             scale_decl = Assign("const auto scale", Value("pctx_scale(pctx)"), "auto")
             return [scale_decl] + stmts
        else:
             return stmts

class AddCntgInit(Visitor):
    """Initialize cntg_size variable if we are using cases."""

    def statement(self, stmt):
        match stmt:
            case Assign(key, Cases(_), value_t) if "cntg_block_size" in key:
                return [
                    Assign("const auto cntg_size", Value("pctx.cntg"), value_t),
                    stmt,
                ]
            case _:
                return super().statement(stmt)

# AddNumberThreadInit is temporarily added for default cholesky configs
class AddAvailThreadInit(Visitor):
    """Initialize available_threads variables if we are using cases."""

    def statement(self, stmt):
        match stmt:
            case Assign(key, Cases(_), value_t) if "cholesky_tile_size" in key:
                # Construct a string representing return value
                return_value = "pctx.cntg <= 120 ? 1 : omp::get_max_threads()"

                # Use this string in the assignment
                conditional_assignment = Assign(
                    "const auto available_threads",
                    Value(return_value),
                    value_t
                )
                return [
                    conditional_assignment,
                    stmt,
                ]
            case _:
                return super().statement(stmt)


@dataclass
class AddBlockSizes(Visitor):
    a_type: str
    b_type: str
    c_type: str
    machine_spec_type: str

    def statement(self, s):
        match s:
            case Assign(key, Value("l1_block"), value_t):
                return Assign(key, Value(f"0.9 * l1_data_cache_size_elements<{self.b_type}, {self.machine_spec_type}>"), value_t)
            case _:
                return super().statement(s)

@dataclass
class LiftLiveCases(Visitor):
    prefix: str

    def statement(self, s):
        match s:
            case Assign(key, Cases(_, _, is_live=True)):
                sanitized_key   = key.removeprefix(self.prefix)

                # assume these names are unique enough to avoid duplicate declaration errors
                scale_var_name   = f"{sanitized_key}_scale"
                limit_var_name   = f"{sanitized_key}_limit"
                value_var_name   = f"{sanitized_key}_value"
                cases_var_name   = f"{key}_cases"
                default_var_name = f"{key}_default"

                return [
                    Decl(scale_var_name , f"const auto", Value("pctx_scale(pctx)")),
                    Assign(key, Value(default_var_name)),
                    For(
                        Decl(f"[{limit_var_name}, {value_var_name}]", "const auto&"),
                        Value(cases_var_name),
                        [
                            IfElse(Condition(f"({scale_var_name} <= {limit_var_name})"),
                                [
                                    Assign(key, Value(value_var_name)),
                                    Break(),
                                ],
                                [],
                            ),
                        ]
                    ),
                ]
            case _:
                return super().statement(s)

@dataclass
class KernelIndexer(Visitor):
    table: str

    def statement(self, s):
        match s:
            case JumpTable(cases):
                cases_new = []
                for cond, ret, comment in cases:
                    assert isinstance(ret, Return)
                    v = ret.value.value
                    v = f"{self.table}[{v}].second.kernel"
                    cases_new.append((cond, Return(Value(v)), comment))
                return JumpTable(cases_new)
            case _:
                return super().statement(s)


class UntunedPass(Visitor):
    """ Replace max_threads assignment values with omp::get_max_threads
        (i.e. disable thread throttling)

        This pass should be used prior to a ReplaceOMPCalls pass so that
        MAX is correctly expanded.
    """
    def statement(self, s):
        match s:
            case Assign(key, value, value_type) if "max_threads" in key:
                return Assign(key, Value("MAX"), value_type)
            case _:
                return super().statement(s)

class LiveTargetPass(Visitor):
    def __init__(self, prefix="spec->"):
        self.prefix = prefix

    def statement(self, s):
        match s:
            case Assign(key, Cases(_, _, is_live=True), value_type):
                name = key.removeprefix(self.prefix)

                return [
                    Assign(
                        f"{key}_cases",
                        Value(f"parse_live_cases<{value_type}>( entry[\"tuned_routine_spec\"][\"{name}\"][\"cases\"].get<nlohmann::json>() )"),
                        value_type,
                    ),
                    Assign(
                        f"{key}_default",
                        Value(f"entry[\"tuned_routine_spec\"][\"{name}\"][\"default\"].get<{value_type}>()"),
                        value_type,
                    ),
                ]
            case Assign(key, _, value_type):
                return Assign(key, Value(f"entry[\"tuned_routine_spec\"][\"{key.removeprefix(self.prefix)}\"].get<{value_type}>()"), value_type)
            case _:
                return super().statement(s)


class L3Blocks(Visitor):
    """Insert call to get_block_sizes(kspec, pctx) for L3 routines

    Initialize block sizes if we encounter a GEMM kernel spec. This
    is expected to be used when assigning block size settings later
    in the function.
    """

    def statement(self, s):
        match s:
            case Assign(key, KernelSpec("gemm") as kspec, "kernel_spec"):
                return [
                    Assign("const auto kspec", kspec),
                    Assign("const auto bsizes", Value("get_block_sizes(kspec, pctx)")),
                    Assign(key, Value("kspec")),
                ]
            case _:
                return super().statement(s)

@dataclass
class FindAndReplaceString(Visitor):
    old: str
    new: str

    def expr(self, e):
        match e:
            case Kernel(kernel_name) if self.old in kernel_name:
                return Kernel(kernel_name.replace(self.old, self.new))
            case KernelSpec(kernel_name) if self.old in kernel_name:
                return KernelSpec(kernel_name.replace(self.old, self.new))
            case Condition(condition) if self.old in condition:
                return Condition(condition.replace(self.old, self.new))
            case _:
                return super().expr(e)
