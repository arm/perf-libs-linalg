# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

from dataclasses import dataclass
from typing import Optional


class Expression():
    """Top-level expression node"""

class Statement():
    """Top-level statement node"""

@dataclass
class Value(Expression):
    """Base value node. Intended to represent primitive values."""
    value: object
    ty: Optional[str] = None

@dataclass
class Array(Expression):
    values: list[object]
    ty: Optional[str] = None

@dataclass
class Kernel(Expression):
    kernel_name: str

@dataclass
class KernelSpec(Expression):
    kernel_name: str
    params: Optional[str] = None

@dataclass
class ConvertMatrixObject(Expression):
    matrix: str
    tag: Optional[str] = None

class Empty(Expression):
    """The empty node (transformed to an empty string)"""

@dataclass
class Condition(Expression):
    """Condition node, represents a boolean condition."""
    condition: str

@dataclass
class Cases(Expression):
    """A list of cases, transformed into a nested ternary expression."""
    cases: list[(Condition, Expression)]
    default: Expression
    is_live: bool = False

@dataclass
class Assign(Statement):
    """Assignment statement, assigns an expression to a named variable."""
    key: str
    value: Expression
    value_type: str = None

@dataclass
class Decl(Statement):
    """Declaration statement -- declares a variable of given type."""
    key: str
    value_type: str
    init: Optional[Expression] = None

@dataclass
class JumpTable(Statement):
    cases: list[(dict, Expression)]

@dataclass
class Return(Statement):
    """Return statement"""
    value: Expression

@dataclass
class TypeDef(Statement):
    key: str
    definition: str

@dataclass
class IfElse(Statement):
    """Simple if-else branch."""
    condition: Condition
    if_branch: list[Statement]
    else_branch: list[Statement]

@dataclass
class For(Statement):
    """Range-based for loop"""
    declaration: Statement
    initializer: Expression
    body: list[Statement]

@dataclass
class Break(Statement):
    """break statement"""
    pass
