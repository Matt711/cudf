# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
The native expression contract.

An expression is compiled to a flat list of nodes (dicts) that the native pipeline evaluates
in order; each node's operands are earlier nodes, and the last node is the result. Supported:

- column references and float64, int32, int64 and string literals;
- ``+ - * /`` producing float64;
- comparisons and boolean logic producing booleans;
- ``when/then/otherwise`` producing float64;
- ``dt.year`` and casts to numeric types.

Anything else raises :class:`NoMatch`, which leaves the subgraph on the regular path.
"""

from __future__ import annotations

from typing import Any

import pylibcudf as plc

from cudf_polars.dsl import expr as E

__all__ = ["NoMatch", "compile_expr", "compute", "select"]


class NoMatch(Exception):
    """A subgraph or expression outside what native regions support."""


_ARITHMETIC = {
    plc.binaryop.BinaryOperator.ADD,
    plc.binaryop.BinaryOperator.SUB,
    plc.binaryop.BinaryOperator.MUL,
    plc.binaryop.BinaryOperator.TRUE_DIV,
}
_PREDICATES = {
    plc.binaryop.BinaryOperator.EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL,
    plc.binaryop.BinaryOperator.LOGICAL_AND,
    plc.binaryop.BinaryOperator.LOGICAL_OR,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
}
_NUMERIC = {
    plc.TypeId.INT16,
    plc.TypeId.INT32,
    plc.TypeId.INT64,
    plc.TypeId.UINT32,
    plc.TypeId.FLOAT64,
}
_F64 = plc.TypeId.FLOAT64
_BOOL = plc.TypeId.BOOL8


def compile_expr(e: E.Expr, names: list[str], nodes: list[dict[str, Any]]) -> int:
    """
    Append the nodes of ``e`` to ``nodes``.

    Parameters
    ----------
    e
        The expression.
    names
        Names of the columns of the table the expression is evaluated on, in order.
    nodes
        The program so far.

    Returns
    -------
    The index of the node holding ``e``'s value.
    """
    if isinstance(e, E.Col):
        nodes.append({"op": "column", "column": names.index(e.name)})
    elif isinstance(e, E.Literal):
        tid = e.dtype.plc_type.id()
        if tid == plc.TypeId.STRING:
            nodes.append({"op": "str_literal", "text": e.value})
        elif tid in (_F64, plc.TypeId.INT32, plc.TypeId.INT64):
            nodes.append({"op": "literal", "value": float(e.value)})
        else:
            raise NoMatch(f"literal of {e.dtype}")
    elif isinstance(e, E.BinOp) and (e.op in _ARITHMETIC or e.op in _PREDICATES):
        out = e.dtype.plc_type.id()
        if out != (_F64 if e.op in _ARITHMETIC else _BOOL):
            raise NoMatch(f"binary op of {e.dtype}")
        lhs = compile_expr(e.children[0], names, nodes)
        rhs = compile_expr(e.children[1], names, nodes)
        nodes.append(
            {
                "op": "binary",
                "binop": int(e.op),
                "dtype": int(out),
                "lhs": lhs,
                "rhs": rhs,
            }
        )
    elif isinstance(e, E.Ternary):
        if e.dtype.plc_type.id() != _F64:
            raise NoMatch(f"ternary of {e.dtype}")
        cond, then, otherwise = (compile_expr(c, names, nodes) for c in e.children)
        nodes.append({"op": "ternary", "lhs": cond, "rhs": then, "third": otherwise})
    elif isinstance(e, E.TemporalFunction) and e.name == E.TemporalFunction.Name.Year:
        nodes.append({"op": "year", "lhs": compile_expr(e.children[0], names, nodes)})
    elif isinstance(e, E.Cast) and e.dtype.plc_type.id() in _NUMERIC:
        lhs = compile_expr(e.children[0], names, nodes)
        nodes.append({"op": "cast", "binop": int(e.dtype.plc_type.id()), "lhs": lhs})
    else:
        raise NoMatch(f"expression {type(e).__name__}")
    return len(nodes) - 1


def compute(
    named: tuple[E.NamedExpr, ...], names: list[str], stages: list[dict[str, Any]]
) -> list[str]:
    """
    Make a column for each named expression.

    Column references are used as they are; the others are computed by one ``compute``
    stage, appended to the table under temporary names (``names`` is updated).

    Returns
    -------
    The column name holding each expression's value.
    """
    exprs: list[list[dict[str, Any]]] = []
    out: list[str] = []
    appended: list[str] = []
    for ne in named:
        if isinstance(ne.value, E.Col):
            out.append(ne.value.name)
            continue
        nodes: list[dict[str, Any]] = []
        compile_expr(ne.value, names, nodes)
        exprs.append(nodes)
        tmp = f"__native_{len(names) + len(appended)}"
        appended.append(tmp)
        out.append(tmp)
    if exprs:
        stages.append({"op": "compute", "exprs": exprs})
        names.extend(appended)
    return out


def select(
    source: list[str],
    target: list[str],
    names: list[str],
    stages: list[dict[str, Any]],
) -> None:
    """Select the ``source`` columns, in order, and call them ``target``."""
    if source != names:
        stages.append({"op": "select", "columns": [names.index(n) for n in source]})
    names[:] = target
