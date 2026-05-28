# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Factory backing the Rust IR translator at ``polars.polars.cudf``."""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any

import polars as pl
from polars import polars as plrs  # type: ignore[attr-defined]

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.to_ast import insert_colrefs

if TYPE_CHECKING:
    from cudf_polars.utils.config import ConfigOptions


__all__ = ["Callbacks", "translate_ir"]


_BINOP_NAMES: dict[str, plc.binaryop.BinaryOperator] = {
    "Eq": plc.binaryop.BinaryOperator.EQUAL,
    "EqValidity": plc.binaryop.BinaryOperator.NULL_EQUALS,
    "NotEq": plc.binaryop.BinaryOperator.NOT_EQUAL,
    "NotEqValidity": plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
    "Lt": plc.binaryop.BinaryOperator.LESS,
    "LtEq": plc.binaryop.BinaryOperator.LESS_EQUAL,
    "Gt": plc.binaryop.BinaryOperator.GREATER,
    "GtEq": plc.binaryop.BinaryOperator.GREATER_EQUAL,
    "Plus": plc.binaryop.BinaryOperator.ADD,
    "Minus": plc.binaryop.BinaryOperator.SUB,
    "Multiply": plc.binaryop.BinaryOperator.MUL,
    "RustDivide": plc.binaryop.BinaryOperator.DIV,
    "TrueDivide": plc.binaryop.BinaryOperator.TRUE_DIV,
    "FloorDivide": plc.binaryop.BinaryOperator.FLOOR_DIV,
    "Modulus": plc.binaryop.BinaryOperator.PYMOD,
    "And": plc.binaryop.BinaryOperator.BITWISE_AND,
    "Or": plc.binaryop.BinaryOperator.BITWISE_OR,
    "Xor": plc.binaryop.BinaryOperator.BITWISE_XOR,
    "LogicalAnd": plc.binaryop.BinaryOperator.LOGICAL_AND,
    "LogicalOr": plc.binaryop.BinaryOperator.LOGICAL_OR,
}


def _schema_from_polars(pl_schema: pl.Schema) -> dict[str, DataType]:
    return {name: DataType(dtype) for name, dtype in pl_schema.items()}


def _contains_nested_lists(val: Any) -> bool:
    if not val:
        return False
    return any(isinstance(_, list) for _ in val)


class Callbacks:
    """Factory whose methods the Rust translator calls to build IR nodes."""

    def __init__(self, config_options: ConfigOptions):
        self.config_options = config_options
        self.errors: list[Exception] = []

    def empty(self, schema: pl.Schema) -> ir.IR:
        return ir.Empty(_schema_from_polars(schema))

    def scan(  # noqa: PLR0913
        self,
        paths: list[str],
        schema: pl.Schema,
        predicate: expr.NamedExpr | None,
        typ: str,
        reader_options_json: str,
        cloud_options_json: str | None,
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
    ) -> ir.IR:
        return ir.Scan(
            _schema_from_polars(schema),
            typ,
            json.loads(reader_options_json) if reader_options_json else {},
            json.loads(cloud_options_json) if cloud_options_json else None,
            list(paths),
            list(with_columns) if with_columns is not None else None,
            skip_rows,
            n_rows,
            tuple(row_index) if row_index is not None else None,
            include_file_paths,
            predicate,
            self.config_options.parquet_options,
        )

    def filter(
        self,
        input: ir.IR,
        predicate: expr.NamedExpr,
        schema: pl.Schema,
    ) -> ir.IR:
        return ir.Filter(_schema_from_polars(schema), predicate, input)

    def simple_projection(
        self,
        input: ir.IR,
        schema: pl.Schema,
    ) -> ir.IR:
        return ir.Projection(_schema_from_polars(schema), input)

    def select(
        self,
        input: ir.IR,
        exprs: list[expr.NamedExpr],
        schema: pl.Schema,
        should_broadcast: bool,  # noqa: FBT001
    ) -> ir.IR:
        return ir.Select(
            _schema_from_polars(schema), list(exprs), should_broadcast, input
        )

    def join(  # noqa: PLR0913
        self,
        input_left: ir.IR,
        input_right: ir.IR,
        left_on: list[expr.NamedExpr],
        right_on: list[expr.NamedExpr],
        schema: pl.Schema,
        how: str,
        nulls_equal: bool,  # noqa: FBT001
        suffix: str,
        coalesce: bool,  # noqa: FBT001
    ) -> ir.IR:
        options = (how, nulls_equal, None, suffix, coalesce, "none")
        return ir.Join(
            _schema_from_polars(schema),
            list(left_on),
            list(right_on),
            options,
            input_left,
            input_right,
        )

    def conditional_join(  # noqa: PLR0913
        self,
        input_left: ir.IR,
        input_right: ir.IR,
        left_on: list[expr.NamedExpr],
        right_on: list[expr.NamedExpr],
        schema: pl.Schema,
        op1_name: str,
        op2_name: str | None,
        nulls_equal: bool,  # noqa: FBT001
        suffix: str,
        coalesce: bool,  # noqa: FBT001
    ) -> ir.IR:
        bool_dtype = DataType(pl.datatypes.Boolean())
        ops = [op1_name] if op2_name is None else [op1_name, op2_name]
        left_name_to_index = {name: i for i, name in enumerate(input_left.schema)}
        right_name_to_index = {name: i for i, name in enumerate(input_right.schema)}
        predicate = functools.reduce(
            functools.partial(
                expr.BinOp, bool_dtype, plc.binaryop.BinaryOperator.LOGICAL_AND
            ),
            (
                expr.BinOp(
                    bool_dtype,
                    _BINOP_NAMES[op],
                    insert_colrefs(
                        left_ne.value,
                        table_ref=plc.expressions.TableReference.LEFT,
                        name_to_index=left_name_to_index,
                    ),
                    insert_colrefs(
                        right_ne.value,
                        table_ref=plc.expressions.TableReference.RIGHT,
                        name_to_index=right_name_to_index,
                    ),
                )
                for op, left_ne, right_ne in zip(ops, left_on, right_on, strict=True)
            ),
        )
        options = (None, nulls_equal, None, suffix, coalesce, "none")
        return ir.ConditionalJoin(
            _schema_from_polars(schema),
            predicate,
            options,
            input_left,
            input_right,
        )

    def named_expr(self, child: expr.Expr, output_name: str) -> expr.NamedExpr:
        return expr.NamedExpr(output_name, child)

    def col(self, name: str, dtype: Any) -> expr.Expr:
        return expr.Col(DataType(dtype), name)

    def literal(self, value: Any, dtype: Any) -> expr.Expr:
        cudf_dtype = DataType(dtype)
        if isinstance(value, plrs.PySeries):
            return expr.LiteralColumn(cudf_dtype, pl.Series._from_pyseries(value))
        if cudf_dtype.id() == plc.TypeId.LIST:  # pragma: no cover
            if _contains_nested_lists(value):
                return expr.LiteralColumn(
                    cudf_dtype, pl.Series([value], dtype=cudf_dtype.polars_type)
                )
            return expr.LiteralColumn(cudf_dtype, pl.Series(value))
        if cudf_dtype.id() == plc.TypeId.STRUCT:
            raise NotImplementedError("Struct literals are not supported")
        return expr.Literal(cudf_dtype, value)

    def binop(
        self,
        left: expr.Expr,
        op_name: str,
        right: expr.Expr,
        dtype: Any,
    ) -> expr.Expr:
        op = _BINOP_NAMES[op_name]
        cudf_dtype = DataType(dtype)
        # libcudf doesn't divide fixed-point columns; route through float64.
        if op == plc.binaryop.BinaryOperator.TRUE_DIV and (
            plc.traits.is_fixed_point(left.dtype.plc_type)
            or plc.traits.is_fixed_point(right.dtype.plc_type)
        ):
            f64 = DataType(pl.Float64())
            return expr.Cast(
                cudf_dtype,
                True,  # noqa: FBT003
                expr.BinOp(
                    f64,
                    op,
                    expr.Cast(f64, True, left),  # noqa: FBT003
                    expr.Cast(f64, True, right),  # noqa: FBT003
                ),
            )
        # libcudf's fixed-point multiply leaves the result unrounded; round
        # to the larger of the two input scales to match polars.
        if op == plc.binaryop.BinaryOperator.MUL and (
            plc.traits.is_fixed_point(left.dtype.plc_type)
            and plc.traits.is_fixed_point(right.dtype.plc_type)
        ):
            left_scale = -left.dtype.plc_type.scale()
            right_scale = -right.dtype.plc_type.scale()
            out_scale = max(left_scale, right_scale)
            return expr.UnaryFunction(
                DataType(pl.Decimal(38, out_scale)),
                "round",
                (out_scale, "half_to_even"),
                expr.BinOp(
                    DataType(pl.Decimal(38, left_scale + right_scale)),
                    op,
                    left,
                    right,
                ),
            )
        return expr.BinOp(cudf_dtype, op, left, right)

    def cast(
        self,
        child: expr.Expr,
        dtype: Any,
        strict: bool,  # noqa: FBT001
    ) -> expr.Expr:
        cudf_dtype = DataType(dtype)
        # libcudf truncates float-to-decimal; polars rounds half-to-even.
        if plc.traits.is_floating_point(
            child.dtype.plc_type
        ) and plc.traits.is_fixed_point(cudf_dtype.plc_type):
            return expr.Cast(
                cudf_dtype,
                strict,
                expr.UnaryFunction(
                    child.dtype,
                    "round",
                    (-cudf_dtype.plc_type.scale(), "half_to_even"),
                    child,
                ),
            )
        if isinstance(child, expr.Literal):
            return child.astype(cudf_dtype)
        return expr.Cast(cudf_dtype, strict, child)

    def ternary(
        self,
        predicate: expr.Expr,
        truthy: expr.Expr,
        falsy: expr.Expr,
        dtype: Any,
    ) -> expr.Expr:
        return expr.Ternary(DataType(dtype), predicate, truthy, falsy)

    def unsupported_ir(
        self, variant_name: str, reason: str | None, schema: pl.Schema
    ) -> ir.IR:
        error = NotImplementedError(f"{variant_name}: {reason or 'not yet implemented'}")
        self.errors.append(error)
        return ir.ErrorNode(_schema_from_polars(schema), str(error))

    def unsupported_expr(
        self, variant_name: str, reason: str | None, dtype: Any
    ) -> expr.Expr:
        error = NotImplementedError(f"{variant_name}: {reason or 'not yet implemented'}")
        self.errors.append(error)
        cudf_dtype = DataType(dtype) if dtype is not None else DataType(pl.Null())
        return expr.ErrorExpr(cudf_dtype, str(error))


def translate_ir(
    nt: Any,
    config_options: ConfigOptions,
) -> tuple[ir.IR, list[Exception]]:
    """Translate the plan rooted at ``nt`` and return ``(root, errors)``."""
    callbacks = Callbacks(config_options)
    root = plrs.cudf.translate_ir(nt, callbacks)
    return root, callbacks.errors
