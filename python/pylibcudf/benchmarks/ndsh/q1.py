# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

import pylibcudf as plc
import pylibcudf.aggregation as plc_agg
import pylibcudf.sorting as plc_sorting
from pylibcudf import binaryop
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    Literal,
    Operation,
    to_expression,
)
from pylibcudf.groupby import GroupBy, GroupByRequest
from pylibcudf.io import parquet as plc_parquet
from pylibcudf.io.types import SourceInfo, TableWithMetadata
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.types import DataType, NullOrder, Order, TypeId


def read_lineitem(filter_expr=None, path="scale-1/lineitem-float.parquet"):
    source = SourceInfo([path])
    opts = plc_parquet.ParquetReaderOptions.builder(source).build()
    if filter_expr is not None:
        opts.set_filter(filter_expr)
    reader = plc_parquet.ChunkedParquetReader(opts)
    first = reader.read_chunk()
    names = first.column_names(include_children=False)
    concatenated_columns = first.tbl.columns()
    while reader.has_next():
        chunk = reader.read_chunk()
        cols = chunk.tbl.columns()
        for i in range(len(concatenated_columns) - 1, -1, -1):
            concatenated_columns[i] = plc.concatenate.concatenate(
                [concatenated_columns[i], cols.pop()]
            )
    tbl = Table(concatenated_columns)
    return TableWithMetadata(tbl, [(name, []) for name in names])


def q1():
    tbl_w_meta = read_lineitem(
        Operation(
            ASTOperator.LESS_EQUAL,
            ColumnNameReference("l_shipdate"),
            Literal(
                Scalar.from_py(
                    datetime.date(1998, 9, 2),
                    DataType(TypeId.TIMESTAMP_DAYS),
                )
            ),
        )
    )

    names = tbl_w_meta.column_names()
    cols = tbl_w_meta.columns
    name_to_col = {names[i]: cols[i] for i in range(len(names))}

    l_returnflag = name_to_col["l_returnflag"]
    l_linestatus = name_to_col["l_linestatus"]
    l_quantity = name_to_col["l_quantity"]
    l_extendedprice = name_to_col["l_extendedprice"]
    l_discount = name_to_col["l_discount"]
    l_tax = name_to_col["l_tax"]

    disc_price = binaryop.binary_operation(
        l_extendedprice,
        binaryop.binary_operation(
            l_extendedprice,
            l_discount,
            binaryop.BinaryOperator.MUL,
            l_extendedprice.type(),
        ),
        binaryop.BinaryOperator.SUB,
        l_extendedprice.type(),
    )
    charge = binaryop.binary_operation(
        disc_price,
        binaryop.binary_operation(
            disc_price,
            l_tax,
            binaryop.BinaryOperator.MUL,
            disc_price.type(),
        ),
        binaryop.BinaryOperator.ADD,
        disc_price.type(),
    )

    gb = GroupBy(Table([l_returnflag, l_linestatus]))
    keys_out, agg_tables = gb.aggregate(
        [
            GroupByRequest(l_quantity, [plc_agg.sum()]),
            GroupByRequest(l_extendedprice, [plc_agg.sum()]),
            GroupByRequest(disc_price, [plc_agg.sum()]),
            GroupByRequest(charge, [plc_agg.sum()]),
            GroupByRequest(l_quantity, [plc_agg.mean()]),
            GroupByRequest(l_extendedprice, [plc_agg.mean()]),
            GroupByRequest(l_discount, [plc_agg.mean()]),
            GroupByRequest(l_quantity, [plc_agg.count()]),
        ]
    )

    result = plc.Table(
        [
            keys_out.columns()[0],
            keys_out.columns()[1],
            agg_tables[0].columns()[0],
            agg_tables[1].columns()[0],
            agg_tables[2].columns()[0],
            agg_tables[3].columns()[0],
            agg_tables[4].columns()[0],
            agg_tables[5].columns()[0],
            agg_tables[6].columns()[0],
            agg_tables[7].columns()[0],
        ]
    )

    return plc_sorting.sort_by_key(
        result,
        keys_out,
        [Order.ASCENDING, Order.ASCENDING],
        [NullOrder.AFTER, NullOrder.AFTER],
    )


def q1_jit():
    tbl_w_meta = read_lineitem(
        Operation(
            ASTOperator.LESS_EQUAL,
            ColumnNameReference("l_shipdate"),
            Literal(
                Scalar.from_py(
                    datetime.date(1998, 9, 2),
                    DataType(TypeId.TIMESTAMP_DAYS),
                )
            ),
        )
    )

    names = tbl_w_meta.column_names()
    cols = tbl_w_meta.columns
    name_to_col = {names[i]: cols[i] for i in range(len(names))}

    l_returnflag = name_to_col["l_returnflag"]
    l_linestatus = name_to_col["l_linestatus"]
    l_quantity = name_to_col["l_quantity"]
    l_extendedprice = name_to_col["l_extendedprice"]
    l_discount = name_to_col["l_discount"]
    l_tax = name_to_col["l_tax"]

    disc_price = plc.transform.compute_column_jit(
        Table([l_extendedprice, l_discount]),
        to_expression(
            "l_extendedprice * (1 - l_discount)",
            ("l_extendedprice", "l_discount"),
        ),
    )
    charge = plc.transform.compute_column_jit(
        Table([l_extendedprice, l_discount, l_tax]),
        to_expression(
            "l_extendedprice * (1 - l_discount) * (1 + l_tax)",
            ("l_extendedprice", "l_discount", "l_tax"),
        ),
    )

    gb = GroupBy(Table([l_returnflag, l_linestatus]))
    keys_out, agg_tables = gb.aggregate(
        [
            GroupByRequest(l_quantity, [plc_agg.sum()]),
            GroupByRequest(l_extendedprice, [plc_agg.sum()]),
            GroupByRequest(disc_price, [plc_agg.sum()]),
            GroupByRequest(charge, [plc_agg.sum()]),
            GroupByRequest(l_quantity, [plc_agg.mean()]),
            GroupByRequest(l_extendedprice, [plc_agg.mean()]),
            GroupByRequest(l_discount, [plc_agg.mean()]),
            GroupByRequest(l_quantity, [plc_agg.count()]),
        ]
    )

    result = plc.Table(
        [
            keys_out.columns()[0],
            keys_out.columns()[1],
            agg_tables[0].columns()[0],
            agg_tables[1].columns()[0],
            agg_tables[2].columns()[0],
            agg_tables[3].columns()[0],
            agg_tables[4].columns()[0],
            agg_tables[5].columns()[0],
            agg_tables[6].columns()[0],
            agg_tables[7].columns()[0],
        ]
    )

    return plc_sorting.sort_by_key(
        result,
        keys_out,
        [Order.ASCENDING, Order.ASCENDING],
        [NullOrder.AFTER, NullOrder.AFTER],
    )


if __name__ == "__main__":
    import time

    # warm up caches
    print("1st Run:")  # noqa: T201
    start = time.perf_counter()
    out = q1_jit()
    end = time.perf_counter()
    print("time (jit):", end - start)  # noqa: T201

    start = time.perf_counter()
    out = q1()
    end = time.perf_counter()
    print("time:", end - start)  # noqa: T201

    print("2nd Run:")  # noqa: T201
    start = time.perf_counter()
    out = q1_jit()
    end = time.perf_counter()
    print("time (jit):", end - start)  # noqa: T201

    start = time.perf_counter()
    out = q1()
    end = time.perf_counter()
    print("time:", end - start)  # noqa: T201

    # import polars as pl
    # print(pl.from_arrow(q1_jit().to_arrow()))
    # print(pl.from_arrow(q1().to_arrow()))
