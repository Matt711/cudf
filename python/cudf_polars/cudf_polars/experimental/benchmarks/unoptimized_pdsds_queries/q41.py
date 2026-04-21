# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q41 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

import operator as op
from functools import reduce
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import sql_sum

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=41, qualification=run_config.qualification
    )
    manufact = params["manufact"]
    rules = params["rules"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Build rule expressions from the rules parameter.
    # Each rule: (i_manufact == rule["manufact"]) & i_color IN colors & i_units IN units & i_size IN sizes
    # Note: the SQL correlated subquery checks i_manufact = i1.i_manufact, so we
    # collect the set of manufacturers that satisfy any rule, then semi-join.
    rule_exprs = [
        (
            (pl.col("i_category") == rule["category"])
            & pl.col("i_color").is_in(rule["colors"])
            & pl.col("i_units").is_in(rule["units"])
            & pl.col("i_size").is_in(rule["sizes"])
        )
        for rule in rules
    ]
    subquery_filter = reduce(op.or_, rule_exprs)

    # Subquery: FROM item WHERE (i_manufact = i1.i_manufact AND rule_conditions)
    # => collect distinct i_manufact values that satisfy any rule
    qualifying_manufacts = (
        item.filter(subquery_filter).select("i_manufact").unique()
    )

    # Main query: SELECT DISTINCT i_product_name FROM item i1
    # WHERE i_manufact_id BETWEEN manufact AND manufact+40
    #   AND (SELECT Count(*) ... ) > 0
    # The correlated subquery count > 0 is equivalent to EXISTS,
    # which we implement as a semi-join on i_manufact.
    result = (
        item.filter(pl.col("i_manufact_id").is_between(manufact, manufact + 40))
        .join(qualifying_manufacts, on="i_manufact", how="semi")
        .select("i_product_name")
        .unique()
    )

    result = result.sort('i_product_name', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("i_product_name", False)],
        limit=100,
    )
