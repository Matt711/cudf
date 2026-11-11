/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/**
 * @file
 * @brief Per-chunk stages of native pipelines: the work a chain runs on each chunk, shared by
 * every executor of it.
 */

namespace cudf_polars_cucascade {

/**
 * @brief One node of a flattened expression: its operands are earlier nodes of the same list, and
 * the last node is the result.
 */
struct expr_node {
  enum kind : int {
    column      = 0,
    literal     = 1,
    binary      = 2,
    year        = 3,
    cast        = 4,
    str_literal = 5,
    ternary     = 6,
  };
  int op{column};
  int column_index{0};  ///< column: index into the current table
  double value{0.0};    ///< literal: a float64 scalar
  std::string text;     ///< str_literal
  int binop{0};         ///< binary: a cudf::binary_operator; cast: a cudf::type_id
  int dtype{10};        ///< binary: the result's cudf::type_id (FLOAT64 by default)
  int lhs{0};           ///< binary, year, cast: operand node; ternary: the condition
  int rhs{0};           ///< binary: operand node; ternary: the value where true
  int third{0};         ///< ternary: the value where false
};

/**
 * @brief Evaluate a flattened expression (see @ref expr_node) against @p table.
 * @return The last node's column.
 */
std::unique_ptr<cudf::column> evaluate_expression(std::vector<expr_node> const& nodes,
                                                  cudf::table_view table,
                                                  ::cuda::stream_ref stream);

/** @brief @ref evaluate_expression on a raw stream (for bindings). */
inline std::unique_ptr<cudf::column> evaluate_expression_on(std::vector<expr_node> const& nodes,
                                                            cudf::table_view table,
                                                            cudaStream_t stream)
{
  return evaluate_expression(nodes, table, ::cuda::stream_ref{stream});
}

/** @brief One step of a chain, applied to each chunk in order. */
struct stage {
  enum kind : int { join = 0, compute = 1, select = 2, groupby = 3 };
  int op{select};
  // join: inner join against a build table; the output is the kept probe columns, then the kept
  // build columns.
  int build{0};
  std::vector<int> probe_keys;
  std::vector<int> keep_probe;
  std::vector<int> keep_build;
  // compute: one appended column per expression
  std::vector<std::vector<expr_node>> exprs;
  // select
  std::vector<int> columns;
  // groupby (last stage): sum of each value column per key
  std::vector<int> keys;
  std::vector<int> sums;
};

/** @brief A build table and the hash table built on its key columns. */
struct build_table {
  cudf::table_view table;
  std::unique_ptr<cudf::hash_join> join;
};

/** @brief Build the hash table of @p table on @p keys (on @p stream). */
build_table make_build_table(cudf::table_view table,
                             std::vector<int> const& keys,
                             ::cuda::stream_ref stream);

/** @brief Time and row counters of @ref run_stages calls (added to, not reset). */
struct stage_counters {
  double join_s{0};
  double compute_s{0};
  double groupby_s{0};
  double joins{0};
  double rows_in{0};   ///< rows into the first join
  double rows_out{0};  ///< rows out of the first join
};

/** @brief The columns @p columns of @p table, in that order. */
cudf::table_view select_columns(cudf::table_view table, std::vector<int> const& columns);

/**
 * @brief Run @p stages on @p input.
 *
 * Join stages probe @p builds (by the stage's `build` index). The result owns its data: a chain
 * ending in a selection copies the selected columns.
 */
std::unique_ptr<cudf::table> run_stages(std::vector<stage> const& stages,
                                        cudf::table_view input,
                                        std::vector<build_table> const& builds,
                                        ::cuda::stream_ref stream,
                                        stage_counters* counters = nullptr);

/**
 * @brief Fold partial sums: @p partials holds @p n_keys key columns, then sums; the result has
 * one row per distinct key with the sums of its sums.
 */
std::unique_ptr<cudf::table> fold_sums(cudf::table_view partials,
                                       int n_keys,
                                       ::cuda::stream_ref stream);

}  // namespace cudf_polars_cucascade
