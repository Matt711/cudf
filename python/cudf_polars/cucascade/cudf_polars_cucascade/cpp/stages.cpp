/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stages.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <chrono>
#include <optional>
#include <string>

namespace cudf_polars_cucascade {

namespace {

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

std::unique_ptr<cudf::table> aggregate_by_keys(cudf::table_view keys,
                                               std::vector<cudf::column_view> const& values,
                                               std::vector<int> const& aggs,
                                               std::vector<int> const& agg_types,
                                               ::cuda::stream_ref stream)
{
  CUDF_EXPECTS(aggs.empty() || aggs.size() == values.size(),
               "native stages: one aggregation per group-by value");
  CUDF_EXPECTS(agg_types.empty() || agg_types.size() == values.size(),
               "native stages: one result type per group-by value");
  std::vector<cudf::groupby::aggregation_request> requests;
  for (std::size_t i = 0; i < values.size(); ++i) {
    cudf::groupby::aggregation_request r;
    r.values = values[i];
    switch (aggs.empty() ? 0 : aggs[i]) {
      case 0: r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>()); break;
      case 1:
        r.aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
        break;
      case 2:
        r.aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
        break;
      default: CUDF_FAIL("native stages: unknown group-by aggregation");
    }
    requests.push_back(std::move(r));
  }
  cudf::groupby::groupby gb(keys);
  auto [k, results] = gb.aggregate(requests, stream);
  auto cols         = k->release();
  for (std::size_t i = 0; i < results.size(); ++i) {
    auto result = std::move(results[i].results.front());
    if (!agg_types.empty() && agg_types[i] >= 0 &&
        result->type().id() != static_cast<cudf::type_id>(agg_types[i])) {
      result = cudf::cast(result->view(),
                          cudf::data_type{static_cast<cudf::type_id>(agg_types[i])},
                          stream);
    }
    cols.push_back(std::move(result));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> sum_by_keys(cudf::table_view keys,
                                         std::vector<cudf::column_view> const& values,
                                         ::cuda::stream_ref stream)
{
  return aggregate_by_keys(keys, values, {}, {}, stream);
}

}  // namespace

cudf::table_view select_columns(cudf::table_view table, std::vector<int> const& columns)
{
  std::vector<cudf::column_view> cols;
  cols.reserve(columns.size());
  for (auto c : columns) {
    CUDF_EXPECTS(c >= 0 && c < table.num_columns(),
                 "native stages: column " + std::to_string(c) + " of a table with " +
                   std::to_string(table.num_columns()) + " columns");
    cols.push_back(table.column(c));
  }
  return cudf::table_view(cols);
}

std::unique_ptr<cudf::column> evaluate_expression(std::vector<expr_node> const& nodes,
                                                  cudf::table_view t,
                                                  ::cuda::stream_ref stream)
{
  CUDF_EXPECTS(!nodes.empty(), "native stages: empty expression");
  // Each node's value: a column (owned or a view into `t`) or a scalar.
  std::vector<std::unique_ptr<cudf::column>> owned(nodes.size());
  std::vector<cudf::column_view> views(nodes.size());
  std::vector<std::unique_ptr<cudf::scalar>> scalars(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    auto const& n = nodes[i];
    switch (n.op) {
      default: CUDF_FAIL("native stages: unknown expression node");
      case expr_node::column: views[i] = t.column(n.column_index); break;
      case expr_node::literal:
        scalars[i] = std::make_unique<cudf::numeric_scalar<double>>(n.value, true, stream);
        break;
      case expr_node::str_literal:
        scalars[i] = std::make_unique<cudf::string_scalar>(n.text, true, stream);
        break;
      case expr_node::year:
        owned[i] = cudf::datetime::extract_datetime_component(
          views[n.lhs], cudf::datetime::datetime_component::YEAR, stream);
        break;
      case expr_node::cast:
        owned[i] = cudf::cast(
          views[n.lhs], cudf::data_type{static_cast<cudf::type_id>(n.binop)}, stream);
        break;
      case expr_node::binary: {
        auto const op  = static_cast<cudf::binary_operator>(n.binop);
        auto const out = cudf::data_type{static_cast<cudf::type_id>(n.dtype)};
        auto const* l  = scalars[n.lhs].get();
        auto const* r  = scalars[n.rhs].get();
        CUDF_EXPECTS(!(l && r), "native stages: binary of two literals");
        owned[i] = l   ? cudf::binary_operation(*l, views[n.rhs], op, out, stream)
                   : r ? cudf::binary_operation(views[n.lhs], *r, op, out, stream)
                       : cudf::binary_operation(views[n.lhs], views[n.rhs], op, out, stream);
        break;
      }
      case expr_node::ternary: {
        auto const* a    = scalars[n.rhs].get();
        auto const* b    = scalars[n.third].get();
        auto const& mask = views[n.lhs];
        owned[i] = a && b ? cudf::copy_if_else(*a, *b, mask, stream)
                   : a    ? cudf::copy_if_else(*a, views[n.third], mask, stream)
                   : b    ? cudf::copy_if_else(views[n.rhs], *b, mask, stream)
                          : cudf::copy_if_else(views[n.rhs], views[n.third], mask, stream);
        break;
      }
    }
    if (owned[i]) { views[i] = owned[i]->view(); }
  }
  CUDF_EXPECTS(owned.back() != nullptr, "native stages: an expression must compute a column");
  return std::move(owned.back());
}

build_table make_build_table(cudf::table_view table,
                             std::vector<int> const& keys,
                             ::cuda::stream_ref stream)
{
  return {table,
          std::make_unique<cudf::hash_join>(
            select_columns(table, keys), cudf::null_equality::UNEQUAL, stream)};
}

std::unique_ptr<cudf::table> run_stages(std::vector<stage> const& stages,
                                        cudf::table_view input,
                                        std::vector<build_table> const& builds,
                                        ::cuda::stream_ref stream,
                                        stage_counters* counters)
{
  std::unique_ptr<cudf::table> current;  // owns the current view's data after the first stage
  auto view  = input;
  bool owned = false;  // whether `view` is all of `current`
  for (auto const& s : stages) {
    auto const t0  = now();
    double* timer  = nullptr;
    switch (s.op) {
      default: CUDF_FAIL("native stages: unknown stage");
      case stage::join: {
        auto const& build = builds.at(s.build);
        auto const rows   = view.num_rows();
        auto [probe_idx, build_idx] =
          build.join->inner_join(select_columns(view, s.probe_keys), std::nullopt, stream);
        auto const n   = static_cast<cudf::size_type>(probe_idx->size());
        auto const i32 = cudf::data_type{cudf::type_id::INT32};
        auto const probe_map =
          cudf::column_view(i32, n, static_cast<void const*>(probe_idx->data()), nullptr, 0);
        auto const build_map =
          cudf::column_view(i32, n, static_cast<void const*>(build_idx->data()), nullptr, 0);
        auto probe_out = cudf::gather(select_columns(view, s.keep_probe),
                                      probe_map,
                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                      stream);
        auto build_out = cudf::gather(select_columns(build.table, s.keep_build),
                                      build_map,
                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                      stream);
        auto cols = probe_out->release();
        for (auto& col : build_out->release()) {
          cols.push_back(std::move(col));
        }
        current = std::make_unique<cudf::table>(std::move(cols));
        view    = current->view();
        owned   = true;
        if (counters != nullptr) {
          timer = &counters->join_s;
          counters->joins += 1;
          if (&s == &stages.front()) {
            counters->rows_in += rows;
            counters->rows_out += n;
          }
        }
        break;
      }
      case stage::compute: {
        std::vector<std::unique_ptr<cudf::column>> cols;
        for (auto i = 0; i < view.num_columns(); ++i) {
          cols.push_back(std::make_unique<cudf::column>(view.column(i), stream));
        }
        for (auto const& e : s.exprs) {
          cols.push_back(evaluate_expression(e, view, stream));
        }
        current = std::make_unique<cudf::table>(std::move(cols));
        view    = current->view();
        owned   = true;
        if (counters != nullptr) { timer = &counters->compute_s; }
        break;
      }
      case stage::select: {
        view  = select_columns(view, s.columns);
        owned = false;
        break;
      }
      case stage::groupby: {
        std::vector<cudf::column_view> values;
        for (auto v : s.sums) {
          values.push_back(view.column(v));
        }
        current =
          aggregate_by_keys(select_columns(view, s.keys), values, s.aggs, s.agg_types, stream);
        view    = current->view();
        owned   = true;
        if (counters != nullptr) { timer = &counters->groupby_s; }
        break;
      }
    }
    if (timer != nullptr) { *timer += now() - t0; }
  }
  if (!owned) { return std::make_unique<cudf::table>(view, stream); }
  return current;
}

std::unique_ptr<cudf::table> fold_sums(cudf::table_view partials,
                                       int n_keys,
                                       ::cuda::stream_ref stream)
{
  std::vector<int> keys(static_cast<std::size_t>(n_keys));
  for (int i = 0; i < n_keys; ++i) {
    keys[static_cast<std::size_t>(i)] = i;
  }
  std::vector<cudf::column_view> values;
  for (int v = n_keys; v < partials.num_columns(); ++v) {
    values.push_back(partials.column(v));
  }
  return sum_by_keys(select_columns(partials, keys), values, stream);
}

}  // namespace cudf_polars_cucascade
