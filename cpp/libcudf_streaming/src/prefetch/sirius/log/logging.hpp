// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Logging stub for the cudf_streaming prefetch port from Sirius PR #1661.
// Maps SIRIUS_LOG_* macros to rapids_logger (the RAPIDS-standard logging layer,
// which cudf already provides as a transitive dependency).
//
// rapids_logger uses printf-style formatting; Sirius uses {}-style (std::format).
// We bridge the gap by pre-formatting with std::format and passing the result
// as a pre-formatted string to rapids_logger's single-argument log() overload.

#pragma once

#include <cudf/logger.hpp>
#include <rapids_logger/logger.hpp>

#include <format>
#include <string>

#define SIRIUS_LOG_LEVEL_TRACE    RAPIDS_LOGGER_LOG_LEVEL_TRACE
#define SIRIUS_LOG_LEVEL_DEBUG    RAPIDS_LOGGER_LOG_LEVEL_DEBUG
#define SIRIUS_LOG_LEVEL_INFO     RAPIDS_LOGGER_LOG_LEVEL_INFO
#define SIRIUS_LOG_LEVEL_WARN     RAPIDS_LOGGER_LOG_LEVEL_WARN
#define SIRIUS_LOG_LEVEL_ERROR    RAPIDS_LOGGER_LOG_LEVEL_ERROR
#define SIRIUS_LOG_LEVEL_CRITICAL RAPIDS_LOGGER_LOG_LEVEL_CRITICAL
#define SIRIUS_LOG_LEVEL_OFF      RAPIDS_LOGGER_LOG_LEVEL_OFF

#ifndef SIRIUS_ACTIVE_LOG_LEVEL
#define SIRIUS_ACTIVE_LOG_LEVEL SIRIUS_LOG_LEVEL_WARN
#endif

// Pre-format with std::format ({}-style), then pass plain std::string to rapids_logger.
#define SIRIUS_LOGGER_CALL(level, fmt_str, ...) \
  cudf::default_logger().log(rapids_logger::level_enum::level, \
    std::format(fmt_str __VA_OPT__(,) __VA_ARGS__))

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_TRACE
#define SIRIUS_LOG_TRACE(...) SIRIUS_LOGGER_CALL(trace, __VA_ARGS__)
#else
#define SIRIUS_LOG_TRACE(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_DEBUG
#define SIRIUS_LOG_DEBUG(...) SIRIUS_LOGGER_CALL(debug, __VA_ARGS__)
#else
#define SIRIUS_LOG_DEBUG(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_INFO
#define SIRIUS_LOG_INFO(...) SIRIUS_LOGGER_CALL(info, __VA_ARGS__)
#else
#define SIRIUS_LOG_INFO(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_WARN
#define SIRIUS_LOG_WARN(...) SIRIUS_LOGGER_CALL(warn, __VA_ARGS__)
#else
#define SIRIUS_LOG_WARN(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_ERROR
#define SIRIUS_LOG_ERROR(...) SIRIUS_LOGGER_CALL(error, __VA_ARGS__)
#else
#define SIRIUS_LOG_ERROR(...) ((void)0)
#endif

#if SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_CRITICAL
#define SIRIUS_LOG_FATAL(...) SIRIUS_LOGGER_CALL(critical, __VA_ARGS__)
#else
#define SIRIUS_LOG_FATAL(...) ((void)0)
#endif
