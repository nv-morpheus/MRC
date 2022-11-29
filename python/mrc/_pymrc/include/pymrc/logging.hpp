/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

namespace mrc::pymrc {
#pragma GCC visibility push(default)

/**
 * @brief Python's logging lib is a pure-python impl, as such the log levels are not defined
 * in the C API.
 * https://docs.python.org/3.8/library/logging.html#logging-levels
 */
namespace py_log_levels {
constexpr int NOTSET   = 0;
constexpr int DEBUG    = 10;
constexpr int INFO     = 20;
constexpr int WARNING  = 30;
constexpr int ERROR    = 40;
constexpr int CRITICAL = 50;
}  // namespace py_log_levels

/**
 * @brief Initializes MRC's logger, calling this function a second time has
 * no impact. The return value inidicates if the logger was initialized,
 * which will be `true` on the first call, and `false` for all subsequant calls.
 * The `py_level` argument is the Python numeric log level and defaults to INFO.
 */
bool init_logging(const std::string& logname, int py_level = py_log_levels::INFO);

/**
 * @brief Returns the currently configured log level of the MRC logger.
 * Safe to call both prior to and after calling `init_logging`
 */
int get_level();

/**
 * @brief Adjusts the log level of MRC's logger, the `py_level` argument is the Python numeric log level
 */
void set_level(int py_level);

/**
 * @brief Logs a message to MRC's logger (currently GLOG).
 * The `py_level` argument is the Python numeric log level
 *
 * Debug messages are logged at the info level, as GLOG's idea of a debug log is
 * an info log that is excluded from non-debug builds.
 *
 * Critical messages are logged at the error level as logging at GLOG's fatal level
 * terminates the program
 */
void log(const std::string& msg,
         int py_level                = py_log_levels::INFO,
         const std::string& filename = std::string(),
         int line                    = 0);

#pragma GCC visibility pop
}  // namespace mrc::pymrc
