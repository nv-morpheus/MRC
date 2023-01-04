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

#include <glog/logging.h>

#include <string>

namespace mrc {

/**
 * @brief Log levels, currently there is a 1:1 mapping to the glog levels.
 * Provided here to limit our direct dependence on glog severity values.
 */
enum class LogLevels
{
    INFO    = google::INFO,
    WARNING = google::WARNING,
    ERROR   = google::ERROR,
    FATAL   = google::FATAL
};

/**
 * @brief Initializes MRC's logger, calling this function a second time has
 * no impact. The return value inidicates if the logger was initialized,
 * which will be `true` on the first call, and `false` for all subsequant calls.
 */
bool init_logging(const std::string& logname, LogLevels level = LogLevels::INFO, bool log_to_stderr = true);

/**
 * @brief Checks if MRC's logger has been initialized via `init_logging`
 *
 * @return true
 * @return false
 */
bool is_initialized();

/**
 * @brief Returns the currently configured log level of the MRC logger.
 * Safe to call both prior to and after calling `init_logging`
 */
LogLevels get_log_level();

/**
 * @brief Adjusts the log level of MRC's logger.
 * Calling this prior to calling `init_logging` has no impact as `init_logging` will set the log level.
 */
void set_log_level(LogLevels level);

}  // namespace mrc
