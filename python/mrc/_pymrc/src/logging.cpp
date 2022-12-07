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

#include "pymrc/logging.hpp"

#include "mrc/core/logging.hpp"

#include <glog/logging.h>  // needs to be included prior to log_severity

#include <ostream>

namespace {

inline mrc::LogLevels py_level_to_mrc(int py_level)
{
    mrc::LogLevels level = mrc::LogLevels::INFO;
    if (py_level >= 40)
    {
        level = mrc::LogLevels::ERROR;
    }
    else if (py_level >= 30)
    {
        level = mrc::LogLevels::WARNING;
    }

    return level;
}

inline int mrc_to_py_level(mrc::LogLevels level)
{
    switch (level)
    {
    case mrc::LogLevels::FATAL:
        return mrc::pymrc::py_log_levels::CRITICAL;
    case mrc::LogLevels::ERROR:
        return mrc::pymrc::py_log_levels::ERROR;
    case mrc::LogLevels::WARNING:
        return mrc::pymrc::py_log_levels::WARNING;
    default:
        return mrc::pymrc::py_log_levels::INFO;
    }
}

}  // namespace

namespace mrc::pymrc {

bool init_logging(const std::string& logname, int py_level)
{
    bool initialized = mrc::init_logging(logname, py_level_to_mrc(py_level));
    if (!initialized)
    {
        LOG(WARNING) << "MRC logger already initialized";
    }

    return initialized;
}

int get_level()
{
    return mrc_to_py_level(mrc::get_log_level());
}

void set_level(int py_level)
{
    mrc::set_log_level(py_level_to_mrc(py_level));
}

void log(const std::string& msg, int py_level, const std::string& filename, int line)
{
    if (!mrc::is_initialized())
    {
        init_logging("mrc");
        LOG(WARNING) << "Log called prior to calling init_logging, initialized with defaults";
    }

    google::LogMessage(filename.c_str(), line, static_cast<int>(py_level_to_mrc(py_level))).stream() << msg;
}

}  // namespace mrc::pymrc
