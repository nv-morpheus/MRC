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

#include "mrc/core/logging.hpp"

#include <atomic>

namespace {
std::atomic<bool> GLOG_INITIALIZED = false;
}  // namespace

namespace mrc {

bool init_logging(const std::string& logname, LogLevels level, bool log_to_stderr)
{
    bool expected   = false;
    bool needs_init = GLOG_INITIALIZED.compare_exchange_strong(expected, true);
    if (needs_init)
    {
        FLAGS_alsologtostderr = log_to_stderr;
        set_log_level(level);
        google::InitGoogleLogging(logname.c_str());
        google::InstallFailureSignalHandler();
    }

    return needs_init;
}

bool is_initialized()
{
    return GLOG_INITIALIZED.load();
}

LogLevels get_log_level()
{
    return static_cast<LogLevels>(FLAGS_minloglevel);
}

void set_log_level(LogLevels level)
{
    FLAGS_minloglevel = static_cast<int>(level);
}

}  // namespace mrc
