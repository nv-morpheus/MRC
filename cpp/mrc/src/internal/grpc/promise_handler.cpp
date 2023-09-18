/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/grpc/promise_handler.hpp"

#include "mrc/utils/string_utils.hpp"  // for MRC_CONCAT_STR

#include <boost/fiber/future/future.hpp>  // for future
#include <glog/logging.h>                 // for COMPACT_GOOGLE_LOG_INFO

#include <atomic>
#include <sstream>  // for operator<<, basic_ostream

namespace mrc::rpc {

std::atomic_size_t PromiseWrapper::s_id_counter = 0;

PromiseWrapper::PromiseWrapper(const std::string& method, bool in_runtime) : id(++s_id_counter), method(method)
{
    size_t runtime_id = 0;

    // try
    // {
    //     runtime_id = resources::Manager::get_resources().runtime_id();
    // } catch (exceptions::MrcRuntimeError& e)
    // {
    //     // Do nothing for now
    // }

    this->prefix = MRC_CONCAT_STR("Promise[" << id << ", " << this << ", " << runtime_id << "](" << method << "): ");
    VLOG(5) << this->to_string() << "#1 creating promise";
}

void PromiseWrapper::set_value(bool val)
{
    auto tmp_prefix = this->to_string();

    // Acquire the mutex to prevent leaving `get_future` before we exit
    // std::unique_lock lock(m_mutex);

    VLOG(5) << tmp_prefix << "#2 setting promise to " << val;
    this->promise.set_value(val);
    VLOG(5) << tmp_prefix << "#3 setting promise to " << val << "... done";
}

bool PromiseWrapper::get_future()
{
    auto future = this->promise.get_future();

    auto value = future.get();

    VLOG(5) << this->to_string() << "#4 got future with value " << value;

    // Before exiting, we must acquire the mutex to prevent this object being cleaned up
    // std::unique_lock lock(m_mutex);

    return value;
}

std::string PromiseWrapper::to_string() const
{
    return this->prefix;
    // return MRC_CONCAT_STR("Promise[" << id << "](" << method << "): ");
}

}  // namespace mrc::rpc
