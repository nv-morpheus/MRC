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

#include "mrc/utils/string_utils.hpp"  // // IWYU pragma: keep for MRC_CONCAT_STR

#include <boost/fiber/future/future.hpp>  // for future
#include <glog/logging.h>                 // for COMPACT_GOOGLE_LOG_INFO

#include <atomic>
#include <sstream>  // for operator<<, basic_ostream

namespace mrc::rpc {

std::atomic_size_t PromiseWrapper::s_id_counter = 0;

PromiseWrapper::PromiseWrapper(const std::string& method, bool in_runtime) : id(++s_id_counter), method(method)
{
#if (!defined(NDEBUG))
    this->prefix = MRC_CONCAT_STR("Promise[" << id << ", " << this << "](" << method << "): ");
#endif
    VLOG(20) << this->to_string() << "#1 creating promise";
}

void PromiseWrapper::set_value(bool val)
{
    auto tmp_prefix = this->to_string();

    VLOG(20) << tmp_prefix << "#2 setting promise to " << val;
    this->promise.set_value(val);
    VLOG(20) << tmp_prefix << "#3 setting promise to " << val << "... done";
}

bool PromiseWrapper::get_future()
{
    auto future = this->promise.get_future();

    auto value = future.get();

    VLOG(20) << this->to_string() << "#4 got future with value " << value;

    return value;
}

std::string PromiseWrapper::to_string() const
{
    return this->prefix;
}

}  // namespace mrc::rpc
