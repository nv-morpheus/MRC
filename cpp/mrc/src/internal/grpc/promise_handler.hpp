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

#pragma once

#include "internal/grpc/progress_engine.hpp"

#include "mrc/node/generic_sink.hpp"
#include "mrc/node/sink_properties.hpp"  // for SinkProperties, Status

#include <boost/fiber/future/promise.hpp>  // for promise

#include <atomic>   // for atomic_size_t
#include <cstddef>  // for size_t
#include <string>

namespace mrc::rpc {

struct PromiseWrapper
{
    PromiseWrapper(std::string method, bool in_runtime = true);

    ~PromiseWrapper();

    size_t id;
    std::string method;
    std::string prefix;
    boost::fibers::promise<bool> promise;

    void set_value(bool val);

    bool get_future();

    std::string to_string() const;

  private:
    static std::atomic_size_t s_id_counter;
};

/**
 * @brief MRC Sink to handle ProgressEvents which correspond to Promise<bool> tags
 */
class PromiseHandler final : public mrc::node::GenericSinkComponent<ProgressEvent>
{
    mrc::channel::Status on_data(ProgressEvent&& event) final
    {
        auto* promise = static_cast<PromiseWrapper*>(event.tag);

        promise->set_value(event.ok);
        return mrc::channel::Status::success;
        delete promise;
    };

    void on_complete() override
    {
        SinkProperties<ProgressEvent>::release_edge_connection();
    };
};

}  // namespace mrc::rpc
