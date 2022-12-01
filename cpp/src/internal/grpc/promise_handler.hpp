/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/fiber/all.hpp>

namespace mrc::internal::rpc {

/**
 * @brief MRC Sink to handle ProgressEvents which correspond to Promise<bool> tags
 */
class PromiseHandler final : public mrc::node::GenericSink<ProgressEvent>
{
    void on_data(ProgressEvent&& event) final
    {
        auto* promise = static_cast<boost::fibers::promise<bool>*>(event.tag);
        promise->set_value(event.ok);
    }
};

}  // namespace mrc::internal::rpc
