/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_resources.h"
#include "testing.pb.h"

#include <nvrpc/context.h>

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
// work-around for known iwyu being confused by protobuf headers
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <functional>
// IWYU pragma: no_include <tuple>

namespace nvrpc {
namespace testing {

class PingPongUnaryContext final : public Context<Input, Output, TestResources>
{
    void ExecuteRPC(Input& input, Output& output) final override;
};

class PingPongStreamingContext final : public StreamingContext<Input, Output, TestResources>
{
    void RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream) final override;

    void StreamInitialized(std::shared_ptr<ServerStream>) final override;
    void RequestsFinished(std::shared_ptr<ServerStream>) final override;

    std::size_t m_Counter;
};

class PingPongStreamingEarlyFinishContext final : public StreamingContext<Input, Output, TestResources>
{
    void RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream) final override;
    void RequestsFinished(std::shared_ptr<ServerStream>) final override;
    void StreamInitialized(std::shared_ptr<ServerStream>) final override;

    std::size_t m_Counter;
};

class PingPongStreamingEarlyCancelContext final : public StreamingContext<Input, Output, TestResources>
{
    void RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream) final override;
    void RequestsFinished(std::shared_ptr<ServerStream>) final override;
    void StreamInitialized(std::shared_ptr<ServerStream>) final override;

    std::size_t m_Counter;
};
}  // namespace testing
}  // namespace nvrpc
