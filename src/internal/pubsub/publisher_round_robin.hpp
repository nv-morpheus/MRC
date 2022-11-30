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

#include "internal/pubsub/publisher_service.hpp"
#include "internal/runtime/partition.hpp"
#include "internal/ucx/endpoint.hpp"

#include "mrc/runtime/remote_descriptor.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace mrc::internal::pubsub {

class PublisherRoundRobin final : public PublisherService
{
    using PublisherService::PublisherService;

  public:
    ~PublisherRoundRobin() final = default;

  private:
    // update local m_next iterator
    void on_update() final;

    // apply the round robin policy
    void apply_policy(mrc::runtime::RemoteDescriptor&& rd) final;

    std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>::const_iterator m_next;

    friend runtime::Partition;
};

}  // namespace mrc::internal::pubsub
