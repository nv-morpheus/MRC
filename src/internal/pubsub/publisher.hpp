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

#include "internal/control_plane/client/subscription_service.hpp"
#include "internal/resources/forward.hpp"

#include "srf/node/source_channel.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/utils/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace srf::internal::pubsub {

class PublisherBackend;

class Publisher final : public srf::pubsub::IPublisher
{
    Publisher(std::string service_name, std::uint64_t tag, resources::PartitionResources& resources);

  public:
    ~Publisher() final = default;

    DELETE_COPYABILITY(Publisher);
    DELETE_MOVEABILITY(Publisher);

    const std::string& service_name() const final;
    const std::uint64_t& tag() const final;
    std::unique_ptr<srf::codable::ICodableStorage> create_storage() final;

    void stop() final
    {
        this->release_channel();
    }

    void kill() final
    {
        this->release_channel();
    }

    bool is_live() const final
    {
        return this->has_channel();
    }

    void await_join() final
    {
        this->release_channel();
    }

  private:
    const std::string m_service_name;
    const std::uint64_t m_tag;
    resources::PartitionResources& m_resources;

    friend PublisherBackend;
};

}  // namespace srf::internal::pubsub
