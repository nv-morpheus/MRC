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
#include "internal/resources/partition_resources.hpp"
#include "internal/runtime/runtime.hpp"

#include <string>

namespace mrc::internal::pubsub {

/**
 * @brief PubSub is a specialization of the generic SubscriptionService
 *
 * This class defines the set of allowed roles.
 */
class Base : public control_plane::client::SubscriptionService
{
  public:
    Base(std::string name, runtime::Partition& runtime) :
      SubscriptionService(std::move(name), runtime.resources().network()->control_plane()),
      m_runtime(runtime)
    {}
    using SubscriptionService::SubscriptionService;

    static const std::string& role_publisher()
    {
        static std::string name = "publisher";
        return name;
    }

    static const std::string& role_subscriber()
    {
        static std::string name = "subscriber";
        return name;
    }

    const std::set<std::string>& roles() const final
    {
        static std::set<std::string> r = {role_publisher(), role_subscriber()};
        return r;
    }

  protected:
    inline runtime::Partition& runtime()
    {
        return m_runtime;
    }

    inline resources::PartitionResources& resources()
    {
        return m_runtime.resources();
    }

  private:
    runtime::Partition& m_runtime;
};

}  // namespace mrc::internal::pubsub
