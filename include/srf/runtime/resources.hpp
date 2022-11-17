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

#pragma once

#include "srf/pubsub/api.hpp"
#include "srf/pubsub/publisher.hpp"
#include "srf/pubsub/publisher_policy.hpp"

#include <string>

namespace srf::runtime {

class IResources
{
  public:
    ~IResources() = default;

    template <typename T>
    pubsub::Publisher<T> make_publisher(std::string name, pubsub::PublisherPolicy policy)
    {
        return {create_publisher(name, policy)};
    }

  private:
    virtual std::unique_ptr<pubsub::IPublisher> create_publisher(const std::string& name,
                                                                 const pubsub::PublisherPolicy& policy) = 0;
};

}  // namespace srf::runtime
