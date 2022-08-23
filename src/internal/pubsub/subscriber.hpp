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

#include "srf/node/rx_source.hpp"
#include "srf/utils/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace srf::internal::pubsub {

template <typename T>
class SubscriberManager;

template <typename T>
class Subscriber
{
    Subscriber(std::string service_name, std::uint64_t tag) : m_service_name(std::move(service_name)), m_tag(tag) {}

  public:
    ~Subscriber() = default;

    DELETE_COPYABILITY(Subscriber);
    DELETE_MOVEABILITY(Subscriber);

    const std::string& service_name()
    {
        return m_service_name;
    }
    const std::uint64_t& tag()
    {
        return m_tag;
    }

  private:
    const std::string m_service_name;
    const std::uint64_t m_tag;

    friend SubscriberManager<T>;
};

}  // namespace srf::internal::pubsub
