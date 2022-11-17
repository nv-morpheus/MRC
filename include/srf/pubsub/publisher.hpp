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

#include "srf/channel/ingress.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/runtime/forward.hpp"

namespace srf::pubsub {

template <typename T>
class Publisher : public channel::Ingress<T>
{
  public:
    inline channel::Status await_write(T&& data) final
    {
        if (m_publisher)
        {
            auto encoded_object = codable::EncodedObject<T>::create(std::move(data), m_publisher->create_storage());
            m_publisher->await_write(std::move(encoded_object));
        }

        return channel::Status::error;
    }

  private:
    Publisher(std::unique_ptr<IPublisher> publisher);
    std::unique_ptr<IPublisher> m_publisher;

    friend runtime::IResources;
};

}  // namespace srf::pubsub
