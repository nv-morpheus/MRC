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

#include "srf/codable/api.hpp"
#include "srf/codable/decode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/pubsub/subscription_service.hpp"
#include "srf/runtime/forward.hpp"

namespace srf::pubsub {

template <typename T>
class Subscriber final : public node::SourceChannel<T>, public SubscriptionService
{
  public:
    ~Subscriber() final
    {
        stop();
        await_join();
    }

  private:
    Subscriber()
    {
        m_decoder = [this](std::unique_ptr<codable::IDecodableStorage> encoding) {
            auto obj = codable::Decoder<T>(*encoding).deserialize();
            this->await_write(std::move(obj));
        };
    }

    void attach_service(std::unique_ptr<ISubscriber> service)
    {
        CHECK(service && !m_service);
        m_service = std::move(service);
    }

    ISubscriptionService& service() const final
    {
        CHECK(m_service);
        return *m_service;
    }

    std::function<void(std::unique_ptr<codable::IDecodableStorage>)> m_decoder;
    std::unique_ptr<ISubscriber> m_service;
    friend runtime::IResources;
};

}  // namespace srf::pubsub
