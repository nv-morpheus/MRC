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
#include "srf/codable/encoded_object.hpp"
#include "srf/control_plane/api.hpp"
#include "srf/node/sink_channel.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/runtime/remote_descriptor.hpp"

#include <string>

namespace srf::pubsub {

class IPublisher : public virtual control_plane::ISubscriptionService
{
  public:
    ~IPublisher() override = default;

    virtual std::unique_ptr<codable::ICodableStorage> create_storage() = 0;

    virtual channel::Status publish(std::unique_ptr<codable::EncodedStorage> encoded_object) = 0;
    virtual channel::Status publish(runtime::RemoteDescriptor&& remote_descriptor)           = 0;
};

class ISubscriber : public virtual control_plane::ISubscriptionService,
                    public srf::node::SourceChannelWriteable<srf::runtime::RemoteDescriptor>
{
  public:
    ~ISubscriber() override = default;
};

}  // namespace srf::pubsub
