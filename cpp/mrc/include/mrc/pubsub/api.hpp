/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/codable/api.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/control_plane/api.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/runtime/remote_descriptor.hpp"

#include <string>

namespace mrc::pubsub {

enum class PublisherPolicy
{
    Broadcast,
    RoundRobin,
};

class IPublisherService : public virtual control_plane::ISubscriptionService,
                          public node::ReadableAcceptor<std::unique_ptr<codable::EncodedStorage>>
{
  public:
    ~IPublisherService() override = default;

    virtual std::unique_ptr<codable::ICodableStorage> create_storage() = 0;
};

class ISubscriberService : public virtual control_plane::ISubscriptionService,
                           public node::WritableAcceptor<mrc::runtime::RemoteDescriptor>
{
  public:
    ~ISubscriberService() override = default;
};

}  // namespace mrc::pubsub
