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

#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"

#include "mrc/codable/api.hpp"
#include "mrc/pubsub/api.hpp"
#include "mrc/runtime/api.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>
#include <string>

namespace mrc::internal::runtime {

class Partition final : public mrc::runtime::IPartition
{
  public:
    Partition(resources::PartitionResources& resources);
    ~Partition() final;

    DELETE_COPYABILITY(Partition);
    DELETE_MOVEABILITY(Partition);

    resources::PartitionResources& resources();

    // IPartition -> IRemoteDescriptorManager& is covariant
    remote_descriptor::Manager& remote_descriptor_manager() final;

    std::unique_ptr<mrc::codable::ICodableStorage> make_codable_storage() final;

  private:
    std::shared_ptr<mrc::pubsub::IPublisherService> make_publisher_service(
        const std::string& name, const mrc::pubsub::PublisherPolicy& policy) final;

    std::shared_ptr<mrc::pubsub::ISubscriberService> make_subscriber_service(const std::string& name) final;

    resources::PartitionResources& m_resources;
    std::shared_ptr<remote_descriptor::Manager> m_remote_descriptor_manager;
};

}  // namespace mrc::internal::runtime
