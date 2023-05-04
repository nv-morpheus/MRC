/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/async_service.hpp"
#include "internal/remote_descriptor/manager.hpp"

#include "mrc/runtime/api.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>
#include <string>

namespace mrc::codable {
struct ICodableStorage;
}  // namespace mrc::codable
namespace mrc::internal::resources {
class PartitionResources;
}  // namespace mrc::internal::resources
namespace mrc::pubsub {
class IPublisherService;
class ISubscriberService;
enum class PublisherPolicy;
}  // namespace mrc::pubsub

namespace mrc::internal::control_plane {
class Client;
}
namespace mrc::metrics {
class Registry;
}

namespace mrc::internal::runtime {

class Runtime;
class PipelinesManager;

class PartitionRuntime final : public mrc::runtime::IPartitionRuntime, public AsyncService
{
  public:
    PartitionRuntime(Runtime& system_runtime, size_t partition_id);
    ~PartitionRuntime() final;

    DELETE_COPYABILITY(PartitionRuntime);
    DELETE_MOVEABILITY(PartitionRuntime);

    size_t partition_id() const;

    resources::PartitionResources& resources();

    control_plane::Client& control_plane() const;

    PipelinesManager& pipelines_manager() const;

    metrics::Registry& metrics_registry() const;

    // IPartition -> IRemoteDescriptorManager& is covariant
    remote_descriptor::Manager& remote_descriptor_manager() final;

    std::unique_ptr<mrc::codable::ICodableStorage> make_codable_storage() final;

  protected:
    runnable::RunnableResources& runnable() override;

  private:
    void do_service_start(std::stop_token stop_token) final;

    std::shared_ptr<mrc::pubsub::IPublisherService> make_publisher_service(
        const std::string& name,
        const mrc::pubsub::PublisherPolicy& policy) final;

    std::shared_ptr<mrc::pubsub::ISubscriberService> make_subscriber_service(const std::string& name) final;

    Runtime& m_system_runtime;
    size_t m_partition_id;

    resources::PartitionResources& m_resources;
    std::shared_ptr<remote_descriptor::Manager> m_remote_descriptor_manager;
};

}  // namespace mrc::internal::runtime
