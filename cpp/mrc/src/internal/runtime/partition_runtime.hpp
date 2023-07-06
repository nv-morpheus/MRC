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

#include "internal/remote_descriptor/manager.hpp"
#include "internal/runtime/runtime_provider.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/runtime/api.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>
#include <string>

namespace mrc::codable {
struct ICodableStorage;
}  // namespace mrc::codable
namespace mrc::resources {
class PartitionResources;
}  // namespace mrc::resources
namespace mrc::pubsub {
class IPublisherService;
class ISubscriberService;
enum class PublisherPolicy;
}  // namespace mrc::pubsub

namespace mrc::control_plane {
class Client;
}
namespace mrc::metrics {
class Registry;
}

namespace mrc::runtime {

class Runtime;
class PipelinesManager;
class WorkerManager;
class DataPlaneSystemManager;

class PartitionRuntime final : public mrc::runtime::IPartitionRuntime,
                               public AsyncService,
                               public IInternalPartitionRuntime,
                               public IInternalPartitionRuntimeProvider
{
  public:
    PartitionRuntime(Runtime& system_runtime, size_t partition_id);
    ~PartitionRuntime() final;

    DELETE_COPYABILITY(PartitionRuntime);
    DELETE_MOVEABILITY(PartitionRuntime);

    size_t partition_id() const override;

    std::size_t gpu_count() const override;

    resources::PartitionResources& resources();

    runnable::RunnableResources& runnable() override;

    control_plane::Client& control_plane() const override;

    DataPlaneSystemManager& data_plane() const override;

    PipelinesManager& pipelines_manager() const override;

    metrics::Registry& metrics_registry() const override;

    IInternalPartitionRuntime& runtime() override;

    // IPartition -> IRemoteDescriptorManager& is covariant
    remote_descriptor::Manager& remote_descriptor_manager() final;

    std::unique_ptr<mrc::codable::ICodableStorage> make_codable_storage() final;

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

    std::shared_ptr<WorkerManager> m_worker_manager;
};

}  // namespace mrc::runtime
