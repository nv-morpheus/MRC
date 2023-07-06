/**
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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/state/root_state.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/resource_manager_base.hpp"
#include "internal/runtime/runtime_provider.hpp"
#include "internal/segment/segment_instance.hpp"
#include "internal/ucx/ucx_resources.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <optional>

namespace mrc::memory {
class DeviceResources;
class HostResources;
}  // namespace mrc::memory
namespace mrc::network {
class NetworkResources;
}  // namespace mrc::network
namespace mrc::runnable {
class RunnableResources;
}  // namespace mrc::runnable

namespace mrc::runtime {

/**
 * @brief Partition Resources define the set of Resources available to a given Partition
 *
 * This class does not own the actual resources, that honor is bestowed on the resources::Manager. This class is
 * constructed and owned by the resources::Manager to ensure validity of the references.
 */
class SegmentsManager : public AsyncService, public InternalPartitionRuntimeProvider
{
  public:
    SegmentsManager(runtime::IInternalPartitionRuntimeProvider& runtime, size_t partition_id);
    ~SegmentsManager() override;

    bool sync_state(const control_plane::state::Worker& worker);

  private:
    void do_service_start(std::stop_token stop_token) override;

    void create_segment(const control_plane::state::SegmentInstance& instance);
    void erase_segment(SegmentAddress address);

    // Running segment instances
    std::map<SegmentAddress, std::shared_ptr<segment::SegmentInstance>> m_instances;
};

}  // namespace mrc::runtime
