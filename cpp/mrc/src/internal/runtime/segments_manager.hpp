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

#include "internal/runtime/runtime_provider.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <stop_token>

namespace mrc::control_plane::state {
struct SegmentInstance;
struct Worker;
}  // namespace mrc::control_plane::state
namespace mrc::segment {
class SegmentInstance;
}  // namespace mrc::segment

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
