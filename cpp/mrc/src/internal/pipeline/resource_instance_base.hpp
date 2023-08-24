/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstdint>
#include <map>
#include <memory>

namespace mrc::runtime {
class Runtime;
}
namespace mrc::manifold {
struct Interface;
}  // namespace mrc::manifold

namespace mrc::pipeline {
class ManifoldDefinition;

template <class ResourceT>
class ResourceInstanceBase : public AsyncService, public runnable::RunnableResourcesProvider
{
  public:
    ResourceInstanceBase(runtime::Runtime& runtime, uint64_t instance_id);
    ~ResourceInstanceBase() override;

  private:
    virtual void pre_service_start() {}
    void do_service_start(std::stop_token stop_token) final;
    virtual void post_service_start() {}

    virtual ResourceT filter_object(const control_plane::state::ControlPlaneState& state)

        void process_state_update(control_plane::state::SegmentInstance& instance);

    runtime::Runtime& m_runtime;

    std::shared_ptr<const ManifoldDefinition> m_definition;

    uint64_t m_instance_id;

    std::shared_ptr<manifold::Interface> m_interface;
};

}  // namespace mrc::pipeline
