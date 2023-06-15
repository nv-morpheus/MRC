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

#include "internal/control_plane/state/root_state.hpp"
#include "internal/runtime/runtime_provider.hpp"

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

namespace mrc::segment {
class EgressPortBase;
class IngressPortBase;
}  // namespace mrc::segment

namespace mrc::pipeline {
class ManifoldDefinition;

class ManifoldInstance final : public AsyncService, public runtime::InternalRuntimeProvider
{
  public:
    ManifoldInstance(runtime::IInternalRuntimeProvider& runtime,
                     std::shared_ptr<const ManifoldDefinition> definition,
                     uint64_t instance_id);
    ~ManifoldInstance() override;

    void register_local_ingress(SegmentAddress address, std::shared_ptr<segment::IngressPortBase> ingress_port);
    void register_local_egress(SegmentAddress address, std::shared_ptr<segment::EgressPortBase> egress_port);

    void unregister_local_ingress(SegmentAddress address);
    void unregister_local_egress(SegmentAddress address);

    std::shared_ptr<manifold::Interface> get_interface() const;

  private:
    void do_service_start(std::stop_token stop_token) final;
    void process_state_update(control_plane::state::SegmentInstance& instance);

    std::shared_ptr<const ManifoldDefinition> m_definition;

    uint64_t m_instance_id;

    std::shared_ptr<manifold::Interface> m_interface;
};

}  // namespace mrc::pipeline
