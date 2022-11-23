/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/pipeline/resources.hpp"
#include "internal/segment/definition.hpp"

#include "mrc/engine/segment/ibuilder.hpp"
#include "mrc/runnable/forward.hpp"
#include "mrc/segment/forward.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace mrc::internal::segment {
class Builder final
{
  public:
    Builder(std::shared_ptr<const Definition> segdef,
            SegmentRank rank,
            pipeline::Resources& resources,
            std::size_t default_partition_id);

    const Definition& definition() const;

    const std::map<std::string, std::shared_ptr<mrc::runnable::Launchable>>& nodes() const;
    const std::map<std::string, std::shared_ptr<mrc::segment::EgressPortBase>>& egress_ports() const;
    const std::map<std::string, std::shared_ptr<mrc::segment::IngressPortBase>>& ingress_ports() const;

  private:
    const std::string& name() const;

    bool has_object(const std::string& name) const;
    ::mrc::segment::ObjectProperties& find_object(const std::string& name);

    void add_object(const std::string& name, std::shared_ptr<::mrc::segment::ObjectProperties> object);
    void add_runnable(const std::string& name, std::shared_ptr<mrc::runnable::Launchable> runnable);

    std::shared_ptr<::mrc::segment::IngressPortBase> get_ingress_base(const std::string& name);
    std::shared_ptr<::mrc::segment::EgressPortBase> get_egress_base(const std::string& name);

    // temporary metrics interface
    std::function<void(std::int64_t)> make_throughput_counter(const std::string& name);

    // definition
    std::shared_ptr<const Definition> m_definition;

    // all objects - ports, runnables, etc.
    std::map<std::string, std::shared_ptr<::mrc::segment::ObjectProperties>> m_objects;

    // only runnables
    std::map<std::string, std::shared_ptr<mrc::runnable::Launchable>> m_nodes;

    // ingress/egress - these are also nodes/objects
    std::map<std::string, std::shared_ptr<::mrc::segment::IngressPortBase>> m_ingress_ports;
    std::map<std::string, std::shared_ptr<::mrc::segment::EgressPortBase>> m_egress_ports;

    pipeline::Resources& m_resources;
    const std::size_t m_default_partition_id;

    friend IBuilder;
};

}  // namespace mrc::internal::segment
