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

#include "mrc/segment/definition.hpp"

#include "mrc/engine/segment/idefinition.hpp"  // for IDefinition
#include "mrc/segment/builder.hpp"             // for Builder
#include "mrc/segment/egress_ports.hpp"        // for EgressPortsBase
#include "mrc/segment/ingress_ports.hpp"       // for IngressPortsBase

#include <utility>  // for move
namespace mrc::segment {
class IBuilder;
}

namespace mrc::segment {

std::shared_ptr<Definition> Definition::create(std::string name,
                                               std::map<std::string, ingress_initializer_t> ingress_initializers,
                                               std::map<std::string, egress_initializer_t> egress_initializers,
                                               segment_initializer_fn_t initializer)
{
    return std::shared_ptr<Definition>(new Definition(std::move(name),
                                                      std::move(ingress_initializers),
                                                      std::move(egress_initializers),
                                                      [initializer](segment::IBuilder& backend) {
                                                          Builder builder(backend);
                                                          initializer(builder);
                                                      }));
}

Definition::Definition(std::string name,
                       std::map<std::string, ingress_initializer_t> ingress_ports,
                       std::map<std::string, egress_initializer_t> egress_ports,
                       backend_initializer_fn_t backend_initializer) :
  segment::IDefinition(std::move(name),
                       std::move(ingress_ports),
                       std::move(egress_ports),
                       std::move(backend_initializer))
{}

std::shared_ptr<Definition> Definition::create(std::string name,
                                               IngressPortsBase ingress_ports,
                                               EgressPortsBase egress_ports,
                                               segment_initializer_fn_t initializer)
{
    return Definition::create(std::move(name),
                              ingress_ports.m_initializers,
                              egress_ports.m_initializers,
                              std::move(initializer));
}

std::shared_ptr<Definition> Definition::create(std::string name,
                                               EgressPortsBase egress_ports,
                                               segment_initializer_fn_t initializer)
{
    return Definition::create(std::move(name), {}, egress_ports.m_initializers, std::move(initializer));
}

std::shared_ptr<Definition> Definition::create(std::string name,
                                               IngressPortsBase ingress_ports,
                                               segment_initializer_fn_t initializer)
{
    return Definition::create(std::move(name), ingress_ports.m_initializers, {}, std::move(initializer));
}

std::shared_ptr<Definition> Definition::create(std::string name, segment_initializer_fn_t initializer)
{
    return Definition::create(std::move(name),
                              std::map<std::string, ingress_initializer_t>{},
                              std::map<std::string, egress_initializer_t>{},
                              std::move(initializer));
}

}  // namespace mrc::segment
