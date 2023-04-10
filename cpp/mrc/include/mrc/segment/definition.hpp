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

#include "mrc/engine/segment/idefinition.hpp"
#include "mrc/segment/initializers.hpp"
#include "mrc/utils/macros.hpp"

#include <map>
#include <memory>
#include <string>

namespace mrc::segment {
struct EgressPortsBase;
struct IngressPortsBase;

class Definition final : public internal::segment::IDefinition
{
  private:
    Definition(std::string name,
               std::map<std::string, ingress_initializer_t> ingress_ports,
               std::map<std::string, egress_initializer_t> egress_ports,
               backend_initializer_fn_t backend_initializer);

    static std::shared_ptr<Definition> create(std::string name,
                                              std::map<std::string, ingress_initializer_t> ingress_initializers,
                                              std::map<std::string, egress_initializer_t> egress_initializers,
                                              segment_initializer_fn_t initializer);

  public:
    ~Definition() final = default;

    DELETE_COPYABILITY(Definition);
    DELETE_MOVEABILITY(Definition);

    static std::shared_ptr<Definition> create(std::string name,
                                              IngressPortsBase ingress_ports,
                                              EgressPortsBase egress_ports,
                                              segment_initializer_fn_t initializer);

    static std::shared_ptr<Definition> create(std::string name,
                                              EgressPortsBase egress_ports,
                                              segment_initializer_fn_t initializer);

    static std::shared_ptr<Definition> create(std::string name,
                                              IngressPortsBase ingress_ports,
                                              segment_initializer_fn_t initializer);

    static std::shared_ptr<Definition> create(std::string name, segment_initializer_fn_t initializer);
};

}  // namespace mrc::segment
