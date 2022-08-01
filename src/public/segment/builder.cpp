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

#include "srf/segment/builder.hpp"

#include "srf/node/port_registry.hpp"

namespace srf::segment {
std::shared_ptr<ObjectProperties> Builder::get_ingress(std::string name, std::type_index type_index) {
    auto base = m_backend.get_ingress_base(name);
    if (!base)
    {
        throw exceptions::SrfRuntimeError("Egress port name not found: " + name);
    }

    auto port_util = node::PortRegistry::find_port_util(type_index);
    auto port = port_util->try_cast_ingress_base_to_object(base);
    if (port == nullptr)
    {
        throw exceptions::SrfRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

std::shared_ptr<ObjectProperties> Builder::get_egress(std::string name, std::type_index type_index) {
    auto base = m_backend.get_egress_base(name);
    if (!base)
    {
        throw exceptions::SrfRuntimeError("Egress port name not found: " + name);
    }

    auto port_util = node::PortRegistry::find_port_util(type_index);

    auto port = port_util->try_cast_egress_base_to_object(base);
    if (port == nullptr)
    {
        throw exceptions::SrfRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}
}
