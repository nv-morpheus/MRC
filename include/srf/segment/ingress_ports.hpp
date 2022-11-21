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

#include "srf/segment/ingress_port.hpp"
#include "srf/segment/ports.hpp"

namespace srf::segment {

struct IngressPortsBase : public Ports<IngressPortBase>
{
    using Ports<IngressPortBase>::Ports;
};

template <typename... TypesT>
struct IngressPorts : public IngressPortsBase
{
    using port_builder_fn_t = typename IngressPortsBase::port_builder_fn_t;

    IngressPorts(std::vector<std::string> names) : IngressPortsBase(std::move(names), get_builders()) {}

  private:
    static std::vector<port_builder_fn_t> get_builders()
    {
        std::vector<port_builder_fn_t> builders;
        (builders.push_back([](const SegmentAddress& address, const PortName& name) {
            return std::make_shared<IngressPort<TypesT>>(address, name);
        }),
         ...);

        return builders;
    }
};

}  // namespace srf::segment
