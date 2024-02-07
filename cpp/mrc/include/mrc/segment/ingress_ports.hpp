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

#include "mrc/node/port_builders.hpp"
#include "mrc/segment/ingress_port.hpp"
#include "mrc/segment/ports.hpp"

#include <array>
#include <cstddef>
#include <typeinfo>
#include <utility>

namespace mrc::segment {

struct IngressPortsBase : public Ports<IngressPortBase>
{
    using Ports<IngressPortBase>::port_info_t;
    using Ports<IngressPortBase>::Ports;
};

// Derive from AutoRegEgressPort so we register the initializers for this type on creation
template <typename T>
struct IngressPortInfo : public IngressPortsBase::port_info_t, public node::AutoRegIngressPort<T>
{
    using IngressPortsBase::port_info_t::PortInfo;
};

template <typename... TypesT>
struct IngressPorts : public IngressPortsBase
{
    using IngressPortsBase::port_info_t;

    static constexpr size_t Count = sizeof...(TypesT);

    IngressPorts(std::array<std::string, Count> names) :
      IngressPortsBase(get_infos(names, std::index_sequence_for<TypesT...>{}))
    {}

  private:
    template <std::size_t... Is>
    static std::vector<std::shared_ptr<port_info_t>> get_infos(const std::array<std::string, Count>& names,
                                                               std::index_sequence<Is...> _)
    {
        std::vector<std::shared_ptr<port_info_t>> infos;
        (infos.push_back(std::make_shared<IngressPortInfo<TypesT>>(
             names[Is],
             std::type_index(typeid(TypesT)),
             [](const SegmentAddress& address, const PortName& name) {
                 return std::make_shared<IngressPort<TypesT>>(SegmentAddress2(address), name);
             })),
         ...);

        return infos;
    }
};

}  // namespace mrc::segment
